# SPDX-License-Identifier: Apache-2.0
"""SGLang LMCache integration — multi-process mode.

Implements :class:`LMCacheMPConnector` and :class:`LMCacheMPLayerwiseConnector`
— drop-in replacements for the in-process
:class:`~lmcache.integration.sglang.sglang_adapter.LMCacheConnector` and
:class:`~lmcache.integration.sglang.sglang_adapter.LMCacheLayerwiseConnector`
that route operations through ZMQ to a standalone LMCache server.

Modeled on the TRT-LLM multi-process adapter
(:mod:`lmcache.integration.tensorrt_llm.tensorrt_mp_adapter`):

- KV pool tensors are shared with the server via :class:`CudaIPCWrapper`.
  SGLang allocates per-layer K and V tensors through PyTorch's caching
  allocator, so the standard wrapper (rather than ``RawCudaIPCWrapper``)
  works.
- The wire protocol is the unmodified ``REGISTER_KV_CACHE`` / ``LOOKUP``
  + ``QUERY_PREFETCH_STATUS`` / ``RETRIEVE`` / ``STORE`` /
  ``FREE_LOOKUP_LOCKS`` / ``END_SESSION`` flow used by vLLM and TRT-LLM.
- ``layout_hints`` carry SGLang-specific shape information
  (``sglang_attention``, ``sglang_num_layers``) so the server's
  :func:`~lmcache.v1.gpu_connector.utils.normalize_kv_and_discover_format`
  can split the flat IPC payload back into the nested form the format
  detector recognizes.
"""

# Standard
from dataclasses import dataclass
from typing import Iterable, List, Optional
import os

# Third Party
from sglang.srt.configs.model_config import ModelConfig
import torch
import torch.distributed as dist
import zmq

# First Party
from lmcache.integration.sglang.sglang_adapter import (
    LoadMetadata,
    StoreMetadata,
)
from lmcache.logging import init_logger
from lmcache.utils import CacheStoreEvent, EngineType
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheEngineKey,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

logger = init_logger(__name__)

DEFAULT_SERVER_URL = "ipc:///tmp/lmcache.sock"
DEFAULT_MQ_TIMEOUT: float = 300.0


def _get_server_url(explicit: Optional[str] = None) -> str:
    """Resolve the server URL: explicit > env var > default."""
    if explicit is not None:
        return explicit
    return os.environ.get("LMCACHE_SERVER_URL", DEFAULT_SERVER_URL)


def _send_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list,
) -> MessagingFuture:
    return mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )


def _wrap_sglang_kv_caches(
    k_pool: List[torch.Tensor],
    v_pool: List[torch.Tensor],
    is_mla: bool,
) -> List[CudaIPCWrapper]:
    """Wrap SGLang's per-layer K/V pool tensors for IPC.

    SGLang's wire payload is a flat ``list[CudaIPCWrapper]``. The server
    reconstructs the nested per-engine structure from
    ``layout_hints["sglang_attention"]`` (see
    :func:`~lmcache.v1.gpu_connector.utils.normalize_kv_and_discover_format`).

    For MHA, K tensors come first (one per layer), then V tensors. For
    MLA, only the K-pool is sent — V is degenerate.
    """
    if is_mla:
        return [CudaIPCWrapper(t) for t in k_pool]
    return [CudaIPCWrapper(t) for t in k_pool] + [CudaIPCWrapper(t) for t in v_pool]


@dataclass
class _PendingLookup:
    """Per-request lookup state retained between scheduler and worker calls."""

    cached_tokens: int
    matched_lookup_submitted: bool


class LMCacheMPConnector:
    """SGLang LMCache MP connector — drop-in for ``LMCacheConnector``.

    Mirrors the public API of
    :class:`~lmcache.integration.sglang.sglang_adapter.LMCacheConnector` —
    ``load_kv``, ``store_kv``, ``chunk_size``, ``reset``, ``close`` —
    but routes every operation through ZMQ to a standalone LMCache server
    instead of an in-process engine.

    The KV pool is shared via CUDA IPC at construction time
    (``REGISTER_KV_CACHE``) and unregistered in :meth:`close`.

    Args:
        sgl_config: SGLang ``ModelConfig`` used for shape derivation.
        tp_size: Tensor parallel size (treated as the LMCache world
            size — pipeline parallelism is not modeled separately).
        rank: Global tensor-parallel rank for this worker.
        k_pool: Per-layer K tensors. Length == ``num_hidden_layers``.
        v_pool: Per-layer V tensors. Same length and shape pattern as
            ``k_pool``. For MLA models, the per-layer V tensors share
            storage with K and are still passed for API symmetry.
        server_url: ZMQ endpoint of the LMCache server. Falls back to
            ``$LMCACHE_SERVER_URL`` then to a default IPC socket.
        mq_timeout: Per-request timeout for blocking MQ calls.
    """

    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        server_url: Optional[str] = None,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
    ) -> None:
        if not k_pool:
            raise ValueError("k_pool cannot be empty during initialization.")
        if len(k_pool) != len(v_pool):
            raise ValueError(
                f"k_pool and v_pool must have the same length; got "
                f"{len(k_pool)} vs {len(v_pool)}"
            )

        self.sgl_config = sgl_config
        self.tp_size = tp_size
        self.world_size = tp_size
        self.global_rank = rank

        if k_pool[0].is_cuda and k_pool[0].device.index is not None:
            self.local_rank = k_pool[0].device.index
        else:
            self.local_rank = rank

        self.num_layers = sgl_config.num_hidden_layers
        self.is_mla = bool(getattr(sgl_config, "is_mla", False))

        self._mq_timeout = float(os.environ.get("LMCACHE_MQ_TIMEOUT", mq_timeout))
        self._server_url = _get_server_url(server_url)
        # Share the process-global ZMQ context — multiple connectors can
        # safely coexist (one per scheduler/tp-rank), and any client
        # close() is idempotent against the singleton.
        self._zmq_context = zmq.Context.instance()
        self._mq_client = MessageQueueClient(self._server_url, self._zmq_context)

        future = _send_request(self._mq_client, RequestType.GET_CHUNK_SIZE, [])
        self._chunk_size: int = future.result(timeout=self._mq_timeout)

        self._instance_id = os.getpid()
        self._registered = False
        self._model_name = str(getattr(sgl_config, "model_path", "unknown_model"))

        self._pending_lookups: dict[str, _PendingLookup] = {}

        self._register_kv_caches(k_pool, v_pool)

        logger.info(
            "LMCache SGLang MP connector: connected to %s "
            "(chunk_size=%d, world_size=%d, rank=%d, num_layers=%d, "
            "attention=%s)",
            self._server_url,
            self._chunk_size,
            self.world_size,
            self.global_rank,
            self.num_layers,
            "MLA" if self.is_mla else "MHA",
        )

    def _register_kv_caches(
        self,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
    ) -> None:
        """Send REGISTER_KV_CACHE with SGLang layout hints."""
        wrapped = _wrap_sglang_kv_caches(k_pool, v_pool, self.is_mla)

        layout_hints = {
            "sglang_attention": "MLA" if self.is_mla else "MHA",
            "sglang_num_layers": self.num_layers,
        }

        future = _send_request(
            self._mq_client,
            RequestType.REGISTER_KV_CACHE,
            [
                self._instance_id,
                wrapped,
                self._model_name,
                self.world_size,
                EngineType.SGLANG,
                layout_hints,
            ],
        )
        try:
            future.result(timeout=self._mq_timeout)
            self._registered = True
            ref = k_pool[0] if self.is_mla else k_pool[0]
            logger.info(
                "LMCache SGLang MP: registered KV caches (instance=%d, "
                "tensor_shape=%s, num_layers=%d, attention=%s)",
                self._instance_id,
                list(ref.shape),
                self.num_layers,
                "MLA" if self.is_mla else "MHA",
            )
        except TimeoutError:
            logger.error(
                "LMCache SGLang MP: KV cache registration timed out after %ss",
                self._mq_timeout,
            )
            raise

    def _create_key(
        self,
        token_ids: List[int],
        request_id: str,
        worker_id: Optional[int],
    ) -> IPCCacheEngineKey:
        aligned_end = (len(token_ids) // self._chunk_size) * self._chunk_size
        return IPCCacheEngineKey(
            model_name=self._model_name,
            world_size=self.world_size,
            worker_id=worker_id,
            token_ids=tuple(token_ids),
            start=0,
            end=aligned_end,
            request_id=request_id,
        )

    def chunk_size(self) -> int:
        """Return the LMCache chunk size (tokens per cached chunk)."""
        return self._chunk_size

    def load_kv(self, load_metadata: LoadMetadata) -> int:
        """Retrieve cached KV blocks from the LMCache server.

        Performs ``LOOKUP`` + ``QUERY_PREFETCH_STATUS`` to learn how many
        tokens the server has cached, then issues a ``RETRIEVE`` for the
        full prefix before returning. Synchronous: by the time this
        returns, the GPU KV slots covered by ``slot_mapping`` have been
        populated and the server's stream has been waited on via the
        IPC event round-trip.

        Args:
            load_metadata: Token IDs, slot mapping, and offset describing
                the prefix to load. ``slot_mapping`` indexes into the
                full per-token slot space (block_size=1 convention on
                the server side).

        Returns:
            Number of tokens that were retrieved from LMCache (will be
            <= ``len(load_metadata.token_ids) - load_metadata.offset``).
        """
        token_ids = list(load_metadata.token_ids)
        offset = load_metadata.offset
        request_id = f"sglang-load-{self._instance_id}-{id(load_metadata):x}"

        if len(token_ids) < self._chunk_size:
            return 0

        cached_tokens = self._submit_lookup(token_ids, request_id)
        if cached_tokens <= offset:
            self._free_lookup_locks(token_ids, 0, cached_tokens, request_id)
            self._end_session(request_id)
            return 0

        retrieve_end = (cached_tokens // self._chunk_size) * self._chunk_size
        if retrieve_end <= offset:
            self._free_lookup_locks(token_ids, 0, retrieve_end, request_id)
            self._end_session(request_id)
            return 0

        # Slot mapping covers the *uncached* suffix on the SGLang side
        # (length == len(token_ids) - offset). The server expects block
        # IDs aligned to the retrieve range [0, retrieve_end). Pad the
        # prefix the engine has already computed with -1 sentinels and
        # truncate to the retrieve range.
        slot_mapping_cpu = load_metadata.slot_mapping.detach().to("cpu")
        full_slots = torch.full((len(token_ids),), -1, dtype=torch.int64)
        full_slots[offset : offset + slot_mapping_cpu.numel()] = slot_mapping_cpu.to(
            torch.int64
        )
        block_ids = full_slots[:retrieve_end].tolist()

        # Skip the prefix the engine already has so we don't overwrite
        # GPU blocks shared with concurrent requests.
        skip_first_n_tokens = offset

        event = torch.cuda.Event(interprocess=True)
        event.record()

        retrieve_key = self._create_key(
            token_ids[:retrieve_end], request_id, worker_id=self.global_rank
        )

        try:
            _send_request(
                self._mq_client,
                RequestType.RETRIEVE,
                [
                    retrieve_key,
                    self._instance_id,
                    block_ids,
                    event.ipc_handle(),
                    skip_first_n_tokens,
                ],
            ).result(timeout=self._mq_timeout)
        except Exception as e:
            logger.warning(
                "LMCache SGLang MP: retrieve failed for req %s: %s",
                request_id,
                e,
            )
            self._end_session(request_id)
            return 0

        self._end_session(request_id)
        return retrieve_end - offset

    def store_kv(self, store_metadata: StoreMetadata) -> None:
        """Store newly-computed KV blocks to the LMCache server."""
        token_ids = list(store_metadata.token_ids)
        if len(token_ids) < self._chunk_size:
            return

        request_id = f"sglang-store-{self._instance_id}-{id(store_metadata):x}"

        block_ids = (
            store_metadata.kv_indices.detach().to("cpu").to(torch.int64).tolist()
        )

        event = torch.cuda.Event(interprocess=True)
        event.record()

        key = self._create_key(token_ids, request_id, worker_id=self.global_rank)

        try:
            _send_request(
                self._mq_client,
                RequestType.STORE,
                [key, self._instance_id, block_ids, event.ipc_handle()],
            ).result(timeout=self._mq_timeout)
        except Exception as e:
            logger.warning(
                "LMCache SGLang MP: store failed for req %s: %s",
                request_id,
                e,
            )
        finally:
            self._end_session(request_id)

    def _submit_lookup(self, token_ids: List[int], request_id: str) -> int:
        """LOOKUP + QUERY_PREFETCH_STATUS — return the cached token count."""
        key = self._create_key(token_ids, request_id, worker_id=None)
        try:
            _send_request(
                self._mq_client, RequestType.LOOKUP, [key, self.tp_size]
            ).result(timeout=self._mq_timeout)
            result = _send_request(
                self._mq_client,
                RequestType.QUERY_PREFETCH_STATUS,
                [request_id],
            ).result(timeout=self._mq_timeout)
        except Exception as e:
            logger.warning("LMCache SGLang MP: lookup failed: %s", e)
            return 0
        if result is None:
            return 0
        return result * self._chunk_size

    def _free_lookup_locks(
        self, token_ids: List[int], start: int, end: int, request_id: str
    ) -> None:
        if end <= start:
            return
        aligned_end = (end // self._chunk_size) * self._chunk_size
        if aligned_end <= start:
            return
        key = IPCCacheEngineKey(
            model_name=self._model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=aligned_end,
            request_id=request_id,
        )
        try:
            _send_request(
                self._mq_client, RequestType.FREE_LOOKUP_LOCKS, [key, self.tp_size]
            )
        except Exception as e:
            logger.warning("LMCache SGLang MP: free_lookup_locks failed: %s", e)

    def _end_session(self, request_id: str) -> None:
        try:
            _send_request(self._mq_client, RequestType.END_SESSION, [request_id])
        except Exception as e:
            logger.debug("LMCache SGLang MP: end_session failed: %s", e)

    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        """Cache-store events are emitted only by the in-process engine."""
        return []

    def reset(self) -> None:
        """No-op — cache eviction is owned by the standalone server."""
        return None

    def close(self) -> None:
        """Unregister the KV cache and close the ZMQ client."""
        if self._registered:
            try:
                _send_request(
                    self._mq_client,
                    RequestType.UNREGISTER_KV_CACHE,
                    [self._instance_id],
                ).result(timeout=self._mq_timeout)
            except Exception as e:
                logger.warning("LMCache SGLang MP: unregister failed: %s", e)
            self._registered = False
        try:
            self._mq_client.close()
        except Exception:
            pass


class LMCacheMPLayerwiseConnector(LMCacheMPConnector):
    """Drop-in for :class:`~lmcache.integration.sglang.sglang_adapter.\
LMCacheLayerwiseConnector` over the multi-process protocol.

    SGLang's :class:`LMCRadixCache` calls ``start_load_kv`` followed by
    ``load_kv_layerwise(layer_id)`` for each layer. The MP server
    transfers all layers in a single kernel invocation, so:

    - :meth:`start_load_kv` performs the full retrieve synchronously and
      returns the token count actually populated.
    - :meth:`load_kv_layerwise` is a no-op — the data is already on the
      GPU by the time SGLang asks per-layer.

    The store side stays compatible with
    :class:`LMCacheLayerwiseConnector.store_kv`, which is also
    synchronous in the in-process layerwise path.
    """

    def __init__(
        self,
        sgl_config: ModelConfig,
        tp_size: int,
        rank: int,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        server_url: Optional[str] = None,
        mq_timeout: float = DEFAULT_MQ_TIMEOUT,
    ) -> None:
        super().__init__(
            sgl_config=sgl_config,
            tp_size=tp_size,
            rank=rank,
            k_pool=k_pool,
            v_pool=v_pool,
            server_url=server_url,
            mq_timeout=mq_timeout,
        )
        self.tp_group = tp_group

    def _global_min_tokens(self, local_tokens: int) -> int:
        if self.tp_size == 1 or self.tp_group is None:
            return local_tokens
        device = torch.device(f"cuda:{self.local_rank}")
        t = torch.tensor([local_tokens], dtype=torch.int32, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN, group=self.tp_group)
        return int(t.item())

    def start_load_kv(self, load_metadata: LoadMetadata) -> int:
        """Synchronous full retrieve — returns retrieved token count.

        Mirrors :meth:`LMCacheLayerwiseConnector.start_load_kv` but
        completes the entire load before returning so the per-layer
        pipeline doesn't need a server-side notification channel.
        """
        local_retrieved = self.load_kv(load_metadata)
        return self._global_min_tokens(local_retrieved)

    def load_kv_layerwise(self, layer_id: int) -> None:
        """No-op — :meth:`start_load_kv` already populated every layer."""
        return None
