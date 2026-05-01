# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for the SGLang LMCache MP adapter.

Spins up a real LMCache MP server in a subprocess, registers a
SGLang-shaped KV cache (per-layer K and V pool tensors via
:class:`CudaIPCWrapper`), then exercises the same store-then-retrieve
flow ``LMCacheMPConnector`` runs end-to-end via the public protocol.

This avoids importing :mod:`sglang` at module scope so the rest of the
LMCache test suite still passes in environments where SGLang is not
installed; the SGLang import is gated on the test running.
"""

# Standard
from typing import Any
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper, IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _has_sglang() -> bool:
    try:
        # Third Party
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


def _stub_sglang_module() -> None:
    """Insert a minimal stub for ``sglang.srt.configs.model_config`` so the
    MP adapter can be imported without the real SGLang wheel.

    The MP connector only references ``ModelConfig`` for a type annotation
    and reads attributes off the instance the caller passes — we never
    construct a ``ModelConfig`` here, so the stub is type-shape only.
    """
    if "sglang" in sys.modules:
        return
    # Standard
    import types

    pkg = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    configs = types.ModuleType("sglang.srt.configs")
    model_config = types.ModuleType("sglang.srt.configs.model_config")

    class _ModelConfig:  # placeholder
        pass

    model_config.ModelConfig = _ModelConfig  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "sglang": pkg,
            "sglang.srt": srt,
            "sglang.srt.configs": configs,
            "sglang.srt.configs.model_config": model_config,
        }
    )


def _free_port() -> int:
    """Pick a free TCP port; brief race window vs server bind is acceptable here."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(server_url: str, timeout: float = 30.0) -> None:
    """Poll PING until the server responds or timeout."""
    # Third Party
    import zmq

    ctx = zmq.Context.instance()
    client = MessageQueueClient(server_url, ctx)
    deadline = time.time() + timeout
    last_err: Exception | None = None
    try:
        while time.time() < deadline:
            try:
                future = client.submit_request(
                    RequestType.PING, [], get_response_class(RequestType.PING)
                )
                ok = future.result(timeout=2.0)
                if ok:
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(0.5)
        raise TimeoutError(
            f"LMCache server did not become ready at {server_url} "
            f"within {timeout}s (last error: {last_err})"
        )
    finally:
        client.close()


@pytest.fixture
def lmcache_server():
    """Start an LMCache MP server in a subprocess and yield its URL."""
    port = _free_port()
    server_url = f"tcp://127.0.0.1:{port}"

    env = os.environ.copy()
    # Make sure libc10_cuda.so / libtorch_cuda.so are reachable from the
    # subprocess loader. ``import torch`` lazy-loads them, but
    # ``import lmcache.c_ops`` needs the symbols up-front, so prepend
    # torch's lib dir to ``LD_LIBRARY_PATH``.
    try:
        # Third Party
        import torch  # noqa: F401

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        env["LD_LIBRARY_PATH"] = (
            torch_lib + ":" + env.get("LD_LIBRARY_PATH", "")
        ).rstrip(":")
    except ImportError:
        pass

    log_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="lmcache_server_", suffix=".log", delete=False
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.multiprocess.server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--chunk-size",
            "128",
            "--l1-size-gb",
            "1",
            "--eviction-policy",
            "LRU",
            "--max-gpu-workers",
            "2",
            "--max-cpu-workers",
            "2",
            "--disable-observability",
        ],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        _wait_for_server(server_url, timeout=60.0)
        yield server_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log_file.close()
        # Surface the server log on test failure so future debugging is
        # cheap. ``conftest`` -captured stderr is shown by pytest.
        with open(log_file.name) as f:
            print(f"--- LMCache server log ({log_file.name}) ---")
            print(f.read())
        os.unlink(log_file.name)


def _make_sglang_mha_pools(
    num_layers: int, page_buffer_size: int, num_kv_heads: int, head_dim: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Allocate SGLang-shaped MHA KV pools on the current CUDA device."""
    device = torch.device("cuda:0")
    shape = (page_buffer_size, num_kv_heads, head_dim)
    k_pool = [
        torch.randn(*shape, dtype=torch.bfloat16, device=device)
        for _ in range(num_layers)
    ]
    v_pool = [
        torch.randn(*shape, dtype=torch.bfloat16, device=device)
        for _ in range(num_layers)
    ]
    return k_pool, v_pool


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for E2E test")
def test_sglang_mp_store_then_retrieve_mha(lmcache_server: str) -> None:
    """Register MHA KV pool, STORE one chunk, RETRIEVE it back, verify match."""
    # Third Party
    import zmq

    NL = 4
    PBS = 1024
    NH = 4
    HS = 64
    chunk_size = 128

    k_pool, v_pool = _make_sglang_mha_pools(NL, PBS, NH, HS)

    # Wrap and register.
    wrapped = [CudaIPCWrapper(t) for t in k_pool] + [CudaIPCWrapper(t) for t in v_pool]
    instance_id = os.getpid()
    model_name = "test-sglang-mp"
    world_size = 1
    layout_hints: dict[str, Any] = {
        "sglang_attention": "MHA",
        "sglang_num_layers": NL,
    }

    ctx = zmq.Context.instance()
    client = MessageQueueClient(lmcache_server, ctx)
    try:

        def _send(req_type: RequestType, payloads: list) -> Any:
            return client.submit_request(
                req_type, payloads, get_response_class(req_type)
            ).result(timeout=30.0)

        cs = _send(RequestType.GET_CHUNK_SIZE, [])
        assert cs == chunk_size

        _send(
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrapped,
                model_name,
                world_size,
                EngineType.SGLANG,
                layout_hints,
            ],
        )

        # Build a chunk-aligned token sequence and a slot mapping that
        # writes to slots [0..chunk_size).
        token_ids = list(range(chunk_size))
        request_id = f"e2e-{uuid.uuid4().hex[:8]}"
        block_ids = list(range(chunk_size))  # block_size=1 → slot indices

        # Snapshot the K-pool slots we're about to write before STORE so
        # we can verify RETRIEVE actually wrote them back.
        original_k = [t[:chunk_size].clone() for t in k_pool]
        original_v = [t[:chunk_size].clone() for t in v_pool]

        store_event = torch.cuda.Event(interprocess=True)
        store_event.record()
        store_key = IPCCacheEngineKey(
            model_name=model_name,
            world_size=world_size,
            worker_id=0,
            token_ids=tuple(token_ids),
            start=0,
            end=chunk_size,
            request_id=request_id,
        )
        ipc_handle, ok = _send(
            RequestType.STORE,
            [store_key, instance_id, block_ids, store_event.ipc_handle()],
        )
        assert ok, "STORE returned ok=False"
        # Wait for the server's store stream to finish on this side.
        torch.cuda.Event.from_ipc_handle(k_pool[0].device, ipc_handle).wait()
        torch.cuda.synchronize()

        # Lookup must report exactly one cached chunk for this prefix.
        lookup_key = store_key.no_worker_id_version()
        _send(RequestType.LOOKUP, [lookup_key, world_size])
        cached_chunks = _send(RequestType.QUERY_PREFETCH_STATUS, [request_id])
        assert cached_chunks == 1, (
            f"expected 1 cached chunk after STORE, got {cached_chunks}"
        )

        # Zero-out the slots we'll RETRIEVE into, so a successful retrieve
        # has to *write* the cached values back.
        for t in k_pool:
            t[:chunk_size].zero_()
        for t in v_pool:
            t[:chunk_size].zero_()
        torch.cuda.synchronize()

        retrieve_event = torch.cuda.Event(interprocess=True)
        retrieve_event.record()
        ret_handle, ret_ok = _send(
            RequestType.RETRIEVE,
            [
                store_key,
                instance_id,
                block_ids,
                retrieve_event.ipc_handle(),
                0,  # skip_first_n_tokens
            ],
        )
        assert ret_ok, "RETRIEVE returned ok=False"
        torch.cuda.Event.from_ipc_handle(k_pool[0].device, ret_handle).wait()
        torch.cuda.synchronize()

        # Verify every layer's K and V were restored.
        for layer_idx, (k_now, k_orig) in enumerate(
            zip(k_pool, original_k, strict=True)
        ):
            assert torch.allclose(k_now[:chunk_size], k_orig), (
                f"K layer {layer_idx} not restored after RETRIEVE"
            )
        for layer_idx, (v_now, v_orig) in enumerate(
            zip(v_pool, original_v, strict=True)
        ):
            assert torch.allclose(v_now[:chunk_size], v_orig), (
                f"V layer {layer_idx} not restored after RETRIEVE"
            )

        # Cleanup.
        _send(RequestType.END_SESSION, [request_id])
        _send(RequestType.UNREGISTER_KV_CACHE, [instance_id])
    finally:
        client.close()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for E2E test")
def test_sglang_mp_store_then_retrieve_mla(lmcache_server: str) -> None:
    """Same store-then-retrieve flow but with the SGLang MLA shape.

    Validates the ``NL_X_NBBS_ONE_HS`` path: a single per-layer list of
    ``[PBS, 1, HS]`` tensors. The wire payload is depth-1; no MHA-style
    midpoint split is needed.
    """
    # Third Party
    import zmq

    NL = 4
    PBS = 1024
    HS = 128
    chunk_size = 128

    device = torch.device("cuda:0")
    k_pool = [
        torch.randn(PBS, 1, HS, dtype=torch.bfloat16, device=device) for _ in range(NL)
    ]

    wrapped = [CudaIPCWrapper(t) for t in k_pool]
    instance_id = os.getpid() ^ 0xCAFE  # avoid collision with MHA test
    model_name = "test-sglang-mp-mla"
    world_size = 1
    layout_hints: dict[str, Any] = {
        "sglang_attention": "MLA",
        "sglang_num_layers": NL,
    }

    ctx = zmq.Context.instance()
    client = MessageQueueClient(lmcache_server, ctx)
    try:

        def _send(req_type: RequestType, payloads: list) -> Any:
            return client.submit_request(
                req_type, payloads, get_response_class(req_type)
            ).result(timeout=30.0)

        _send(
            RequestType.REGISTER_KV_CACHE,
            [
                instance_id,
                wrapped,
                model_name,
                world_size,
                EngineType.SGLANG,
                layout_hints,
            ],
        )

        token_ids = list(range(chunk_size))
        request_id = f"e2e-mla-{uuid.uuid4().hex[:8]}"
        block_ids = list(range(chunk_size))
        original = [t[:chunk_size].clone() for t in k_pool]

        store_event = torch.cuda.Event(interprocess=True)
        store_event.record()
        store_key = IPCCacheEngineKey(
            model_name=model_name,
            world_size=world_size,
            worker_id=0,
            token_ids=tuple(token_ids),
            start=0,
            end=chunk_size,
            request_id=request_id,
        )
        ipc_handle, ok = _send(
            RequestType.STORE,
            [store_key, instance_id, block_ids, store_event.ipc_handle()],
        )
        assert ok
        torch.cuda.Event.from_ipc_handle(k_pool[0].device, ipc_handle).wait()
        torch.cuda.synchronize()

        # LOOKUP serves as a barrier — it waits until the L1 host-side
        # commit (``finish_write``) launched at the tail of STORE has
        # actually run, so the chunks are readable when RETRIEVE is
        # issued. Without this, RETRIEVE races the commit and gets
        # ``KEY_NOT_READABLE``.
        lookup_key = store_key.no_worker_id_version()
        _send(RequestType.LOOKUP, [lookup_key, world_size])
        cached_chunks = _send(RequestType.QUERY_PREFETCH_STATUS, [request_id])
        assert cached_chunks == 1, f"expected 1 cached chunk, got {cached_chunks}"

        for t in k_pool:
            t[:chunk_size].zero_()
        torch.cuda.synchronize()

        retrieve_event = torch.cuda.Event(interprocess=True)
        retrieve_event.record()
        ret_handle, ret_ok = _send(
            RequestType.RETRIEVE,
            [store_key, instance_id, block_ids, retrieve_event.ipc_handle(), 0],
        )
        assert ret_ok
        torch.cuda.Event.from_ipc_handle(k_pool[0].device, ret_handle).wait()
        torch.cuda.synchronize()

        for layer_idx, (now, orig) in enumerate(zip(k_pool, original, strict=True)):
            assert torch.allclose(now[:chunk_size], orig), (
                f"MLA layer {layer_idx} not restored after RETRIEVE"
            )

        _send(RequestType.END_SESSION, [request_id])
        _send(RequestType.UNREGISTER_KV_CACHE, [instance_id])
    finally:
        client.close()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA required for E2E test")
def test_sglang_mp_connector_class_e2e(lmcache_server: str) -> None:
    """Drive the full :class:`LMCacheMPConnector` API against a live server."""
    _stub_sglang_module()
    # First Party
    from lmcache.integration.sglang.sglang_adapter import LoadMetadata, StoreMetadata
    from lmcache.integration.sglang.sglang_mp_adapter import LMCacheMPConnector

    NL = 4
    PBS = 1024
    NH = 4
    HS = 64
    chunk_size = 128

    k_pool, v_pool = _make_sglang_mha_pools(NL, PBS, NH, HS)

    # SGLang's ModelConfig is heavy — fake the minimal interface our
    # connector touches.
    class _FakeModelConfig:
        num_hidden_layers = NL
        is_mla = False
        model_path = "test-sglang-mp"

    connector = LMCacheMPConnector(
        sgl_config=_FakeModelConfig(),
        tp_size=1,
        rank=0,
        k_pool=k_pool,
        v_pool=v_pool,
        server_url=lmcache_server,
    )
    try:
        assert connector.chunk_size() == chunk_size

        token_ids = list(range(chunk_size))
        slot_mapping = torch.arange(chunk_size, dtype=torch.int64)
        kv_indices = torch.arange(chunk_size, dtype=torch.int64)

        original_k = [t[:chunk_size].clone() for t in k_pool]
        original_v = [t[:chunk_size].clone() for t in v_pool]

        connector.store_kv(
            StoreMetadata(
                last_node=None,
                token_ids=token_ids,
                kv_indices=kv_indices,
                offset=0,
            )
        )

        for t in k_pool:
            t[:chunk_size].zero_()
        for t in v_pool:
            t[:chunk_size].zero_()
        torch.cuda.synchronize()

        num_loaded = connector.load_kv(
            LoadMetadata(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                offset=0,
            )
        )
        assert num_loaded == chunk_size, (
            f"expected to retrieve {chunk_size} tokens, got {num_loaded}"
        )

        for layer_idx, (k_now, k_orig) in enumerate(
            zip(k_pool, original_k, strict=True)
        ):
            assert torch.allclose(k_now[:chunk_size], k_orig), (
                f"K layer {layer_idx} not restored via connector.load_kv"
            )
        for layer_idx, (v_now, v_orig) in enumerate(
            zip(v_pool, original_v, strict=True)
        ):
            assert torch.allclose(v_now[:chunk_size], v_orig), (
                f"V layer {layer_idx} not restored via connector.load_kv"
            )
    finally:
        connector.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
