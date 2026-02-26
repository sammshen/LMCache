# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING, Any, Tuple

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)

# Error message for accessing non-existent attributes in GPU KV Cache
_ATTRIBUTE_NOT_EXIST_ERROR = "trying to access an attribute of the GPU KV Cache "
"that does not exist for the format detected {format}. "
"A misalignment with the GPUKVFormat must be resolved"


def need_gpu_interm_buffer(lmcache_config: LMCacheEngineConfig):
    """
    Check if the GPU Connector needs to create an intermediate
    buffer on the GPU
    """
    if lmcache_config.enable_pd:
        return False
    else:
        return True


def assert_layerwise_gpu_connector(gpu_connector: "GPUConnectorInterface"):
    """
    Assert that a GPU Connector is a layerwise connector.
    """
    # Import at runtime to avoid circular dependency
    # First Party
    from lmcache.v1.gpu_connector.gpu_connectors import (
        SGLangLayerwiseGPUConnector,
        VLLMBufferLayerwiseGPUConnector,
        VLLMPagedMemLayerwiseGPUConnector,
    )

    assert isinstance(
        gpu_connector,
        (
            VLLMPagedMemLayerwiseGPUConnector,
            VLLMBufferLayerwiseGPUConnector,
            SGLangLayerwiseGPUConnector,
        ),
    )


def legible_print_gpu_kv_format(gpu_kv_format: "lmc_ops.GPUKVFormat"):
    """
    Print the GPU KV Format in a legible way
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        logger.info(
            "GPU KV Format: "
            "[num_blocks, num_layers, 2, block_size, num_heads, head_size]"
        )
        logger.info("Currently used by:\n  - vLLM CROSS_LAYER mode")

    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS:
        logger.info(
            "GPU KV Format: "
            "List[num_layers] of "
            "[2, num_blocks, block_size, num_heads, head_size]"
        )
        logger.info("Currently used by:\n  - vLLM non-MLA flash attention")

    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS:
        logger.info(
            "GPU KV Format: "
            "List[num_layers] of "
            "[num_blocks, 2, block_size, num_heads, head_size]"
        )
        logger.info("Currently used by:\n  - vLLM non-MLA flash infer")

    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        logger.info(
            "GPU KV Format: List[num_layers] of [num_blocks, block_size, head_size]"
        )
        logger.info("Currently used by:\n  - vLLM MLA")

    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        logger.info(
            "GPU KV Format: "
            "List[2] -> List[num_layers] of "
            "[page_buffer_size, num_heads, head_size]"
        )
        logger.info(
            "Currently used by:\n  - SGLang MHA (flash attention and flash infer)"
        )

    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        logger.info(
            "GPU KV Format: List[num_layers] of [page_buffer_size, 1, head_size]"
        )
        logger.info("Currently used by:\n  - SGLang MLA")

    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS:
        logger.info(
            "GPU KV Format: "
            "List[num_layers] of "
            "[2, num_blocks, block_size, num_heads, head_size] (permuted)"
        )
        logger.info(
            "Currently used by:\n  - vLLM non-MLA flash attention with permuted layout"
        )

    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS:
        logger.info(
            "GPU KV Format: "
            "List[num_layers] of "
            "[num_blocks, 2, block_size, num_heads, head_size] (permuted)"
        )
        logger.info(
            "Currently used by:\n  - vLLM non-MLA flash infer with permuted layout"
        )

    else:
        logger.warning(f"Unknown GPU KV Format: {gpu_kv_format}")


def _list_depth_tensor_dim(kv_caches: Any) -> Tuple[int, int]:
    """
    Get the number of external wrapping lists in the kv_caches.

    Assumption: kv_caches is of the form
    List[List[...List[torch.Tensor]]]
    """
    depth = 0
    while isinstance(kv_caches, list):
        depth += 1
        if not kv_caches:
            raise ValueError("encountered an empty list")
        kv_caches = kv_caches[0]
    if not isinstance(kv_caches, torch.Tensor):
        raise ValueError("encountered a non-tensor inside")
    return depth, kv_caches.ndim


def is_permuted(kv_cache: torch.Tensor) -> tuple[int, int] | None:
    """
    Detect if KV cache uses permuted memory layout.

    Args:
        kv_cache: A tensor with shape (2, num_blocks, block_size, num_heads, head_size)
                  or (num_blocks, 2, block_size, num_heads, head_size)

    Returns:
        None if contiguous (default layout)
        (dim1, dim2) tuple indicating which dimensions are permuted in memory
        For vLLM: returns (2, 3) meaning block_size and num_heads are swapped
    """
    if kv_cache.is_contiguous():
        return None

    # For shape: (2, num_blocks, block_size, num_heads, head_size)
    # Contiguous: stride[2] > stride[3] (block_size has larger stride than heads)
    # Permuted:   stride[3] > stride[2] (heads has larger stride than block_size)

    if len(kv_cache.shape) >= 5:
        if kv_cache.stride(2) > kv_cache.stride(3):
            return None
        else:
            # Dimensions 2 and 3 are permuted (block_size and num_heads)
            return (2, 3)

    # Fallback: assume contiguous
    return None


def ensure_contiguous_for_transfer(
    kv_caches: Any, permuted_dims: tuple[int, int] | None
) -> Tuple[Any, bool]:
    """
    Ensure KV cache tensors are contiguous for kernel transfer.

    Args:
        kv_caches: List of KV cache tensors per layer or single tensor
        permuted_dims: Tuple of permuted dimension indices, or None if contiguous

    Returns:
        Tuple of (contiguous_tensors, made_copy)
        - contiguous_tensors: List of contiguous tensors or single contiguous tensor
        - made_copy: True if copies were made, False if already contiguous
    """
    if permuted_dims is None:
        # Already contiguous or assumed contiguous
        return kv_caches, False

    # Permuted layout: must make contiguous
    logger.warning(
        "Permuted layout detected (dims %s). Making contiguous copy of KV caches "
        "for LMCache transfer. This will use additional GPU memory.",
        permuted_dims,
    )

    # Handle both list and single tensor cases
    if isinstance(kv_caches, list):
        contiguous_caches = [kv.contiguous() for kv in kv_caches]
    else:
        contiguous_caches = kv_caches.contiguous()

    return contiguous_caches, True


def discover_gpu_kv_format(
    kv_caches: Any, serving_engine: EngineType
) -> "lmc_ops.GPUKVFormat":
    """
    Discover the GPU KV Cache Format and memory layout from the kv_caches.

    The logic is that "external" layers are lists and there is one tensor internally.
    We "unwrap" layers until we find the tensor.

    Please see csrc/mem_kernels.cuh for the naming schema of the GPUKVFormat.

    Returns:
        The detected GPUKVFormat
    """
    # list_depth: number of external wrapping lists
    # tensor_dim: number of dimensions of the internal tensor
    list_depth, tensor_dim = _list_depth_tensor_dim(kv_caches)
    logger.info("list_depth: %d, tensor_dim: %d", list_depth, tensor_dim)
    list_dims = []
    ptr = kv_caches
    for _ in range(list_depth):
        list_dims.append(len(ptr))
        ptr = ptr[0]
    # ptr is now the tensor
    tensor_dims = list(ptr.shape)
    dims_str = (
        "".join(f"[{d}]" for d in list_dims) + f"[{', '.join(map(str, tensor_dims))}]"
    )
    logger.info("GPU KV Cache Dimensions: %s", dims_str)

    detected_format = None
    permuted_dims = None

    if serving_engine == EngineType.VLLM:
        if list_depth == 0:
            # vllm cross layer
            detected_format = lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS
        elif list_depth == 1:
            if tensor_dim == 5:
                permuted_dims = is_permuted(kv_caches[0])

                # Assert for vLLM that if permuted, it's dims (2, 3)
                if permuted_dims is not None:
                    assert permuted_dims == (2, 3), (
                        f"vLLM KV cache expected permutation of dims (2, 3) "
                        f"but got {permuted_dims}"
                    )

                if kv_caches[0].shape[0] == 2:
                    # vllm non-MLA flash attention
                    if permuted_dims is not None:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS
                    else:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
                elif kv_caches[0].shape[1] == 2:
                    # vllm non-MLA flash infer
                    if permuted_dims is not None:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS
                    else:
                        detected_format = lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
            elif tensor_dim == 3:
                # vllm MLA
                detected_format = lmc_ops.GPUKVFormat.NL_X_NB_BS_HS
    elif serving_engine == EngineType.SGLANG:
        if list_depth == 1:
            if kv_caches[0].shape[1] == 1:
                # sglang MLA
                detected_format = lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS
        elif list_depth == 2:
            # sglang MHA (flash attention and flash infer)
            detected_format = lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS

    if detected_format is not None:
        legible_print_gpu_kv_format(detected_format)
        if permuted_dims is not None:
            logger.info("Detected KV cache permuted dims: %s", permuted_dims)
        return detected_format
    else:
        raise ValueError(
            "currently unsupported kv_caches format "
            f"with list depth {list_depth} and tensor dimension {tensor_dim}"
        )


def get_num_layers(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of layers from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[1]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        return len(kv_caches)
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return len(kv_caches)
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return len(kv_caches)
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return len(kv_caches[0])
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return len(kv_caches)
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_num_blocks(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of blocks from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[0]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        return kv_caches[0].shape[1]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return kv_caches[0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_block_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the block size from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        return kv_caches[0].shape[2]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_page_buffer_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get page buffer size (num_blocks * block_size) from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        return kv_caches.shape[0] * kv_caches.shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        # List[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        return kv_caches[0].shape[1] * kv_caches[0].shape[2]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # List[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # List[num_layers] of [num_blocks, block_size, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]
        return kv_caches[0][0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # List[num_layers] of [page_buffer_size, 1, head_size]
        return kv_caches[0].shape[0]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_num_heads(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of heads from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[4]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        return kv_caches[0].shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return kv_caches[0].shape[3]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=gpu_kv_format))
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].shape[1]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_hidden_dim_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the hidden dimension from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[4] * kv_caches.shape[5]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return kv_caches[0].shape[3] * kv_caches[0].shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].shape[1] * kv_caches[0][0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].shape[2]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_head_size(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the head size from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.shape[5]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        return kv_caches[0].shape[4]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return kv_caches[0].shape[4]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].shape[2]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].shape[2]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_tokens_per_layer(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of tokens per layer from the kv_caches
    (num_blocks * block_size or page_buffer_size)
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        return kv_caches.shape[0] * kv_caches.shape[3]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        # List[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape[0] * k_cache_shape[1]
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # List[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape[0] * k_cache_shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # List[num_layers] of [num_blocks, block_size, head_size]
        return kv_caches[0].shape[0] * kv_caches[0].shape[1]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # List[2] -> List[num_layers] of [page_buffer_size, num_heads, head_size]
        return kv_caches[0][0].shape[0]
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # List[num_layers] of [page_buffer_size, 1, head_size]
        return kv_caches[0].shape[0]
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def get_elements_per_layer(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> int:
    """
    Get the number of elements per layer from the kv_caches
    (including both K and V for non-MLA)
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        # [num_blocks, num_layers, 2, block_size, num_heads, head_size]
        # For one layer: [num_blocks, 2, block_size, num_heads, head_size]
        num_blocks = kv_caches.shape[0]
        block_size = kv_caches.shape[3]
        num_heads = kv_caches.shape[4]
        head_size = kv_caches.shape[5]
        return num_blocks * 2 * block_size * num_heads * head_size
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        # List[num_layers] of [2, num_blocks, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][0].shape
        return k_cache_shape.numel() * 2
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        # List[num_layers] of [num_blocks, 2, block_size, num_heads, head_size]
        k_cache_shape = kv_caches[0][:, 0].shape
        return k_cache_shape.numel() * 2
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        # List[num_layers] of [num_blocks, block_size, head_size] (MLA)
        return kv_caches[0].numel()
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        # List[2] -> List[num_layers] of
        # [page_buffer_size, num_heads, head_size] (separate K and V)
        return kv_caches[0][0].numel() * 2
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        # List[num_layers] of [page_buffer_size, 1, head_size] (MLA)
        return kv_caches[0].numel()
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")


def assert_is_vllm_flash_attn_or_flash_infer(gpu_kv_format: "lmc_ops.GPUKVFormat"):
    """
    Ensure that we have a GPU KV Cache Format
    that is either vLLM's flash attention or flash infer.
    """
    assert (
        gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS
        or gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS
        or gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS
        or gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS
    )


def is_mla(gpu_kv_format: "lmc_ops.GPUKVFormat") -> bool:
    """
    Check if the GPU KV Format is MLA
    """
    return (
        gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS  # vllm MLA
        or gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS  # sglang MLA
    )


def get_dtype(kv_caches: Any, gpu_kv_format: "lmc_ops.GPUKVFormat") -> torch.dtype:
    """
    Get the dtype from the kv_caches
    """
    if gpu_kv_format == lmc_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS:
        return kv_caches.dtype
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_NH_BS_HS,
    ):
        return kv_caches[0].dtype
    elif gpu_kv_format in (
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_BS_NH_HS,
        lmc_ops.GPUKVFormat.NL_X_NB_TWO_NH_BS_HS,
    ):
        return kv_caches[0].dtype
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NB_BS_HS:
        return kv_caches[0].dtype
    elif gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
        return kv_caches[0][0].dtype
    elif gpu_kv_format == lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS:
        return kv_caches[0].dtype
    else:
        raise ValueError(f"Unknown GPU KV Format: {gpu_kv_format}")
