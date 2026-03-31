"""
Minimal GPU connector for Ollama integration.

Unlike vLLM/TRT-LLM connectors that copy between GPU paged buffers and
LMCache memory objects, this connector works with pre-serialized byte
buffers. Ollama serializes its GGML KV tensors to bytes on its side
(via ml.Tensor.Bytes()), sends them over the socket, and this connector
wraps/unwraps them for the storage manager.
"""

import torch
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryObj


class BytesGPUConnector(GPUConnectorInterface):
    """A 'connector' that wraps raw bytes into MemoryObj.

    No GPU involved -- named GPUConnector only to satisfy the interface.
    """

    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int,
                 chunk_size: int, dtype: torch.dtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.hidden_dim = num_kv_heads * head_dim
        self.dtype = dtype
        # Shape for a single chunk in 2LTD format
        self._chunk_shape = torch.Size([2, num_layers, chunk_size, self.hidden_dim])

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Retrieve: extract bytes from memory_obj for sending back to Ollama."""
        size = memory_obj.meta.get_size()
        raw = memory_obj.raw_data[:size].clone().cpu()
        kwargs["result_container"].append(raw.numpy().tobytes())

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Store: slice the chunk [start:end] from the full KV bytes into memory_obj.

        kwargs["kv_bytes"]: full KV bytes in 2LTD layout [2, num_layers, total_tokens, hidden_dim]
        kwargs["total_tokens"]: total number of tokens in kv_bytes
        """
        kv_bytes = kwargs["kv_bytes"]
        total_tokens = kwargs["total_tokens"]
        full_tensor = torch.frombuffer(
            bytearray(kv_bytes), dtype=self.dtype
        ).reshape(2, self.num_layers, total_tokens, self.hidden_dim)
        chunk = full_tensor[:, :, start:end, :].contiguous()
        # Write into raw_data as flat bytes, matching the memory layout
        chunk_bytes = chunk.view(torch.uint8)
        memory_obj.raw_data[:chunk_bytes.numel()].copy_(chunk_bytes.flatten())

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for mo_list, s, e in zip(memory_objs, starts, ends):
            for mo in (mo_list if isinstance(mo_list, list) else [mo_list]):
                self.from_gpu(mo, s, e, **kwargs)

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for mo_list, s, e in zip(memory_objs, starts, ends):
            for mo in (mo_list if isinstance(mo_list, list) else [mo_list]):
                self.to_gpu(mo, s, e, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        return self._chunk_shape

    def get_dtype(self) -> torch.dtype:
        return self.dtype
