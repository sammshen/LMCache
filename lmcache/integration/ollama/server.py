"""
LMCache sidecar server for Ollama.

Usage:
    PYTHONHASHSEED=0 python -m lmcache.integration.ollama.server \
        --model llama3.1:8b \
        --num-layers 32 \
        --num-kv-heads 8 \
        --head-dim 128 \
        --chunk-size 256 \
        --cache-dir ~/.ollama/kvcache \
        --max-disk-gb 20 \
        --port 11435
"""

import argparse
import base64

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.integration.ollama.bytes_connector import BytesGPUConnector

app = FastAPI()
engine: LMCacheEngine = None


class LookupRequest(BaseModel):
    tokens: list[int]

class LookupResponse(BaseModel):
    num_cached_tokens: int

class StoreRequest(BaseModel):
    tokens: list[int]
    kv_data: str  # base64-encoded bytes, 2LTD layout

class StoreResponse(BaseModel):
    stored: bool

class RetrieveRequest(BaseModel):
    tokens: list[int]
    num_tokens: int  # how many prefix tokens to retrieve

class RetrieveResponse(BaseModel):
    kv_data: str  # base64-encoded bytes
    num_tokens: int


@app.post("/lookup", response_model=LookupResponse)
def lookup(req: LookupRequest):
    """Check how many prefix tokens are cached."""
    n = engine.lookup(tokens=req.tokens)
    return LookupResponse(num_cached_tokens=n)


@app.post("/store", response_model=StoreResponse)
def store(req: StoreRequest):
    """Store KV cache data for a token sequence."""
    kv_bytes = base64.b64decode(req.kv_data)
    engine.store(tokens=req.tokens, kv_bytes=kv_bytes, total_tokens=len(req.tokens))
    return StoreResponse(stored=True)


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """Retrieve cached KV data for a token prefix."""
    result_container = []
    ret_mask = engine.retrieve(
        tokens=req.tokens[:req.num_tokens],
        result_container=result_container,
    )

    if result_container:
        num_tokens = int(ret_mask.sum().item()) if ret_mask is not None else 0
        # Each chunk is [2, L, chunk_size, H] as flat bytes.
        # Reassemble into contiguous [2, L, total_tokens, H].
        chunk_shape = engine.metadata.get_shapes()[0]  # [2, L, chunk_size, H]
        chunks = []
        for blob in result_container:
            t = torch.frombuffer(bytearray(blob), dtype=engine.metadata.kv_dtype)
            chunks.append(t.reshape(chunk_shape))
        combined = torch.cat(chunks, dim=2).contiguous()
        kv_data = base64.b64encode(combined.numpy().tobytes()).decode()
    else:
        kv_data = ""
        num_tokens = 0

    return RetrieveResponse(kv_data=kv_data, num_tokens=num_tokens)


def _noop_broadcast(tensor_or_obj, src):
    """No-op broadcast for single-process Ollama deployment."""
    return tensor_or_obj


def create_engine(args) -> LMCacheEngine:
    """Initialize LMCache engine with disk-only storage."""
    hidden_dim = args.num_kv_heads * args.head_dim
    kv_shape = (args.num_layers, 2, args.chunk_size, args.num_kv_heads, args.head_dim)

    metadata = LMCacheMetadata(
        model_name=args.model,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=kv_shape,
        chunk_size=args.chunk_size,
    )

    config = LMCacheEngineConfig(
        chunk_size=args.chunk_size,
        local_cpu=True,
        max_local_cpu_size=1.0,
        local_disk=args.cache_dir,
        max_local_disk_size=args.max_disk_gb,
    )

    connector = BytesGPUConnector(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        chunk_size=args.chunk_size,
        dtype=torch.float16,
    )

    eng = LMCacheEngineBuilder.get_or_create(
        "ollama-instance", config, metadata, connector,
        broadcast_fn=_noop_broadcast,
        broadcast_object_fn=_noop_broadcast,
    )
    eng.post_init()
    return eng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--cache-dir", default="~/.ollama/kvcache")
    parser.add_argument("--max-disk-gb", type=float, default=20.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    global engine
    engine = create_engine(args)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
