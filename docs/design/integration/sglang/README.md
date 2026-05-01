# SGLang integration

## Adapter shape

```
lmcache/integration/sglang/
├── __init__.py             # Optional-import surface
├── utils.py                # ENGINE_NAME, lmcache_get_config
├── sglang_adapter.py       # In-process — engine in SGLang process
└── sglang_mp_adapter.py    # Multi-process — engine in standalone server
```

Both adapters expose the same SGLang-facing class surface
(`LMCacheConnector` / `LMCacheLayerwiseConnector`) so the call sites in
SGLang's `LMCRadixCache` can swap modes without code changes — only the
connector class is different.

| Mode | Class | Backend |
|---|---|---|
| In-process | `LMCacheConnector` / `LMCacheLayerwiseConnector` | Singleton `LMCacheEngine` inside the SGLang scheduler process |
| Multi-process | `LMCacheMPConnector` / `LMCacheMPLayerwiseConnector` | Standalone LMCache ZMQ server (`python -m lmcache.v1.multiprocess.server`) |

## SGLang KV cache layout

SGLang allocates per-layer paged buffers, not a single cross-layer
pool. The shape varies by attention variant:

| Variant | Per-layer K shape | Per-layer V shape | Format |
|---|---|---|---|
| MHA (flash attention / flash infer) | `[PBS, NH, HS]` | `[PBS, NH, HS]` | `TWO_X_NL_X_NBBS_NH_HS` |
| MLA | `[PBS, 1, HS]` | (shares K storage) | `NL_X_NBBS_ONE_HS` |

`PBS` (page buffer size) = `num_blocks * block_size`. SGLang's paged
allocator manages slots at token granularity, so the natural mapping
to LMCache's block-level transfer kernel is **block_size = 1, num_blocks
= PBS** — each "block" is one token slot, and the `block_ids` passed
to `STORE` / `RETRIEVE` are token-level slot indices. This is what
[`get_block_size`](../../../../lmcache/v1/gpu_connector/utils.py)
returns for the SGLang formats.

## In-process flow

```
SGLang LMCRadixCache.match_prefix
  ↓
LMCacheLayerwiseConnector.start_load_kv(load_md)
  ↓ engine.lookup(tokens) → matched count
  ↓ engine.retrieve_layer(tokens, kvcaches=[k_pool, v_pool], slot_mapping)
  ↓
LMCacheLayerwiseConnector.load_kv_layerwise(layer_id)
  ↓ steps the per-layer generator the engine returned
```

`LMCacheConnector` (non-layerwise) flattens `k_pool + v_pool` into a
flat list — that's recognized by the format detector for MLA but not
for MHA. SGLang's actual call site uses `LMCacheLayerwiseConnector`,
which keeps the nested `[k_pool, v_pool]` form.

## Multi-process flow

```
LMCacheMPConnector.__init__
  ↓ MessageQueueClient over ZMQ
  ↓ REGISTER_KV_CACHE(instance_id, [CudaIPCWrapper(t) for t in k+v],
  │                   model_name, world_size,
  │                   EngineType.SGLANG,
  │                   {"sglang_attention": "MHA"|"MLA",
  │                    "sglang_num_layers": NL})
LMCacheMPConnector.load_kv(load_md)
  ↓ LOOKUP(key, world_size)
  ↓ QUERY_PREFETCH_STATUS(request_id) → cached chunks
  ↓ RETRIEVE(key, instance_id, block_ids, ipc_event, skip_first_n_tokens)
  ↓ wait on returned IPC event
LMCacheMPConnector.store_kv(store_md)
  ↓ STORE(key, instance_id, block_ids, ipc_event)
```

The pattern mirrors the TRT-LLM MP adapter (`tensorrt_mp_adapter.py`):

- `CudaIPCWrapper` for the K/V tensors (SGLang allocates through
  PyTorch's caching allocator, so the standard wrapper applies — no
  `RawCudaIPCWrapper` like TRT-LLM needs).
- `LOOKUP` + `QUERY_PREFETCH_STATUS` (two-phase) instead of in-process
  `engine.lookup`.
- `RETRIEVE` / `STORE` carry an interprocess CUDA event so the server
  can synchronize against the SGLang stream.
- `LOOKUP` between `STORE` and a subsequent `RETRIEVE` doubles as a
  barrier — it waits until the L1 host-side commit launched at the tail
  of `STORE` has run, so the chunks are readable.
- `END_SESSION` clears per-request server-side state after each
  load/store call.

## Layout reshape on the server

`REGISTER_KV_CACHE` carries a single flat `KVCache = list[CudaIPCWrapper]`
on the wire, but SGLang's MHA format detector wants depth-2 (`[K_layers,
V_layers]`). The MP adapter sends `[k_pool[0], …, k_pool[NL-1],
v_pool[0], …, v_pool[NL-1]]`; on the server side
[`normalize_kv_and_discover_format`](../../../../lmcache/v1/gpu_connector/utils.py)
splits this back at its midpoint when `layout_hints["sglang_attention"]
== "MHA"`, using `sglang_num_layers` for validation.

This reshape-via-hints pattern is the same shape as TRT-LLM's 4-D pool
reshape: the wire payload stays a flat `list[CudaIPCWrapper]`, and
engine-specific layout reconstruction happens centrally in
`normalize_kv_and_discover_format` rather than in per-engine adapter
code.

## Layerwise connector under MP

SGLang's `LMCRadixCache` calls `start_load_kv` then `load_kv_layerwise`
for each layer. The MP server transfers all layers in one
`multi_layer_block_kv_transfer` kernel invocation, so the MP layerwise
connector does the full retrieve synchronously in `start_load_kv` and
makes `load_kv_layerwise(layer_id)` a no-op. This loses per-layer
overlap with compute but keeps the wire protocol unmodified.

## Configuration

The adapter resolves the server endpoint in this order:

1. `server_url=...` argument to the connector constructor.
2. `LMCACHE_SERVER_URL` environment variable.
3. `ipc:///tmp/lmcache.sock` (default).

Other LMCache configuration (chunk size, eviction policy, L1 size, L2
adapter) is owned by the standalone server process — start it with the
same `--chunk-size` / `--l1-size-gb` / `--eviction-policy` flags as for
vLLM or TRT-LLM.

## Constraints

- The adapter currently issues store/retrieve **synchronously** —
  `load_kv` blocks until the server's stream has finished. Async
  pipelining (`get_finished` style) is intentionally not modeled because
  SGLang's `LMCRadixCache` already serializes against `store_stream` /
  `load_stream`.
- The MP connector returns no `kv_events` — `get_kv_events` is a
  no-op. The standalone server's observability subscribers are the
  source of truth for KV cache events in MP mode.

## Tests

| Test | What it covers |
|---|---|
| `tests/v1/test_sglang_mp_e2e.py::test_sglang_mp_store_then_retrieve_mha` | Real LMCache server subprocess + SGLang MHA pool + protocol-level STORE/LOOKUP/RETRIEVE roundtrip with byte-perfect verification |
| `tests/v1/test_sglang_mp_e2e.py::test_sglang_mp_store_then_retrieve_mla` | Same as above but SGLang MLA shape |
| `tests/v1/test_sglang_mp_e2e.py::test_sglang_mp_connector_class_e2e` | Drives the full `LMCacheMPConnector` API (`store_kv` then `load_kv`) against a live server |
| `tests/v1/test_sglang_inprocess_e2e.py::test_sglang_inprocess_store_then_retrieve_mla` | Baseline: in-process `LMCacheConnector` round-trip with MLA pool |
