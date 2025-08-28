# LMCache Prefetch All
This is an example to demonstrate how to preload all of the persisted KV Caches from disk / remote store to local CPU.

## Prerequisites
Your server should have at least 2 GPUs.  

This will use port 8000 and 8001 for 2 vllms and port 8002 and 8003 for the corresponding LMCache workers. The controller itself occupies port 9000 and 9001.

This example will use redis specifically. Start a redis server on `localhost:6379`

## Steps
1. Start two vllm engines at port 8000 and port 8001:

```bash
PYTHONHASHSEED=123 CUDA_VISIBLE_DEVICES=0 LMCACHE_CONFIG_FILE=instance1.yaml vllm serve Qwen/Qwen3-32B-AWQ --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

```bash
PYTHONHASHSEED=123 CUDA_VISIBLE_DEVICES=1 LMCACHE_CONFIG_FILE=instance2.yaml vllm serve Qwen/Qwen3-32B-AWQ --port 8001 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

2. Start the lmcache controller at port 9000 and the monitor at port 9001:

```bash
lmcache_controller --host localhost --port 9000 --monitor-port 9001
```

3. Send a long context to vllm engine 1, which will store inside of the redis backend:  
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "request 1: '"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```

4. Send a prefetch_all request for vllm engine 2:
```bash
curl -X POST http://localhost:9000/prefetch_all \
  -H "Content-Type: application/json" \
  -d '{
    "instance_id": "lmcache_instance_2"
  }'
```

You should be able to see a return message indicating the KV cache has been prefetched from 

```plaintext
{"event_id": "xxx", "num_keys": 54}
```

This means 54 keys were prefetched from the remote backend to your local cpu

5. Send the same long context to vllm engine 2, which will retrieve from local cpu instead of the shared remote backend. 
```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-AWQ",
    "prompt": "request 1: '"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```