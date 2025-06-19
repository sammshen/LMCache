#!/bin/bash

# ASSUMPTION: vllm and lmcache wheels are available in the environment (please run env-setup.sh before)

# Overview:
# This script is used to deploy 2 vLLM + LMCache serving engines on port 8000 and 8001
# They will have a peer to peer connection through an LMCache Server deployed on 8500
# The purpose is to send requests to the first serving engine to store KV Caches and then send requests to the second serving engine to retrieve KV Caches
# This way the responses returned by the second serving engine can be used to test the correctness of LMCache KV Transfer

# Arguments:
MODEL_URL=$1

# Utility:
free_port() {
    if [ -z "$1" ]; then
        echo "Usage: free_port <port>"
        return 1
    fi

    local port=$1
    local pid

    pid=$(lsof -t -i :"$port")

    if [ -z "$pid" ]; then
        echo "No process is using port $port"
    else
        echo "Killing process(es) using port $port: $pid"
        sudo kill -9 $pid
    fi
}

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

source correctness_venv/bin/activate

# Deploy the lmcache server

free_port 65432
nohup lmcache_server localhost 65432 &

# Place the model in the HF cache

python hf-cache-model.py --model-url "$MODEL_URL"

MODEL_URL="hf-cache" # hf-cache-model.py will place the model in the HF cache

# Deploy the first vLLM + LMCache serving engine on port 8000

# KV producer
free_port 8000
nohup env \
    CUDA_VISIBLE_DEVICES=0 \
    LMCACHE_REMOTE_URL="lm://localhost:65432" \
    LMCACHE_REMOTE_SERDE="naive" \
    vllm serve "$MODEL_URL" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer"}' \
    > lmcache-1.log 2>&1 &

# Deploy the second vLLM + LMCache serving engine on port 8001

# KV consumer
free_port 8001
nohup env \
    CUDA_VISIBLE_DEVICES=1 \
    LMCACHE_REMOTE_URL="lm://localhost:65432" \
    LMCACHE_REMOTE_SERDE="naive" \
    vllm serve "$MODEL_URL" \
    --port 8001 \
    --trust-remote-code \
    --max-model-len 8192 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}' \
    > lmcache-2.log 2>&1 &

# Wait for both serving engines to be ready
total_time_elapsed=0
until curl --fail -s http://localhost:8000/v1/models | grep -q "$MODEL_URL" && curl --fail -s http://localhost:8001/v1/models | grep -q "$MODEL_URL"; do
  echo "Waiting for model $MODEL_URL to be loaded..."
  sleep 10
  echo "--------------------------------"
  echo "Most recent serving engine 1 (port 8000) logs:"
  echo "--------------------------------"
  tail -n 10 lmcache-1.log
  echo "--------------------------------"
  echo "Most recent serving engine 2 (port 8001) logs:"
  echo "--------------------------------"
  tail -n 10 lmcache-2.log
  echo "--------------------------------"
  total_time_elapsed=$((total_time_elapsed + 10))
done




