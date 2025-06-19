#!/bin/bash

# ASSUMPTION: vllm wheel is available in the environment (please run env-setup.sh before)

# Overview:
# This script is used to deploy a single vLLM serving engine on port 8000

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

# Deploy the vllm serving engine (without LMCache)

free_port 8000
nohup vllm serve "$MODEL_URL" \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    > vllm.log 2>&1 &

# Wait for the server to be ready
total_time_elapsed=0
until curl --fail -s http://localhost:8000/v1/models | grep -q "$MODEL_URL"; do
  echo "Waiting for model $MODEL_URL to be loaded..."
  sleep 10
  echo "--------------------------------"
  echo "Most recent serving engine logs:"
  echo "--------------------------------"
  tail -n 10 vllm.log
  echo "--------------------------------"
  total_time_elapsed=$((total_time_elapsed + 10))
done