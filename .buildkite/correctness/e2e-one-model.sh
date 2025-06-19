#!/bin/bash

# Runs the correctness tests using vLLM as a baseline for the KV Transfer of LMCache for a single model

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# ASSUMPTIONS: env-setup.sh has been run
source .venv/bin/activate

# Arguments:
MODEL_URL=$1

# Deploy 1x vLLM on the model
bash deploy-1-vllm.sh "$MODEL_URL"

# Run MMLU on 1x vLLM with the model with 15 subjects
python 1-mmlu.py --model "$MODEL_URL" --number-of-subjects 15

# Deploy a 2x LMCache setup (one KV producer, one KV consumer via LMCache server) on the model
bash deploy-2-lmcache.sh "$MODEL_URL"

# Run MMLU on 2x LMCache setup with the model with 15 subjects
python 2-mmlu.py --model "$MODEL_URL" --number-of-subjects 15

# Summarize the results with a picture
python summarize-results.py --model "$MODEL_URL"