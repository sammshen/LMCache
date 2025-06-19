#!/bin/bash

# Runs the correctness tests using vLLM as a baseline for the KV Transfer of LMCache for a single model

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# ASSUMPTIONS: env-setup.sh has been run
source correctness_venv/bin/activate

bash env-setup.sh

bash e2e-one-model.sh meta-llama/Llama-3.1-8B
bash e2e-one-model.sh deepseek-ai/DeepSeek-V2-Lite


bash env-cleanup.sh