#!/bin/bash

# Runs the correctness tests using vLLM as a baseline for the KV Transfer of LMCache for a single model

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

bash env-setup.sh

# Activate the virtual environment for the remainder of this script so that
# any additional Python invocations (outside the nested scripts) run inside it.
# env-setup.sh creates or reuses `.venv`, but since it is executed in a
# subshell its environment changes do not propagate back here.
source .venv/bin/activate

bash e2e-one-model.sh meta-llama/Llama-3.1-8B
bash e2e-one-model.sh deepseek-ai/DeepSeek-V2-Lite

bash env-cleanup.sh