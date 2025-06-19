#!/bin/bash
set -e

# ASSUMPTION: CUDA is installed in /usr/local/cuda-{version}
CUDA_VERSION="12.1"
export CUDA_HOME="/usr/local/cuda-${CUDA_VERSION}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# If a cached virtual-environment already exists (restored by Buildkite cache),
# simply activate it and return early to make this script idempotent.
if [ -d ".venv" ]; then
  echo "[env-setup] Reusing previously cached Python virtual environment (.venv)"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exit 0
fi

uv venv --python 3.12
source .venv/bin/activate

# Extra dependencies needed for the MMLU scripts
uv pip install requests pandas numpy tqdm matplotlib fastapi
# Install lmcache from source
# the lmcache wheel also gives us access to:
# lmcache_server entrypoint
# lmcache_controller entrypoint
cd ../../
uv pip install -e . 

uv pip install vllm

# come back to the correctness directory
cd $SCRIPT_DIR
# Download the MMLU dataset
wget -q --show-progress https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar