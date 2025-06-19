#!/bin/bash
set -e

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# NOTE: please run env-cleanup.sh after this script

python -m venv correctness_venv
source correctness_venv/bin/activate

# Get the latest pre-release version of lmcache from TestPyPI
latest_version=$(curl -s https://test.pypi.org/simple/lmcache/ \
  | grep -oP 'lmcache-\K[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+' \
  | sort -V | tail -n 1)

if [ -z "$latest_version" ]; then
  echo "Failed to fetch latest version."
  exit 1
fi

echo "Latest version of pre-release lmcache found: $latest_version"

pip install vllm
# Install latest pre-release wheel of lmcache
# the lmcache wheel also gives us access to:
# lmcache_server entrypoint
# lmcache_controller entrypoint
pip install --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple lmcache==$latest_version

# Extra dependencies needed for the MMLU scripts
pip install requests pandas numpy tqdm json matplotlib

# Download the MMLU dataset
wget -q --show-progress https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar