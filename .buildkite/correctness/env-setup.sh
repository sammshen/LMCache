#!/bin/bash
set -e

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# NOTE: please run env-cleanup.sh after this script

# Step 1: Install system dependencies for building Python
sudo apt update
sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl git \
  libncursesw5-dev xz-utils tk-dev libxml2-dev \
  libxmlsec1-dev libffi-dev liblzma-dev ca-certificates

# Step 2: Install pyenv if not already installed
if [ ! -d "$HOME/.pyenv" ]; then
  git clone https://github.com/pyenv/pyenv.git "$HOME/.pyenv"
fi

# Step 3: Add pyenv to the current shell environment
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Step 4: Install desired Python version if not already installed
PYTHON_VERSION=3.10.14
if ! pyenv versions --bare | grep -qx "$PYTHON_VERSION"; then
  pyenv install "$PYTHON_VERSION"
fi

# Step 5: Create virtual environment using installed Python version
VENV_DIR="correctness_venv"
pyenv shell "$PYTHON_VERSION"
"$PYENV_ROOT/versions/$PYTHON_VERSION/bin/python" -m venv "$VENV_DIR"

# Step 6: Activate the virtual environment and confirm
source "$VENV_DIR/bin/activate"
python --version

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

# Extra dependencies needed for the MMLU scripts
pip install requests pandas numpy tqdm matplotlib fastapi

pip install vllm
# Install lmcache from source
# the lmcache wheel also gives us access to:
# lmcache_server entrypoint
# lmcache_controller entrypoint
cd ../../
pip install -e . 

# come back to the correctness directory
cd $SCRIPT_DIR
# Download the MMLU dataset
wget -q --show-progress https://people.eecs.berkeley.edu/~hendrycks/data.tar
tar xf data.tar