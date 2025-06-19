#!/bin/bash
set -e

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# ASSUMPTION: env-setup.sh was run before this script

# Clean up the environment
# This is necessary because we are installing pre-release wheels of LMCache. Thus, if we do not clean up the environment, we will accumulate a lot of disk space. 
deactivate || true # || true in case we are not in the venv
rm -rf correctness_venv || true # || true in case the venv does not exist

# In case pip installed the packages in cache as well
pip cache purge
