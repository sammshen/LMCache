#!/bin/bash
set -e

# Make sure all the scripts run and cooperate with each other in the .buildkite/correctness directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $SCRIPT_DIR

# ASSUMPTION: env-setup.sh was run before this script

# Clean up the environment
# This is necessary because we are installing pre-release wheels of LMCache. Thus, if we do not clean up the environment, we will accumulate a lot of disk space. 
# Deactivate the virtual environment if it is active. Do **not** delete the
# `.venv` directory so that the Buildkite cache plugin can persist it across
# pipeline steps.

# Deactivate the venv (safe-to-call even if not active)
deactivate || true

# Remove the virtual environment directory to free disk space. This runs at
# the very end of the CI step, *after* any caching or artifact upload, so it
# does not interfere with Buildkite's cache plugin.
echo "[env-cleanup] Removing virtual environment (.venv)"
rm -rf .venv || true

# In case pip installed the packages in cache as well
pip cache purge
