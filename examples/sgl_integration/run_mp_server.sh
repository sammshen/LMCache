#!/usr/bin/env bash
# Launch the standalone LMCache MP server SGLang will connect to.
#
# Usage: ./run_mp_server.sh [PORT]
#
# Defaults to 5555. Tail the log to see store/retrieve events.

set -euo pipefail

PORT="${1:-5555}"
LOG_DIR="${LOG_DIR:-/tmp}"
LOG_FILE="${LOG_DIR}/lmcache_mp_server_${PORT}.log"

# Make libc10_cuda.so / libtorch_cuda.so reachable to the loader so
# ``lmcache.c_ops`` (built against torch) imports cleanly in the
# subprocess. ``import torch`` lazy-loads them; lmcache wants them on
# import.
TORCH_LIB="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Starting LMCache MP server on port ${PORT}, logging to ${LOG_FILE}"
exec python -m lmcache.v1.multiprocess.server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --chunk-size 256 \
    --l1-size-gb 8 \
    --eviction-policy LRU \
    --max-gpu-workers 4 \
    --max-cpu-workers 4 \
    2>&1 | tee "${LOG_FILE}"
