# SGLang & LMCache Integration

Two modes are supported: the in-process engine (LMCache runs inside
the SGLang scheduler process) and the multi-process engine (LMCache
runs as a standalone ZMQ server, with SGLang as a client).

## Install
This project depends on a pending pull request in the SGLang repository. Until PR is merged, please use the code from that specific branch instead of the SGLang main branch.
```bash
git clone https://github.com/Oasis-Git/sglang/tree/lmcache
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

## In-process mode

```bash
export LMCACHE_CONFIG_FILE=lmcache_config.yaml
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --port 30000 --tp 2 --page-size 32 --enable-lmcache
```

LMCache runs alongside the SGLang scheduler. Cache state lives and dies
with the SGLang process.

## Multi-process mode

Two terminals — one for the LMCache MP server, one for SGLang.

**Terminal 1**: start the standalone LMCache server.

```bash
./run_mp_server.sh 5555
```

This launches `python -m lmcache.v1.multiprocess.server` with sane
defaults (chunk size 256, 8 GB L1, LRU). Edit `run_mp_server.sh` to
change them.

**Terminal 2**: start SGLang. Point it at the running server via
`LMCACHE_SERVER_URL`, and use `lmcache_mp_config.yaml` so the
`LMCRadixCache` instantiates the MP connector instead of the in-process
one. (SGLang's `--enable-lmcache` flag picks up the connector class
based on the LMCache config — set the connector class env var the
same way you would for vLLM-MP, e.g. via SGLang's CLI.)

```bash
export LMCACHE_CONFIG_FILE=lmcache_mp_config.yaml
export LMCACHE_SERVER_URL=tcp://localhost:5555
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --port 30000 --tp 2 --page-size 32 --enable-lmcache
```

You should see `LMCache SGLang MP connector: connected to
tcp://localhost:5555 ...` in the SGLang logs and `Stored N tokens` /
`Retrieved N out of N required tokens` events in the server log
(`/tmp/lmcache_mp_server_5555.log`).

If you hope to run the benchmark, please refer to https://github.com/sgl-project/sglang/tree/main/benchmark/hicache.

## Verifying store / retrieve without a real model

The repo's E2E tests do exactly this — they spin up the LMCache MP
server in a subprocess, register an SGLang-shaped KV pool, and verify
`STORE` then `LOOKUP` then `RETRIEVE` round-trips bytes correctly.
Run them with:

```bash
pytest -v tests/v1/test_sglang_mp_e2e.py tests/v1/test_sglang_inprocess_e2e.py
```

These tests don't need SGLang installed (only `torch` + LMCache + a
GPU) and serve as a fast confirmation that the integration plumbs end
to end before pulling in the full SGLang stack.

