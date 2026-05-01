.. _quickstart:

Quickstart
==========

This guide helps you get LMCache running end-to-end in a couple of minutes. Use the tabs below to switch the engine. Steps are the same; only the libraries and launch commands change.

.. tab-set::
   :sync-group: engine

   .. tab-item:: vLLM

      **Install LMCache**

      .. code-block:: bash

         uv venv --python 3.12
         source .venv/bin/activate
         uv pip install lmcache vllm

      LMCache supports two deployment modes with vLLM:

      - **Multiprocess (MP) mode** -- **recommended.** LMCache runs as a
        standalone service and vLLM attaches via ``LMCacheMPConnector``.
        Scales better, exposes management/observability endpoints, and
        supports sharing one cache across multiple engine instances.
      - **In-process mode** -- LMCache runs inside the vLLM process via
        ``LMCacheConnectorV1``. Single command, convenient for quick
        single-node experiments.

      .. tab-set::
         :sync-group: vllm-mode

         .. tab-item:: MP mode (recommended)
            :sync: mp

            Start the LMCache server:

            .. code-block:: bash

               # chunk-size 16 is an illustrative demo value so a short
               # prompt produces visible cache traffic; use the default
               # (256) in production.
               lmcache server \
                   --l1-size-gb 20 --eviction-policy LRU --chunk-size 16

            The ZMQ port (default **5555**) accepts connections from vLLM;
            the HTTP frontend (default **8080**) serves the management and
            metrics endpoints. See :doc:`../mp/quickstart` and
            :doc:`../mp/configuration` for the full list of
            ``lmcache server`` and connector options.

            Start vLLM with the MP connector in a separate terminal:

            .. code-block:: bash

               vllm serve Qwen/Qwen3-8B \
                   --port 8000 --kv-transfer-config \
                   '{"kv_connector":"LMCacheMPConnector", "kv_role":"kv_both"}'

            **Test** -- open a new terminal and send two requests whose
            prompts share a prefix:

            **First request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **Second request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **You should see LMCache logs like this** -- in MP mode the
            store/retrieve logs come from the standalone ``lmcache server``
            process, one entry per chunk.

            **First request** -- cache is empty, so every aligned chunk is
            offloaded:

            .. code-block:: text

               [2026-04-22 19:49:56,316] LMCache INFO: Stored 16 tokens in 0.023 seconds (server.py:390:lmcache.v1.multiprocess.server)
               [2026-04-22 19:49:56,555] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               [2026-04-22 19:49:56,691] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               ...

            **Second request** -- the shared prefix is retrieved from CPU
            RAM; only the new tail is stored:

            .. code-block:: text

               [2026-04-22 19:50:04,686] LMCache INFO: Retrieved 16 tokens in 0.003 seconds (server.py:573:lmcache.v1.multiprocess.server)
               [2026-04-22 19:50:04,832] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               [2026-04-22 19:50:04,968] LMCache INFO: Stored 16 tokens in 0.005 seconds (server.py:390:lmcache.v1.multiprocess.server)
               ...

            For request-level statistics (hit ratio, bytes transferred) see
            :doc:`../mp/observability`.

         .. tab-item:: In-process mode
            :sync: inproc

            Start vLLM with LMCache embedded in the engine process:

            .. code-block:: bash

               # The chunk size here is only for illustration purpose, use default one (256) later
               LMCACHE_CHUNK_SIZE=8 \
               vllm serve Qwen/Qwen3-8B \
                   --port 8000 --kv-transfer-config \
                   '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

            .. note::
               To customize further, create a config file. See
               :doc:`../api_reference/configurations` for all options.

            **Alternative simpler command:**

            .. code-block:: bash

               vllm serve <MODEL NAME> \
                   --kv-offloading-backend lmcache \
                   --kv-offloading-size <SIZE IN GB> \
                   --disable-hybrid-kv-cache-manager

            The ``--disable-hybrid-kv-cache-manager`` flag is mandatory.
            All configuration options from the
            :doc:`../api_reference/configurations` page still apply.

            **Test** -- open a new terminal and send two requests whose
            prompts share a prefix:

            **First request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **Second request**

            .. code-block:: bash

               curl http://localhost:8000/v1/completions \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "Qwen/Qwen3-8B",
                   "prompt": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models",
                   "max_tokens": 100,
                   "temperature": 0.7
                 }'

            **You should see LMCache logs like this** -- in-process mode
            emits the logs inline with the vLLM engine core.

            **First request** -- prompt is offloaded to LMCache:

            .. code-block:: text

               (EngineCore_DP0 pid=458469) [2025-09-30 00:08:43,982] LMCache INFO: Stored 31 out of total 31 tokens. size: 0.0040 gb, cost 1.95 ms, throughput: 1.98 GB/s; offload_time: 1.88 ms, put_time: 0.07 ms

            **Second request** -- hits the cache and stores the new tail:

            .. code-block:: text

               Reqid: cmpl-6709d8795d3c4464b01999c9f3fffede-0, Total tokens 32, LMCache hit tokens: 24, need to load: 8
               (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,502] LMCache INFO: Retrieved 8 out of 24 required tokens (from 32 total tokens). size: 0.0011 gb, cost 0.55 ms, throughput: 1.98 GB/s;
               (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,509] LMCache INFO: Storing KV cache for 8 out of 32 tokens (skip_leading_tokens=24)
               (EngineCore_DP0 pid=494270) [2025-09-30 01:12:36,510] LMCache INFO: Stored 8 out of total 8 tokens. size: 0.0011 gb, cost 0.43 ms, throughput: 2.57 GB/s; offload_time: 0.40 ms, put_time: 0.03 ms

            - **Total tokens 32**: The new prompt has 32 tokens after tokenization.
            - **LMCache hit tokens: 24**: 24 tokens (full 8-token chunks) were found in the cache from the first request that stored 31 tokens.
            - **Need to load: 8**: vLLM auto prefix caching uses block size 16; 16 tokens already sit in GPU RAM, so LMCache only loads 24-16=8.
            - **Why 24 hit tokens instead of 31?** LMCache hashes every 8 tokens (8, 16, 24, 31). It matches page-aligned chunks, so it uses the 24-token hash.
            - **Stored another 8 tokens**: The new 8 tokens form a full chunk and are stored for future reuse.

   .. tab-item:: SGLang

      **Install SGLang**

      .. code-block:: bash

         uv venv --python 3.12
         source .venv/bin/activate
         uv pip install --prerelease=allow lmcache "sglang"

      **Start SGLang with LMCache**

      .. code-block:: bash

         cat > lmc_config.yaml <<'EOF'
         chunk_size: 8  # demo only; use 256 for production
         local_cpu: true
         use_layerwise: true
         max_local_cpu_size: 10  # GB
         EOF

         export LMCACHE_CONFIG_FILE=$PWD/lmc_config.yaml

         python -m sglang.launch_server \
           --model-path Qwen/Qwen3-8B \
           --host 0.0.0.0 \
           --port 30000 \
           --enable-lmcache

      .. note::
         Configure LMCache via the config file. See :doc:`../api_reference/configurations` for the full list.

      **Test** -- open a new terminal and send two requests whose prompts
      share a prefix:

      **First request**

      .. code-block:: bash

         curl http://localhost:30000/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-8B",
             "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts"}],
             "max_tokens": 100,
             "temperature": 0.7
           }'

      **Second request**

      .. code-block:: bash

         curl http://localhost:30000/v1/chat/completions \
           -H "Content-Type: application/json" \
           -d '{
             "model": "Qwen/Qwen3-8B",
             "messages": [{"role": "user", "content": "Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models"}],
             "max_tokens": 100,
             "temperature": 0.7
           }'

      **You should see LMCache logs like this:**

      **First request** -- prompt plus generated tokens are stored:

      .. code-block:: text

         Prefill batch, #new-seq: 1, #new-token: 35, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 74, token usage: 0.00, cuda graph: True, gen throughput (token/s): 1.63, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 114, token usage: 0.00, cuda graph: True, gen throughput (token/s): 87.95, #queue-req: 0,
         LMCache INFO: Stored 128 out of total 135 tokens. size: 0.0195 GB, cost 12.8890 ms, throughput: 1.5153 GB/s (cache_engine.py:623:lmcache.v1.cache_engine)

      **Second request** -- Radix Cache and LMCache share the prefix; only the new portion is stored:

      .. code-block:: text

         Prefill batch, #new-seq: 1, #new-token: 10, #cached-token: 30, token usage: 0.00, #running-req: 0, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 64, token usage: 0.00, cuda graph: True, gen throughput (token/s): 8.29, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 104, token usage: 0.00, cuda graph: True, gen throughput (token/s): 87.95, #queue-req: 0,
         Decode batch, #running-req: 1, #token: 144, token usage: 0.00, cuda graph: True, gen throughput (token/s): 87.89, #queue-req: 0,
         LMCache INFO: Stored 112 out of total 140 tokens. size: 0.0171 GB, cost 11.1986 ms, throughput: 1.5261 GB/s (cache_engine.py:623:lmcache.v1.cache_engine)

      - **Total tokens 140**: SGLang stores KV cache for both prefill and decode tokens together, so total = 40 prompt + 100 generated = 140 tokens.
      - **Cached tokens: 30**: SGLang's Radix Attention Cache reused 30 tokens from the first request.
      - **LMCache hit tokens: 24**: LMCache detected 24 tokens (3 full 8-token chunks) stored from the first request. Since Radix Cache already provides 30 tokens in GPU memory, these 24 tokens don't need to be loaded from LMCache or stored again.
      - **New tokens: 10**: Only 10 prompt tokens need prefill computation (40 prompt - 30 cached = 10).
      - **Stored 112 out of 140**: 24 tokens (3 full chunks) are already in LMCache and skipped. Of the remaining 116 tokens, 112 (14 full 8-token chunks) are stored.

   .. tab-item:: TensorRT-LLM

      .. warning::
         The TensorRT-LLM integration is **not yet available in stable PyPI
         releases** of either project. Both must be installed from source
         (specifically, from each project's ``main`` / ``dev`` branch):

         - **LMCache** ``dev`` branch -- ``lmcache==0.4.4`` (the current
           PyPI release) does not include ``lmcache.integration.tensorrt_llm``.
         - **TensorRT-LLM** ``main`` branch -- the ``connector: "lmcache"``
           preset registry was added by `PR #12626
           <https://github.com/NVIDIA/TensorRT-LLM/pull/12626>`__ (merged
           April 2026) and is not in TRT-LLM 1.2.0.

         These instructions will be simplified to a plain ``pip install``
         once both projects ship a release containing the integration.

      **Install LMCache with TensorRT-LLM (from source)**

      Run inside the NVIDIA PyTorch container so CUDA / MPI / NCCL versions
      match what TensorRT-LLM expects:

      .. code-block:: bash

         docker run --gpus all -it --rm \
             -v $PWD:/workspace -w /workspace \
             nvcr.io/nvidia/pytorch:26.02-py3 bash

         python -m venv /workspace/venv --system-site-packages
         source /workspace/venv/bin/activate
         pip install -U pip wheel setuptools

         # TensorRT-LLM from main (heavy build -- can take 30+ minutes;
         # if NVIDIA publishes a nightly wheel that postdates April 2026,
         # `pip install --pre tensorrt-llm` is faster).
         pip install "tensorrt-llm @ git+https://github.com/NVIDIA/TensorRT-LLM.git@main"

         # LMCache from dev -- builds the c_ops CUDA extension during install.
         pip install "lmcache @ git+https://github.com/LMCache/LMCache.git@dev"

      Verify the install:

      .. code-block:: bash

         python -c "import lmcache.c_ops, lmcache.integration.tensorrt_llm"
         python -c "from tensorrt_llm._torch.pyexecutor.connectors.registry \
             import CONNECTOR_REGISTRY; print(list(CONNECTOR_REGISTRY))"
         # Should list 'lmcache' and 'lmcache-mp'.

      **Configure LMCache via environment variables**

      .. code-block:: bash

         export PYTHONHASHSEED=0  # required -- chunk hashing depends on stable hash()
         export LMCACHE_CHUNK_SIZE=256
         export LMCACHE_LOCAL_CPU=True
         export LMCACHE_MAX_LOCAL_CPU_SIZE=2.0  # GiB
         export LMCACHE_LOG_LEVEL=DEBUG       # so the connector logs below show up

      **Run TensorRT-LLM with the LMCache connector**

      Save as ``trtllm_lmcache_demo.py`` and run with ``python trtllm_lmcache_demo.py``.

      The ``if __name__ == "__main__":`` guard is **mandatory** -- TRT-LLM's
      ``LLM`` constructor spawns MPI worker processes that re-import this
      file, and an unguarded module body re-instantiates the engine and
      aborts with ``MPI_ABORT``.

      The 3-request shape (long prompt → different long prompt → original
      long prompt) is also mandatory for proving LMCache contributed:
      TRT-LLM has its own GPU block reuse, so a 2-request "shared prefix"
      demo gets satisfied entirely from the GPU pool and never reaches
      LMCache. Sizing ``max_tokens=512`` forces the second request to
      evict the first request's blocks, so the third request must fall
      through to LMCache.

      .. code-block:: python

         from tensorrt_llm import LLM, SamplingParams
         from tensorrt_llm.llmapi.llm_args import (
             KvCacheConfig, KvCacheConnectorConfig,
         )

         LONG_PROMPT_A = "..."  # ~400 tokens of prose, e.g. an article
         LONG_PROMPT_B = "..."  # ~400 tokens of *different* prose


         def main():
             llm = LLM(
                 model="Qwen/Qwen2-1.5B-Instruct",
                 backend="pytorch",
                 # Tiny GPU KV pool so prompt B forces eviction of prompt A.
                 kv_cache_config=KvCacheConfig(
                     enable_block_reuse=True, max_tokens=512,
                 ),
                 kv_connector_config=KvCacheConnectorConfig(connector="lmcache"),
             )
             sp = SamplingParams(max_tokens=32)

             # Request 1 -- cold cache, KV stored to LMCache.
             llm.generate([LONG_PROMPT_A], sp)
             # Request 2 -- different prompt, evicts request 1's GPU blocks.
             llm.generate([LONG_PROMPT_B], sp)
             # Request 3 -- original prompt, GPU blocks gone, must hit LMCache.
             out = llm.generate([LONG_PROMPT_A], sp)
             print(out[0].outputs[0].text)


         if __name__ == "__main__":
             main()

      **You should see LMCache logs like this** on the third ``generate``
      call (DEBUG-level for the connector lines, INFO-level for the
      ``Retrieved`` line):

      .. code-block:: text

         LMCache TRT-LLM scheduler: req 2050 ... trt_matched=64 lmcache_cached=256 new_matched=192
         LMCache INFO: Retrieved 256 out of 382 required tokens (from 382 total tokens). size: 0.0068 gb, cost 0.5448 ms, throughput: 12.5468 GB/s
         LMCache TRT-LLM worker: start_load_kv retrieve=0.786ms num_loads=1

      - ``trt_matched``: tokens TRT-LLM matched in its own GPU block pool
        (small after eviction).
      - ``lmcache_cached``: tokens LMCache has cached for this request.
      - ``new_matched``: tokens LMCache supplies *beyond* what TRT-LLM
        already had. **Must be non-zero** to prove LMCache contributed.
      - ``num_loads=1`` on ``start_load_kv``: confirms the worker actually
        copied KV from LMCache into the GPU pool.

      Output text matching between requests 1 and 3 alone is **not**
      sufficient evidence -- TRT-LLM block reuse can produce identical
      output on its own. Always check for the three log lines above.

🎉 **You now have LMCache caching and reusing KV caches across all three engines.**

Next Steps
----------

- **Performance Testing**: Try the :doc:`benchmarking` section to experience LMCache's performance benefits with more comprehensive examples
- **More Examples**: Explore the :doc:`quickstart/index` section for detailed examples including KV cache sharing across instances and disaggregated prefill