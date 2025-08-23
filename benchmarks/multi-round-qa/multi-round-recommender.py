# This is a script that is hardware agnostic in providing you a minimal
# deployment and benchmark workload that allows you to gleam insights about 
# how LMCache speeds up inference. If you would like a personalized deployment 
# configuration for your model and hardware and how to optimize traffic routing
# in your orchestration layer etc., please contact the LMCache team directly

# underlying principle: QPS and num users are configurable in real deployments

# Example: python multi-round-recommender.py
# "Engine core proc EngineCore_0 died unexpectedly, shutting down client." is expected because we hacked the collective RPC interface

from vllm import LLM, SamplingParams
import time
import subprocess
import psutil

# fixed params for this simple pedagogical benchmark
shared_system_prompt_tokens = 1000
user_conversation_history_tokens = 10000
output_tokens = 100
num_rounds = 5
# overestimate a little (shared system prompt will be reused in vllm)
# avoid a workload which will cause preemption in vllm scheduler
max_per_request_tokens = 12000
# llama 8B
kv_cache_usage_in_one_request_gb = 1.8
# 2000 extra token buffer for max_model_len
max_model_len = max_per_request_tokens + 2000
model = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# don't touch any other configuration on vllm like gpu memory utilization
# or enforce eager to keep the benchmark organic
# load format dummy just to get stats
llm = LLM(model=model,
          max_model_len=max_model_len,
          load_format="dummy")

memory_info = llm.collective_rpc("determine_available_memory")
print(f"type of memory_info: {type(memory_info)}")
# stay conservative in back-of-envelope estimations
GB_available = memory_info[0] * 0.9 / 1024**3
print(f"Available KV cache memory: {GB_available} GB")

# concatenated hi's are tokenized as epxected on meta-llama/Meta-Llama-3.1-8B-Instruct
dummy_prompt = "hi" * max_per_request_tokens

sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=output_tokens)
# estimate around how long each request will be in the EngineCore
# (IMPORTANT: not TTFT but E2E)
start = time.perf_counter()
llm.generate(dummy_prompt, sampling_params)
end = time.perf_counter()
e2e_time_in_engine = end - start
print(f"E2E time in engine: {e2e_time_in_engine} seconds")

vm = psutil.virtual_memory()
available_pinnable_cpu_size_gb = vm.available / 1024**3
memlock = subprocess.check_output("ulimit -l", shell=True, text=True).strip()
if memlock != "unlimited":
    print(f"OS restricts pinnable CPU size to {memlock} bytes")
    available_pinnable_cpu_size_gb = int(memlock) / 1024**3
else: 
    print(f"You have unlimited pinnable CPU size")
print(f"CPU DRAM that can be page-locked: {available_pinnable_cpu_size_gb} GB")

max_num_reqs = int(GB_available / kv_cache_usage_in_one_request_gb) + 1
# how many users should we route to this single engine? 
num_users = int(min(max_num_reqs * 2, (available_pinnable_cpu_size_gb * 0.75) / kv_cache_usage_in_one_request_gb)) + 1

# stack don't stack requests too aggressively
qps = (1 / e2e_time_in_engine) * 1.5
time_to_run = int(num_users * num_rounds  * qps) + 1

vllm_recommendation_string = (
    f"vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \\\n"
    f"--load-format dummy \\\n"
    f"--max-num-seqs {max_num_reqs - 1}"
)

lmcache_recommendation_string = (
    f"LMCACHE_MAX_LOCAL_CPU_SIZE={int((num_users * kv_cache_usage_in_one_request_gb) * 1.2)} \\\n"
    f"vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \\\n"
    f"--max-num-seqs {max_num_reqs - 1} \\\n"
    f"--load-format dummy \\\n"
    f"--kv-transfer-config \\\n"
    f"\'{{\"kv_connector\": \"LMCacheConnectorV1\", \"kv_role\": \"kv_both\"}}\'"
)

multi_round_qa_recommendation_string = (
    f"python multi-round-qa.py \\\n"
    f"--num-users {num_users} \\\n"
    f"--num-rounds {num_rounds} \\\n"
    f"--qps {qps} \\\n"
    f"--shared-system-prompt {shared_system_prompt_tokens} \\\n"
    f"--user-history-prompt {user_conversation_history_tokens} \\\n"
    f"--answer-len {output_tokens} \\\n"
    f"--model meta-llama/Meta-Llama-3.1-8B-Instruct \\\n"
    f"--base-url http://localhost:8000/v1 \\\n"
    f"--time {int(time_to_run) + 5}"
)


print(
    f"\n\nDeployment and Workload Recommendations: \n"
    f"----------------------------------------\n\n"
    f"1. vLLM Deployment: \n"
    f"-----------------\n"
    f"{vllm_recommendation_string}"
    f"\n\n"
    f"2. LMCache Deployment: \n"
    f"--------------------\n"
    f"{lmcache_recommendation_string}"
    f"\n\n"
    f"3. Multi-Round QA Workload Generation: \n"
    f"----------------------------------------\n"
    f"{multi_round_qa_recommendation_string}\n\n\n"
)


print(
    f"\"Engine core proc EngineCore_0 died unexpectedly, shutting down client.\""
    f"may occur because we hacked the collective RPC interface"    
)