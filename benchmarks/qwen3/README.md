# Reproducible Qwen3-0.6B comparison

This directory is one model-specific use case of the generic benchmark
framework in [../inferx_bench/](../inferx_bench/) and
[../bench.py](../bench.py); the entry points below are thin wrappers that
pin the frozen Qwen3 workload and these results roots. Adding a different
model means a registry entry plus its own directory — see
[../README.md](../README.md).

Current CUDA attention defaults to **FlashInfer**; the scalar attention kernel has
been removed. `--attention-backend flashinfer` selects it explicitly; `default` and
`flash` are compatibility aliases. The old attention environment flag no longer
controls execution. See [FLASHINFER_DEFAULT.md](FLASHINFER_DEFAULT.md) for supported
geometry, correctness validation, and remaining numerical limits. Historical
results below retain their original implementation and conditions.

The latest optimization runs and remaining correctness limits are recorded in
[OPTIMIZATION_20260922.md](OPTIMIZATION_20260922.md). The candidate can be reproduced with:

```sh
cmake --build build-cuda13 --target inferx -j 8
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag optimized_new_run \
  --flash-attention --cuda-graphs --decode-linear --split-decode --packed-projections
```

The graph, projection and split-decode flags remain experimental; FlashInfer
attention is now the default. Throughput improvements do not establish exact
cross-engine token equivalence. Metadata uploads and the rounding-preserving
normalization fusions apply to the default path as well. The driver records every
flag and rejects an InferX binary that changes during a benchmark run.

See [REPORT.md](REPORT.md) for historical results and the blockers at that time.
See [NUMERICAL.md](NUMERICAL.md) for the reference-rounding corrections, identical-prefix
logit replay, and the validation still required before new performance claims.

Run from the repository root using the existing Python environment and CUDA Release build
(requires GPU access; existing tags cannot be overwritten):

```sh
cmake --build build-cuda13 --target inferx -j 8
python/.venv/bin/python benchmarks/qwen3/run.py --engine vllm --tag vllm_baseline
# Current default is GPU sampling with FlashInfer attention and eager decode.
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag sampling_rerun
# FlashInfer is the default; additional optimizations remain opt-in.
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag attention_rerun --flash-attention
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag graphs_candidate --flash-attention --cuda-graphs
python/.venv/bin/python benchmarks/qwen3/analyze.py vllm_baseline sampling_rerun attention_rerun graphs_candidate --require-exact
```

Never run engines or GPU tests concurrently. Tags must be new directories; failed runs
are retained. `make_workload.py` documents how the checked-in workload was generated;
do not regenerate/change it between comparisons. Its SHA256, checkpoint SHA256,
config hash, binary hash, source diff, software versions, GPU, driver, commands,
raw outputs and token timestamps are saved with each run. Early runs saved tracked-file
diffs only, so retain this repository and its added files. Future runs also save a
source archive, explicit experimental flags and relevant runtime environment settings.
A process lock prevents two instances of this driver from running simultaneously;
it cannot prevent unrelated GPU applications from running. Profiling is forcibly disabled
in throughput jobs. The NVML monitor fails the run if it cannot collect memory data.

## Fixed conditions

- Local `models/Qwen3-0.6B`, unquantized BF16 weights/activations/KV, one GPU.
- 24 Cartesian configurations: inputs 128/512/1024, outputs 32/128, and
  (max scheduled sequences, simultaneous requests) = (1,1), (4,4), (4,8), (16,16).
- Same token IDs, greedy argmax, temperature 0, no penalties, ignore EOS,
  exactly the requested output length. No tokenization/detokenization or networking
  inside either measured path. Request admission and engine scheduling ARE timed.
- Token budget 4096; paged KV block size 16; 2048 blocks = 32768 KV tokens =
  3,758,096,384 KV bytes. No prefix reuse, speculative decoding, or quantization.
- One complete warmup per configuration and three complete measured repetitions;
  initialization/JIT/graph construction during startup excluded from throughput.
  Lazy construction during warmup excluded equally. Medians and trial ranges saved.
- vLLM generation_config='vllm' avoids checkpoint generation defaults changing
  sampling. vLLM V1 model runner (`VLLM_USE_V2_MODEL_RUNNER=0`) is required because
  V2 fails with "UVA is not available" on WSL. Compilation and graphs remain on.
  gpu_memory_utilization=0.7 passes the startup guard; explicit KV bytes control
  actual KV allocation. The initial failures are preserved, never counted as data.
- Maximum exercised context is 1152 tokens. vLLM max_model_len=2048; InferX
  retains its checkpoint context limit but receives only the same fixed requests.

## Metrics and limits

All timestamps are wall-clock times when the engine returns token IDs, including
scheduling and CPU overhead. InferX observes samples only when the scheduler has
completed the prompt; samples from partial prefill chunks are not output tokens.
All requests arrive together at t=0, so TTFT includes queueing.

- End-to-end throughput: all output tokens / full request-set elapsed time.
- Decode-window throughput: output tokens except the first per request / interval
  from the earliest first token to the final token. With queueing/mixed scheduling,
  this includes intervening prefill and idle time; it is not isolated kernel speed.
- Effective prefill throughput: all prompt tokens / time until every request has
  received its first token. Includes sampling, queueing and intervening decode.
  This is a service-level prefill measure, not isolated attention/GEMM throughput.
- TTFT mean and p95 over requests; ITL mean and p95 of actual consecutive token
  arrival differences, without subtracting independent prefill measurements.
- Peak GPU memory: whole-device NVML used bytes sampled every 5 ms, including
  initialization, warmup, and measurements; baseline and absolute peak also saved.
  Each batch-size process's peak applies to all its configurations. WSL does not
  provide reliable per-process accounting. Display/OS activity is included and
  short peaks may be missed. Do not label this as exact allocator peak memory.

The benchmark exercises a synchronized burst with queueing, not a network server,
Poisson arrivals, or a sustained closed-loop client load. Results apply only to
these stated conditions. Synthetic exact-length prompts truncate/repeat the existing
prompt corpus and are not a representative application-quality evaluation.

`analyze.py` validates complete case sets and output lengths, reports exact greedy
agreement and first divergence, and records nondeterminism. A performance number
alone is not a correctness pass. Exact token agreement is retained as a strict
check; any numerical tolerance checks are additional evidence, not a replacement.

vLLM's incremental [LLMEngine API](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/llm_engine.py)
is used to observe real first-token and inter-token timings; installed source is
the authority for the specific API version recorded in each run.

## Validation commands

```sh
python3 benchmarks/inferx_bench/test_validate.py
ctest --test-dir build-cuda13 --output-on-failure -R '^(scheduler_test|scheduler_output_test|model_runner_test)$'
# Requires GPU access:
env LD_LIBRARY_PATH=/usr/lib/wsl/lib ctest --test-dir build-cuda13 --output-on-failure -R '^ops_test$'
# Existing saved stages; exits nonzero for the unresolved equivalence failure:
MPLCONFIGDIR=/tmp/inferx-matplotlib python/.venv/bin/python benchmarks/qwen3/analyze.py vllm_baseline inferx_baseline inferx_gpu_sampling inferx_flash_attention --require-exact
python3 benchmarks/qwen3/report.py
```

The original InferX baseline refers to the unoptimized binary measured before GPU
sampling was implemented. Running current source without experimental flags does
not recreate that old baseline. Historical stage diffs and binary hashes are under
`results/`; current defaults and remaining validation are described in REPORT.md.
`report.py` specifically summarizes the four historical measured tags above.

## FlashInfer-default HTTP length sweep

The [2026-09-22 serving comparison](serve_results/varied_lengths_20260922/REPORT.md)
uses `vllm bench serve` against both InferX and vLLM across input lengths
128/512/1024, output lengths 32/128/512, and concurrency 1/16, with two scheduled
trials per configuration. Both engines use eager execution and matching cache
and scheduler capacities. The report retains unsuccessful trials alongside
conditional throughput and latency measurements; raw commands, logs, result
JSON, and reproducible analysis are in the same artifact directory.

The [2026-09-23 rerun](serve_results/rerun_20260923/REPORT.md) repeats the full
18-case HTTP matrix with two trials per engine using an immutable snapshot of
the current binary. Its [concurrency investigation](serve_results/rerun_20260923/CONCURRENCY.md)
adds mixed streaming/non-streaming load with 64 clients, disconnect recovery,
active-request shutdown, constrained-KV serial/concurrent controls, and a
model-free HTTP executor probe. Performance trials and intentionally adverse
reproductions are reported separately; production source was left unchanged.
