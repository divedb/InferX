# Qwen3-0.6B optimization results

Historical report. GPU access was restored on September 22; see
[OPTIMIZATION_20260922.md](OPTIMIZATION_20260922.md) for subsequent measurements.

**Target not achieved.** None of the 24 fully measured configurations exceeds vLLM throughput. The faster attention candidate has unresolved numerical equivalence; it is opt-in. GPU access became blocked by the session permissions on continuation, so full graph benchmarking and numerical investigation could not be completed. No missing measurements are estimated.

## What was measured

RTX 4080 SUPER, driver 591.86, WSL2; vLLM 0.29.0, PyTorch 2.13.0. Identical local unquantized Qwen3-0.6B BF16 weights and KV, greedy sampling, exact token-ID prompts, 4096 scheduled-token budget, 2048 KV blocks of 16 tokens (3584 MiB KV). 24 configurations × 3 measured repetitions for each of four stages: **288 trials**, plus one full unmeasured warmup per configuration and stage. Engines ran serially. Full conditions, metric definitions and reproduction commands are in [README.md](README.md).

vLLM uses its V1 model runner because V2 fails on this WSL installation with UVA unavailable. Compilation and CUDA graphs remain enabled; prefix caching is disabled in both engines. vLLM startup compilation initially selected an obsolete nvcc; explicitly selecting `/usr/local/cuda` resolved it. Failed startup logs are preserved and excluded.

## Optimization record

1. **GPU greedy sampling — enabled by default.** Two-stage parallel argmax preserves lowest-index tie breaking and the original non-finite-value behavior. Reuses reduction workspace and a pinned host output buffer; copies four bytes per request instead of 303,872 bytes of BF16 logits and eliminates the CPU vocabulary scan. Every output token matches the original InferX baseline in all 72 measured trials. Per-case end-to-end improvement: 1.002–1.338×. The smallest differences are within likely measurement noise.

2. **FlashInfer paged attention — experimental, disabled by default.** Replaces serial per-key attention with tensor-core prefill and parallel decode. Builds device-side tile metadata once per forward and reuses it across layers, with persistent workspace. Falls back to the original kernel for unsupported geometry. Passed a double-precision reference test at the optimized 128-dimensional head size, including GQA, ragged cached prefixes, decode, shuffled pages and page boundaries, at absolute tolerance 0.005. That local test does not establish full-model equivalence. End-to-end speedup over original InferX: 1.37–6.34×; vs vLLM: 0.57–0.88× (geometric mean 0.69×).

3. **Decode CUDA graphs — experimental, disabled by default.** Per-batch capture/replay after an eager warmup, stable language-model-head capacity, live device metadata, and graph cleanup. Custom models must opt into the stable-buffer contract. One completed smoke configuration (512 input, 128 output, batch/concurrency 4) took 488.59 ms and matched all eager-attention tokens. This is **one diagnostic trial**, with no matched memory measurement or complete matrix; it is excluded from every comparison plot and aggregate. No graph performance claim is made. Current opt-in code and final default gating compiled, but could not be rerun on the GPU after permissions changed.

## Profile evidence

An independent 4-request, 512-input/128-output diagnostic recorded 2450.87 ms in 7168 attention calls and 908.58 ms in 50,432 linear calls across warmup plus one trial. Attention was about 73% of the combined time of these two measured operator groups. CUDA-event instrumentation serializes operations and is never enabled in throughput runs; it is not a complete, unperturbed kernel breakdown.

Nsight CUDA API totals included 2629.57 ms in stream synchronization (72% of traced API time), 652.72 ms in kernel launch APIs and 235.39 ms in synchronous copies. WSL tracing supplied no GPU kernel activity or memory-transfer records; the empty exports and exporter log are retained. Synchronization wait overlaps GPU execution and must not be interpreted as removable CPU overhead. See [profiles](profiles).

## Correctness status

Exact greedy agreement is a failing gate, not a relaxed success criterion:

| Stage | Exact sequences, first repeat | Exact sequences, all paired repeats |
|---|---:|---:|
| inferx_baseline | 118/174 | 355/522 |
| inferx_gpu_sampling | 118/174 | 355/522 |
| inferx_flash_attention | 110/174 | 344/522 |

vLLM outputs vary across repeats in 7 of 24 configurations; all measured InferX stages are deterministic across their repeats. The cause has not been established. Free-running token agreement after the first divergence is not a numerical error metric. Next validation must replay identical prefixes and compare logits and argmax margins against vLLM and an independent reference, including first-divergence positions. Neither BF16 rounding nor sampling ties have been assumed to explain the differences.

`analyze.py --require-exact` saves the plots/results and exits nonzero for these data. It checks workload/checkpoint hashes, hardware/software identity, complete unique case/repeat coverage, sequence counts, exact output lengths, valid token IDs and monotonic finite timestamps. No tolerance was widened and no workloads were removed.

## All configurations

End-to-end **output tokens/s**, medians of three trials. “B” is maximum scheduled sequences; “C” is requests arriving together, including queued requests. Each measured attention ratio is below 0.95, so all 24 underperform even under a ±5% descriptive matching band. This band describes speed only; it changes no correctness requirement. Trial ranges are saved in CSV and plotted as whiskers.

| Input | Output | B | C | vLLM | Original InferX | GPU sampling | Experimental attention | Attention/vLLM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 1 | 1 | 347.4 | 164.6 | 165.9 | 236.4 | 0.680× |
| 128 | 32 | 4 | 4 | 1279.1 | 548.5 | 588.7 | 763.1 | 0.597× |
| 128 | 32 | 4 | 8 | 1280.5 | 548.3 | 594.9 | 750.8 | 0.586× |
| 128 | 32 | 16 | 16 | 3806.1 | 1493.1 | 1969.7 | 2881.7 | 0.757× |
| 128 | 128 | 1 | 1 | 374.4 | 147.7 | 148.1 | 226.1 | 0.604× |
| 128 | 128 | 4 | 4 | 1346.4 | 513.1 | 546.6 | 771.1 | 0.573× |
| 128 | 128 | 4 | 8 | 1343.2 | 518.8 | 549.2 | 787.3 | 0.586× |
| 128 | 128 | 16 | 16 | 4301.0 | 1565.3 | 2094.2 | 3165.5 | 0.736× |
| 512 | 32 | 1 | 1 | 346.0 | 76.3 | 77.1 | 206.6 | 0.597× |
| 512 | 32 | 4 | 4 | 1004.9 | 231.8 | 236.9 | 732.8 | 0.729× |
| 512 | 32 | 4 | 8 | 999.1 | 231.3 | 238.7 | 696.4 | 0.697× |
| 512 | 32 | 16 | 16 | 2090.7 | 471.2 | 509.2 | 1791.8 | 0.857× |
| 512 | 128 | 1 | 1 | 367.7 | 76.2 | 77.7 | 210.5 | 0.572× |
| 512 | 128 | 4 | 4 | 1153.3 | 270.3 | 279.9 | 789.2 | 0.684× |
| 512 | 128 | 4 | 8 | 1152.9 | 269.9 | 280.0 | 799.1 | 0.693× |
| 512 | 128 | 16 | 16 | 2979.4 | 772.1 | 876.1 | 2413.8 | 0.810× |
| 1024 | 32 | 1 | 1 | 312.8 | 41.8 | 42.0 | 198.5 | 0.635× |
| 1024 | 32 | 4 | 4 | 751.6 | 106.3 | 107.9 | 595.5 | 0.792× |
| 1024 | 32 | 4 | 8 | 752.0 | 106.3 | 107.4 | 594.4 | 0.790× |
| 1024 | 32 | 16 | 16 | 1258.1 | 174.1 | 177.0 | 1104.0 | 0.878× |
| 1024 | 128 | 1 | 1 | 349.0 | 45.6 | 46.0 | 206.9 | 0.593× |
| 1024 | 128 | 4 | 4 | 972.2 | 153.4 | 157.6 | 717.3 | 0.738× |
| 1024 | 128 | 4 | 8 | 972.1 | 152.7 | 155.8 | 720.3 | 0.741× |
| 1024 | 128 | 16 | 16 | 2071.7 | 381.8 | 405.6 | 1746.8 | 0.843× |

## Latency, phase rates and memory

All six requested metric families appear in the [Matplotlib comparison](comparison.png) ([PDF](comparison.pdf)) and [summary.csv](summary.csv); mean and p95 TTFT/ITL are included in the CSV. The CSV also includes trial throughput ranges. Decode rates use the observed decode window; prefill rates are effective service rates through the last request’s first token. With mixed prefill/decode and queueing, both include overlapping work and are **not isolated GPU phase rates**.

For 512 input / 128 output / B4 / C4, the experimental attention candidate vs vLLM:

| Metric | InferX attention | vLLM |
|---|---:|---:|
| End-to-end output tokens/s | 789.25 | 1153.31 |
| Decode-window tokens/s | 818.12 | 1220.74 |
| Effective prefill input tokens/s | 73725.87 | 73687.18 |
| Mean TTFT, ms | 27.78 | 27.37 |
| Mean ITL, ms | 4.89 | 3.28 |
| Sampled peak device memory above idle, MiB | 5175.10 | 5432.25 |

The attention candidate improves TTFT in some cases but still has higher ITL. GPU memory above idle is approximately 5107–5175 MiB for the attention candidate and 5432–5521 MiB for vLLM. Memory is whole-device NVML sampled every 5 ms, includes initialization/warmup, and is shared by every case in the same batch-size process. It is a sampled high-water observation, not an exact per-case or per-process peak. Display/OS allocations are included and short peaks can be missed.

## Validation and remaining work

- Completed before GPU access was lost: sampling BF16/FP32 tests with ties/NaNs/infinities, attention reference tests, runner tests, all four benchmark matrices, and one graph smoke test.
- Completed after continuation: current Release build; scheduler, scheduler-output and CPU model-runner CTest targets; six benchmark-integrity tests; archived-data validation and plots. The strict cross-engine correctness check correctly fails.
- Still required: teacher-forced full-model numerical diagnosis; graph replay testing across changing batches, request turnover and KV page transitions; full graph matrix; fresh matched vLLM runs to bracket drift; sustained concurrency/arrival-pattern tests; isolated phase profiling and more precise per-case memory measurement.
- Current session cannot access the GPU (`GPU access blocked by the operating system`) and prohibits escalation. No unsupported GPU-access workaround was attempted.

Current defaults retain the token-preserving GPU-sampling optimization. Attention requires `--flash-attention` in the benchmark driver; graphs require `--cuda-graphs`. The historical attention measurements predate this default-off gate. No GPU timings are claimed for the final source after gating and benchmark-validation edits.

Artifacts preserve all trials, outputs, timestamps, memory samples, commands, hashes, and tracked-source diffs. Early runs did not archive untracked source files; the updated driver now records a source archive for future runs. The existing baseline prompts are exact-length synthetic token sequences, not an application quality suite. Three consecutive repetitions on one WSL GPU support only limited statistical and hardware generalization.
