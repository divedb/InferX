"""Generate a candid report from the measured stages; does not launch a GPU job."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
s = json.loads((ROOT/'summary.json').read_text())
a = json.loads((ROOT/'agreement.json').read_text())
cases = json.loads((ROOT/'workload.json').read_text())['cases']
ref = s['vllm_baseline']
base = s['inferx_baseline']
sampling = s['inferx_gpu_sampling']
fast = s['inferx_flash_attention']
ratios = [fast[c]['e2e_tokens_s']/ref[c]['e2e_tokens_s'] for c in ref]
gains = [fast[c]['e2e_tokens_s']/base[c]['e2e_tokens_s'] for c in ref]
sgains = [sampling[c]['e2e_tokens_s']/base[c]['e2e_tokens_s'] for c in ref]
lines = ['# Qwen3-0.6B optimization results', '',
'**Target not achieved.** None of the 24 fully measured configurations exceeds vLLM throughput. '
'The faster attention candidate has unresolved numerical equivalence; it is opt-in. '
'GPU access became blocked by the session permissions on continuation, so full graph benchmarking '
'and numerical investigation could not be completed. No missing measurements are estimated.', '',
'## What was measured', '',
'RTX 4080 SUPER, driver 591.86, WSL2; vLLM 0.29.0, PyTorch 2.13.0. '
'Identical local unquantized Qwen3-0.6B BF16 weights and KV, greedy sampling, exact token-ID prompts, '
'4096 scheduled-token budget, 2048 KV blocks of 16 tokens (3584 MiB KV). '
'24 configurations × 3 measured repetitions for each of four stages: **288 trials**, '
'plus one full unmeasured warmup per configuration and stage. Engines ran serially. '
'Full conditions, metric definitions and reproduction commands are in [README.md](README.md).', '',
'vLLM uses its V1 model runner because V2 fails on this WSL installation with UVA unavailable. '
'Compilation and CUDA graphs remain enabled; prefix caching is disabled in both engines. '
'vLLM startup compilation initially selected an obsolete nvcc; explicitly selecting '
'`/usr/local/cuda` resolved it. Failed startup logs are preserved and excluded.', '',
'## Optimization record', '',
'1. **GPU greedy sampling — enabled by default.** Two-stage parallel argmax preserves '
'lowest-index tie breaking and the original non-finite-value behavior. Reuses reduction '
'workspace and a pinned host output buffer; copies four bytes per request instead of '
'303,872 bytes of BF16 logits and eliminates the CPU vocabulary scan. '
'Every output token matches the original InferX baseline in all 72 measured trials. '
 f'Per-case end-to-end improvement: {min(sgains):.3f}–{max(sgains):.3f}×. '
'The smallest differences are within likely measurement noise.', '',
'2. **FlashInfer paged attention — experimental, disabled by default.** Replaces serial '
'per-key attention with tensor-core prefill and parallel decode. Builds device-side tile '
'metadata once per forward and reuses it across layers, with persistent workspace. '
'Falls back to the original kernel for unsupported geometry. Passed a double-precision '
'reference test at the optimized 128-dimensional head size, including GQA, ragged '
'cached prefixes, decode, shuffled pages and page boundaries, at absolute tolerance 0.005. '
'That local test does not establish full-model equivalence. '
 f'End-to-end speedup over original InferX: {min(gains):.2f}–{max(gains):.2f}×; '
 f'vs vLLM: {min(ratios):.2f}–{max(ratios):.2f}× '
 f'(geometric mean {math.exp(sum(map(math.log,ratios))/len(ratios)):.2f}×).', '',
'3. **Decode CUDA graphs — experimental, disabled by default.** Per-batch capture/replay '
'after an eager warmup, stable language-model-head capacity, live device metadata, '
'and graph cleanup. Custom models must opt into the stable-buffer contract. One completed '
'smoke configuration (512 input, 128 output, batch/concurrency 4) took 488.59 ms and '
'matched all eager-attention tokens. This is **one diagnostic trial**, with no matched '
'memory measurement or complete matrix; it is excluded from every comparison plot and '
'aggregate. No graph performance claim is made. Current opt-in code and final default '
'gating compiled, but could not be rerun on the GPU after permissions changed.', '',
'## Profile evidence', '',
'An independent 4-request, 512-input/128-output diagnostic recorded 2450.87 ms in '
'7168 attention calls and 908.58 ms in 50,432 linear calls across warmup plus one trial. '
'Attention was about 73% of the combined time of these two measured operator groups. '
'CUDA-event instrumentation serializes operations and is never enabled in throughput runs; '
'it is not a complete, unperturbed kernel breakdown.', '',
'Nsight CUDA API totals included 2629.57 ms in stream synchronization (72% of traced '
'API time), 652.72 ms in kernel launch APIs and 235.39 ms in synchronous copies. '
'WSL tracing supplied no GPU kernel activity or memory-transfer records; the empty '
'exports and exporter log are retained. Synchronization wait overlaps GPU execution '
'and must not be interpreted as removable CPU overhead. See [profiles](profiles).', '',
'## Correctness status', '',
'Exact greedy agreement is a failing gate, not a relaxed success criterion:', '',
'| Stage | Exact sequences, first repeat | Exact sequences, all paired repeats |',
'|---|---:|---:|']
for tag in ('inferx_baseline','inferx_gpu_sampling','inferx_flash_attention'):
    aa=a[tag]
    lines.append(f"| {tag} | {sum(v['exact_sequences'] for v in aa.values())}/174 | {sum(v['exact_sequences_all_trials'] for v in aa.values())}/522 |")
lines += ['',
'vLLM outputs vary across repeats in 7 of 24 configurations; all measured InferX stages '
'are deterministic across their repeats. The cause has not been established. '
'Free-running token agreement after the first divergence is not a numerical error metric. '
'Next validation must replay identical prefixes and compare logits and argmax margins '
'against vLLM and an independent reference, including first-divergence positions. '
'Neither BF16 rounding nor sampling ties have been assumed to explain the differences.', '',
'`analyze.py --require-exact` saves the plots/results and exits nonzero for these data. '
'It checks workload/checkpoint hashes, hardware/software identity, complete unique '
'case/repeat coverage, sequence counts, exact output lengths, valid token IDs and '
'monotonic finite timestamps. No tolerance was widened and no workloads were removed.', '',
'## All configurations', '',
'End-to-end **output tokens/s**, medians of three trials. “B” is maximum scheduled '
'sequences; “C” is requests arriving together, including queued requests. '
'Each measured attention ratio is below 0.95, so all 24 underperform even under a '
'±5% descriptive matching band. This band describes speed only; it changes no '
'correctness requirement. Trial ranges are saved in CSV and plotted as whiskers.', '',
'| Input | Output | B | C | vLLM | Original InferX | GPU sampling | Experimental attention | Attention/vLLM |',
'|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in cases:
    k=c['id']
    lines.append(f"| {c['input_len']} | {c['output_len']} | {c['batch']} | {c['concurrency']} | {ref[k]['e2e_tokens_s']:.1f} | {base[k]['e2e_tokens_s']:.1f} | {sampling[k]['e2e_tokens_s']:.1f} | {fast[k]['e2e_tokens_s']:.1f} | {fast[k]['e2e_tokens_s']/ref[k]['e2e_tokens_s']:.3f}× |")
lines += ['', '## Latency, phase rates and memory', '',
'All six requested metric families appear in the [Matplotlib comparison](comparison.png) '
'([PDF](comparison.pdf)) and [summary.csv](summary.csv); mean and p95 TTFT/ITL '
'are included in the CSV. The CSV also includes trial throughput ranges. '
'Decode rates use the observed decode window; prefill rates are effective service '
'rates through the last request’s first token. With mixed prefill/decode and queueing, '
'both include overlapping work and are **not isolated GPU phase rates**.', '',
'For 512 input / 128 output / B4 / C4, the experimental attention candidate vs vLLM:', '',
'| Metric | InferX attention | vLLM |', '|---|---:|---:|']
k='p512_o128_b4_c4'
for metric,title in [('e2e_tokens_s','End-to-end output tokens/s'),('decode_tokens_s','Decode-window tokens/s'),('prefill_tokens_s','Effective prefill input tokens/s'),('ttft_ms','Mean TTFT, ms'),('itl_ms','Mean ITL, ms'),('peak_device_delta_mib','Sampled peak device memory above idle, MiB')]:
    lines.append(f'| {title} | {fast[k][metric]:.2f} | {ref[k][metric]:.2f} |')
lines += ['',
'The attention candidate improves TTFT in some cases but still has higher ITL. '
'GPU memory above idle is approximately 5107–5175 MiB for the attention candidate '
'and 5432–5521 MiB for vLLM. Memory is whole-device NVML sampled every 5 ms, includes '
'initialization/warmup, and is shared by every case in the same batch-size process. '
'It is a sampled high-water observation, not an exact per-case or per-process peak. '
'Display/OS allocations are included and short peaks can be missed.', '',
'## Validation and remaining work', '',
'- Completed before GPU access was lost: sampling BF16/FP32 tests with ties/NaNs/infinities, '
'attention reference tests, runner tests, all four benchmark matrices, and one graph smoke test.',
'- Completed after continuation: current Release build; scheduler, scheduler-output and '
'CPU model-runner CTest targets; six benchmark-integrity tests; archived-data validation '
'and plots. The strict cross-engine correctness check correctly fails.',
'- Still required: teacher-forced full-model numerical diagnosis; graph replay testing '
'across changing batches, request turnover and KV page transitions; full graph matrix; '
'fresh matched vLLM runs to bracket drift; sustained concurrency/arrival-pattern tests; '
'isolated phase profiling and more precise per-case memory measurement.',
'- Current session cannot access the GPU (`GPU access blocked by the operating system`) '
'and prohibits escalation. No unsupported GPU-access workaround was attempted.', '',
'Current defaults retain the token-preserving GPU-sampling optimization. Attention '
'requires `--flash-attention` in the benchmark driver; graphs require `--cuda-graphs`. '
'The historical attention measurements predate this default-off gate. No GPU timings '
'are claimed for the final source after gating and benchmark-validation edits.', '',
'Artifacts preserve all trials, outputs, timestamps, memory samples, commands, hashes, '
'and tracked-source diffs. Early runs did not archive untracked source files; '
'the updated driver now records a source archive for future runs. The existing '
'baseline prompts are exact-length synthetic token sequences, not an application '
'quality suite. Three consecutive repetitions on one WSL GPU support only limited '
'statistical and hardware generalization.', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
