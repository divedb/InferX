# September 22 optimization measurements

The final experimental candidate exceeds vLLM **aggregate throughput** on the fixed
24-case Qwen3-0.6B suite: **1.053×** against the later fresh vLLM run and **1.084×**
against the earlier fresh run (geometric mean of per-case throughput ratios).
It is **1.215×** the graph-enabled InferX implementation at the start of this session.
This is not an unconditional replacement claim: **exact cross-engine token equivalence
still fails**, and **4 of 24** cases remain slower than the later vLLM run.

## Conditions and reproducibility

RTX 4080 SUPER on WSL, local Qwen3-0.6B, BF16 weights/activations/KV, greedy generation,
unchanged token-ID workload, token budget, KV capacity, and requested output lengths.
Every full performance tag contains 24 cases × 3 measured repetitions plus a complete
warmup per case. Initialization and warmup are excluded equally. Engines ran serially.
See [README.md](README.md) for the exact benchmark contract.

Fresh reference tags: `vllm_fresh_20260922` before optimization and
`vllm_after_20260922` after the main optimization series. The same pre-vectorization
InferX binary was measured in `inferx_final_20260922` and `inferx_confirm_20260922`;
they reached 1.068× and 1.073× the later vLLM run. The final retained vector-cache
change is `inferx_vector_cache_20260922`, reaching 1.053×. This spread, and the
reference-to-reference variation, limit claims for cases near parity. No confidence
interval or statistically significant per-case win is asserted from three repetitions.
The GPU also serves the Windows display. A pre-existing CUDA diagnostic process was
sleeping on a pipe; it was not terminated or counted as benchmark work.

```sh
cmake --build build-engine --target inferx -j 8
python/.venv/bin/python benchmarks/qwen3/run.py --engine vllm --tag new_reference
python/.venv/bin/python benchmarks/qwen3/run.py --engine inferx --tag new_candidate \
  --flash-attention --cuda-graphs --decode-linear --split-decode --packed-projections
MPLCONFIGDIR=/tmp/inferx-matplotlib python/.venv/bin/python benchmarks/qwen3/analyze.py \
  new_reference new_candidate --require-exact --output-dir /tmp/new_comparison
```

The last command intentionally exits nonzero while exact equivalence fails. It still
saves validated throughput, latency, memory, output agreement, and plots. The driver
records experimental flags, hashes, commands, outputs, timestamps, memory samples,
and source archives. It now detects an InferX binary replaced during a benchmark.
The final/confirmation/vector-cache measurements used an unchanged binary throughout
each full matrix. Historical experiment archives are retained, including rejected paths.

## Retained implementation changes

- One pinned asynchronous metadata upload replaces eight synchronous uploads per step.
  Stable tensor offsets remain valid for captured graphs; staging memory is retained
  until stream completion.
- Greedy argmax runs inside decode graphs. The result copy still synchronizes before
  the scheduler consumes token IDs.
- Residual addition and RMSNorm are fused while preserving intermediate BF16 rounding.
  Q/K normalization and RoPE are fused with the same rounding boundaries.
- Experimental vectorized small-batch projections use FP32 accumulation and BF16 output.
  Unsupported shapes and larger batches retain cuBLAS.
- Experimental packed QKV and gate/up weights reduce projection calls. Original weight
  views share the packed allocation, avoiding permanent duplicate weights.
- Experimental split-KV decode attention exposes more parallel work at small batches.
  Device-side partition metadata is rebuilt on replay and reused across layers.
- Vectorized prefill KV writes copy eight BF16 values per thread, with the original
  path retained for small or unaligned inputs. An isolated graph-timed microbenchmark
  demonstrates the kernel-level benefit; whole-model timings remain authoritative.

The additional decode-linear, packed-projection, and split-attention paths stay opt-in
because they can change floating-point reduction order and greedy outputs. The prior
experimental attention/graph gates are retained. No quantization, prefix reuse,
speculative decoding, output-length reduction, or workload filtering was introduced.

A cuBLASLt calibration experiment did not improve the aggregate workload consistently
and was removed from the implementation; its `graph_tuned_20260922` archive remains.
A 128-row prefill tile did not show a consistent win and remains an explicit experimental
option (`--prefill-tile 128`); the measured final candidate uses the original 64-row tile.

## Validation and numerical limits

- Model, model-runner, scheduler, scheduler-output, RMSNorm and operator CTest targets
  pass on CUDA. Six archive-integrity tests and seven numerical-comparison tests pass.
- Exact operator tests cover fused residual/norm, fused norm/RoPE (partial and full
  rotation), packed column ordering, SiLU rounding, and vectorized cache writes over
  shuffled pages with untouched slots. Projection and attention kernels are checked
  against independent double-precision calculations; the attention tolerance remains
  0.005. No tolerance was widened.
- A graph-enabled runner regression covers variable batches, request turnover, live
  positions, logit row selection and KV-page transitions.
- `inferx_eager_check_20260922` runs the complete matrix with one measured repetition.
  All **174/174** request sequences match the corresponding final graph-run outputs.
  It is a correctness diagnostic, not included in the three-repeat performance summary.
- The metadata/graph-sampling change and each normalization fusion preserved **522/522**
  paired sequences of its preceding stage. Reduction-changing experimental kernels do
  not have that token-preservation guarantee.
- Final candidate vs later vLLM: **326/522** paired sequences match exactly; first-repeat
  agreement is **108/174**. The strict comparison fails. vLLM itself varies across some
  repeats and between its two fresh runs; that does not excuse candidate differences.
- The fresh identical-prefix diagnostic examines one divergence prefix saved in three
  repetitions (`p128_o128_b1_c1`). Against Transformers BF16 SDPA it has maximum logit
  error **0.19140625**, RMS error **0.046742**, and a different argmax. The independent
  reference ties the two selected tokens; InferX favors its token by 0.0625. This
  explains that local decision only, not all mismatches. Zero tolerance was used as
  an exact-logit diagnostic, not adopted as a full-model acceptance threshold.

The installed vLLM native fused residual/RMSNorm computes statistics on the FP32
residual sum before storing its BF16 residual, while the retained InferX fusion
preserves InferX's explicit BF16-add semantics. Together with compiler fusion,
attention and GEMM reduction differences, this remains a concrete numerical
investigation area. Full batched identical-prefix vLLM logit comparisons are still
required before claiming numerical interchangeability.

## All final configurations

End-to-end output tokens/s; median of three repetitions. B is scheduled sequence
capacity, C is concurrent requests. Ratios use the later fresh vLLM run.

| Input | Output | B | C | vLLM | InferX | InferX/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 1 | 1 | 359.6 | 383.2 | 1.066× |
| 128 | 32 | 4 | 4 | 1230.8 | 1468.5 | 1.193× |
| 128 | 32 | 4 | 8 | 1230.7 | 1459.8 | 1.186× |
| 128 | 32 | 16 | 16 | 3964.0 | 4075.0 | 1.028× |
| 128 | 128 | 1 | 1 | 386.9 | 382.3 | 0.988× |
| 128 | 128 | 4 | 4 | 1390.0 | 1415.3 | 1.018× |
| 128 | 128 | 4 | 8 | 1393.4 | 1421.5 | 1.020× |
| 128 | 128 | 16 | 16 | 4135.8 | 4503.4 | 1.089× |
| 512 | 32 | 1 | 1 | 356.2 | 346.7 | 0.974× |
| 512 | 32 | 4 | 4 | 967.5 | 1040.3 | 1.075× |
| 512 | 32 | 4 | 8 | 967.5 | 1037.2 | 1.072× |
| 512 | 32 | 16 | 16 | 2002.5 | 2012.2 | 1.005× |
| 512 | 128 | 1 | 1 | 349.4 | 369.3 | 1.057× |
| 512 | 128 | 4 | 4 | 1119.6 | 1331.0 | 1.189× |
| 512 | 128 | 4 | 8 | 1195.1 | 1231.2 | 1.030× |
| 512 | 128 | 16 | 16 | 2860.0 | 2866.7 | 1.002× |
| 1024 | 32 | 1 | 1 | 297.7 | 313.6 | 1.053× |
| 1024 | 32 | 4 | 4 | 780.8 | 782.6 | 1.002× |
| 1024 | 32 | 4 | 8 | 778.9 | 782.1 | 1.004× |
| 1024 | 32 | 16 | 16 | 1305.9 | 1191.5 | 0.912× |
| 1024 | 128 | 1 | 1 | 331.5 | 349.7 | 1.055× |
| 1024 | 128 | 4 | 4 | 928.8 | 1124.7 | 1.211× |
| 1024 | 128 | 4 | 8 | 924.4 | 1042.8 | 1.128× |
| 1024 | 128 | 16 | 16 | 2081.7 | 2057.1 | 0.988× |

All latency families, decode/prefill rates, memory samples and trial ranges are in
[summary.csv](optimization_20260922/summary.csv), with [plots](optimization_20260922/comparison.png)
and [strict agreement results](optimization_20260922/agreement.json). Peak memory
above the observed idle baseline is 5159–5375 MiB for the final candidate versus
5325–5441 MiB for the later vLLM run. These are sampled whole-device peaks, including
startup and display allocations, not per-request allocator measurements.

[Test log](diagnostics/optimization_tests_20260922.log),
[identical-prefix report](diagnostics/fresh_prefixes_20260922/comparison.json),
[cache-write microbenchmark source](profiles/vector_cache_micro.cu), and
[cache-write timings](profiles/vector_cache_micro.log) are preserved. Reproduce the
isolated kernel experiment with `nvcc -O3 -arch=sm_89
benchmarks/qwen3/profiles/vector_cache_micro.cu -o /tmp/cache_micro`, then run
`LD_LIBRARY_PATH=/usr/lib/wsl/lib /tmp/cache_micro` with no other benchmark running.
