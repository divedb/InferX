# Numerical diagnosis before further performance claims

September 22 update: CUDA execution is available again and the operator fixtures
pass. Fresh throughput and a failing identical-prefix logit diagnostic are recorded
in [OPTIMIZATION_20260922.md](OPTIMIZATION_20260922.md). The GPU-access statements
below describe the earlier session, not the current environment.

The throughput target remains unmet. The historical optimized attention results
are 0.57–0.88× vLLM, with failing exact-token comparisons. No new GPU throughput
or full-model correctness result is claimed for the changes below.

## Concrete reference differences corrected

- Llama/Qwen RMSNorm casts normalized activations back to the input dtype before
  multiplying by the weight. InferX previously used FlashInfer's single-rounding
  RMSNorm throughout. The model now requests the intermediate rounding explicitly,
  including Q/K normalization and the final norm. The generic operator retains its
  original default and Gemma behavior. Both CPU and CUDA implement the option.
- BF16 RoPE now rounds cosine/sine and the individual products before addition or
  subtraction. Previously it performed the entire rotation in FP32. Explicit
  non-contracting intrinsics retain these boundaries. Inverse frequencies use the
  reference's reciprocal-of-positive-power expression.

These differences were established from the installed Transformers Qwen3 code,
vLLM's rotary cache construction, the local vLLM CUDA RMSNorm/rotation sources,
and the vendored FlashInfer normalization source. They are operation-level
discrepancies, **not an established explanation of every historical token mismatch**.
Other reference differences remain to be measured, including the intermediate
rounding in eager Transformers SiLU versus fused vLLM SiLU-and-multiply, attention
backend numerics, and batch-dependent GEMM/reduction behavior.

The RMSNorm regression uses an independent PyTorch BF16 fixture whose four values
all distinguish the old and new formulas. It checks exact values at widths 4, 128
and 1024, with two rows, both in place and out of place. A separate exact RoPE
fixture covers positions 0/17/1024, grouped-query heads, and unrotated columns.

## Identical-prefix replay

`numerical.py prepare` validates the existing archives and extracts each request's
prefix immediately before its first mismatch. Matching requests contribute their
final prediction. Every paired repeat is retained, including nondeterministic
vLLM repetitions. The full historical matrix produces **522 prefixes**.

```sh
cmake --build build-engine --target inferx inferx_ops_test inferx_rms_norm_test -j 8
python/.venv/bin/python benchmarks/qwen3/numerical.py prepare \
  --reference benchmarks/qwen3/results/vllm_baseline \
  --candidate benchmarks/qwen3/results/inferx_flash_attention \
  --output /tmp/qwen3-prefixes.json

# Set ATOL and RTOL to the agreed full-model logit limits. No defaults are inferred.
env LD_LIBRARY_PATH=/usr/lib/wsl/lib python/.venv/bin/python benchmarks/qwen3/numerical.py run \
  --fixture /tmp/qwen3-prefixes.json --output /tmp/qwen3-numerical-eager \
  --atol "${ATOL:?set the required logit atol}" --rtol "${RTOL:?set the required logit rtol}"
# Repeat in a fresh output directory with --flash-attention, and with --chunk-size 16
# to exercise chunked prefill and cached continuation over shuffled physical pages.
```

Runs preserve fixture/checkpoint/config/binary hashes, package versions, source,
raw full-vocabulary logits and a per-prefix comparison. Transformers BF16 SDPA is
the independent logit oracle. Both engines receive the same prompt chunks and
teacher-forced continuation tokens. InferX runs the real `Model::Forward` using
paged KV. Each comparison requires:

1. Every logit satisfies `abs(candidate - reference) <= atol + rtol * abs(reference)`.
2. The greedy argmax matches the independent reference exactly.
3. The greedy token also matches the saved vLLM token for that prefix.

The report includes maximum/RMS error, failing-element count, both argmax margins,
and the reference score gap for the candidate's selected token. Nonfinite,
truncated or malformed output fails. Numerical closeness does not excuse a changed
greedy token. The existing attention tolerance of 0.005 is unchanged and is **not**
adopted as a full-model logit tolerance.

This replay is an eager, single-sequence diagnostic. It does not reproduce the
original scheduler's batching or validate graph replay. Saved vLLM archives have
tokens only: current vLLM logits at identical prefixes still need to be collected
on a GPU. `--device cpu` explicitly selects a CPU Transformers oracle; InferX
still requires CUDA, and a CPU oracle cannot establish CUDA-reference parity.

## Validation in this session

- CUDA compilation succeeded for the operator/model tests, benchmark and replay tool.
- Six CPU RMSNorm tests passed, including the new exact reference fixture.
- Seven numerical-gate tests and six existing archive-integrity tests passed.
- Scheduler, scheduler-output and model-runner CTest targets passed.
- All 522 historical replay prefixes were extracted successfully.
- GPU execution remains unavailable: NVML reports “GPU access blocked by the
  operating system”; the CUDA regression fails during runtime initialization.
  CUDA fixtures, full-model tolerances, fresh throughput and graph replay remain
  unverified. Existing experimental attention/graph gates stay in place.

Once GPU execution is available, run the operator fixtures and identical-prefix
diagnosis first, resolve remaining errors without widening tolerances, then rerun
the complete matched benchmark matrix with fresh tags and `analyze.py --require-exact`.
Further performance tuning must use new profiles and measured comparisons; the
historical measurements cannot establish a win for the changed implementation.
