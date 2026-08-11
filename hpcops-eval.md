# H20 evaluation: InferX kernels vs. hpc-ops

Status: plan, ready to execute once the H20 box is reachable. Box setup follows
`deploy.md`; this file adds only what differs on Hopper and for hpc-ops.

Goal: measure InferX's current CUDA implementations on an H20 (sm90), compare
per-op against `third_party/hpc-ops` (pinned at `heads/main`, Hopper-only,
tuned for exactly this GPU), and shortlist which ops are worth replacing —
for speed or for maintainability. This is an evaluation, not an integration:
nothing links against hpc-ops yet.

## 1. What we run today vs. what hpc-ops offers

InferX device code lives in `csrc/` (~6.9k lines). Mapping against the hpc-ops
operator catalog:

| InferX op | Where | hpc-ops counterpart | Verdict |
|---|---|---|---|
| Paged prefill/decode attention, bf16 (FlashInfer headers) | `flashinfer_attention.cu`, `flashinfer_prefill.cu` | Prefill/decode paged attention BF16/FP8 + dynamic decode scheduling | **Benchmark — top candidate.** Their headline claim is up to 2.22x decode vs FlashInfer on H20, and the dynamic split-k scheduler targets exactly our mixed-length decode batches. Applies to the GQA models (qwen2, gpt-oss), not DeepSeek-V2. |
| MLA attention | `mla.cu` | none | **Keep.** hpc-ops has no multi-head latent attention; DeepSeek-V2 stays on our kernels + FlashInfer regardless. |
| Decode-step sampling | `ArgmaxSample`/`SampleTokens` in `layers.cu` | Fused sampler (claims up to 8.5x vs FlashInfer at vocab 120k) | **Benchmark.** Cheap to compare, and sampling sits on the decode critical path. |
| MoE router + dispatch, bf16 | `moe.cu` | BF16xFP32 router GEMM; fused FP8 MoE pipeline | **Benchmark the router GEMM; note the FP8 MoE.** Their fused MoE is FP8-centric — it becomes relevant only if we adopt an FP8 MoE path (we serve V2-Lite MoE in bf16 today). |
| Dense GEMM (cuBLASLt) | `gemm_cublaslt.cc` | BF16xFP32 GEMM (router/state-compress shapes only) | **Low priority.** Narrow shape coverage; cuBLASLt remains the general path. |
| W4A16 / MXFP4 quantized GEMM | `w4a16_gemm.cu`, `mxfp4_gemm.cu`, `quantize.cu` | roadmap only (4/8-bit mixed precision) | **Keep.** Nothing to compare yet. |
| RMSNorm, RoPE(+KV append, +fp8 KV), SiLU-mul, misc | `layers.cu` | RMSNorm, RoPE + KV store, activation quantization | **Benchmark, maintainability lens.** Ours are simple and correct; replace only if the fused RoPE+store variant wins clearly at our shapes. |
| NCCL allreduce, then separate RMSNorm (TP=2) | `src/comm/` + `layers.cu` | Fused AllReduce + Residual + RMSNorm (claims up to 1.76x vs NCCL) | **Benchmark only if the rental has ≥2 GPUs** — single-GPU H20 cannot exercise this. Flagged as an open question below. |

Bear in mind while reading results: their baselines were benchmarked on H20,
where FlashInfer's sm90 paths are not at their best; wins may not transfer to
other architectures, which is fine — any adoption stays gated to sm90.

## 2. Why extraction is feasible (if we adopt anything)

hpc-ops ships as a Python `.abi3.so` module for vLLM/SGLang; InferX cannot link
it as-is. But torch only appears in the `entry.cc` binding layer (plus the
allreduce header); the kernel implementations underneath
(`src/attention/decode/sm90/`, `src/sampler/`, …) take raw pointers. Adoption
would mean compiling the needed `.cu` files into `csrc/` with our own thin
entry points, built only when `90a` is in `INFERX_CUDA_ARCHS`, dispatched at
runtime on compute capability 9.0, with the current kernels as the fallback on
everything else. MIT license, so vendoring kernels is clean.

## 3. Box setup deltas vs. deploy.md

- `INFERX_CUDA_ARCHS=90a` (the Hopper-specific target; hpc-ops requires it and
  FlashInfer's wgmma paths want it). Everything else in the deploy.md checklist
  (CUDA 13.0 toolkit, GCC 13, CMake 3.28+, rustup, folly deps, data-disk
  layout) applies unchanged.
- hpc-ops needs its own venv: `torch==2.7.0` (cu128 wheel), `numpy`, `pytest`,
  then `pip install -e third_party/hpc-ops` (its setup.py drives CMake; needs
  only CUDA 12.8+, satisfied by 13.0). This venv is for running *their*
  benchmarks — InferX's build does not touch it.
- Checkpoints: reuse the DSV2-Lite checkpoint for serve-level numbers, and a
  qwen2 GQA checkpoint for the attention comparison (MLA models cannot exercise
  hpc-ops attention).

## 4. Procedure

Phase 1 — InferX baseline on H20 (also useful independently of hpc-ops:
first sm90 datapoint, plus the pending serve smoke test + vLLM baseline):

```bash
(cd build && ctest -L kernel)            # correctness first: 28 kernel tests
./build/bench/attention_bench            # prefill + paged decode sweeps
./build/bench/decode_step_bench          # end-to-end decode step
./build/bench/decode_breakdown_bench     # per-op share of the decode step  <-- ranks what matters
./build/bench/prefill_bench
./build/bench/gemm_bench
python bench/serve_bench.py              # serve-level TTFT/TBT
```

Save all output under `bench-results/h20-baseline/`.

Phase 2 — hpc-ops microbenchmarks at InferX's shapes. Their suites live in
`third_party/hpc-ops/benchmark/{attention_decode,sampler,rope_norm_store_kv,route_gemm,fuse_allreduce_rmsorm,fused_moe}`.
Run each at shapes matching our serving config (qwen2 head/dim config, our
page size, vocab sizes of our models, hidden sizes for norm/rope; batch 1-64,
KV lengths 512-32k) and, where the harness supports it, at their defaults too
(to reproduce the README claims and sanity-check the setup).

Phase 3 — comparison and shortlist. For each op: speedup at our shapes,
weighted by that op's share of the decode step from `decode_breakdown_bench`.
Adoption bar (both must hold):

1. ≥20% op-level win at our shapes, and
2. either ≥3% projected end-to-end decode-step win, or a genuine
   maintainability trade (deleting our code for theirs, not adding a second
   copy — realistically only plausible for the sampler and rope+store, not
   for attention where MLA forces us to keep our paths anyway).

Phase 4 — for whatever survives: extract that op's kernels per §2 into a
branch, wire the runtime sm90 gate, validate numerics against the current
implementation (bit-comparison where deterministic, tolerance elsewhere), and
re-run phase 1 to confirm the projected win is real.

## 5. Open questions

- How many GPUs does the rental have? Single H20 → the fused
  allreduce+RMSNorm row is untestable and TP=2 numbers wait for a 2-GPU
  rental.
- hpc-ops pin: `heads/main` today. Before any extraction, pin to a tag/commit
  in `.gitmodules` the way every other submodule is pinned.
- FP8: their attention and MoE wins are largest in FP8. We have fp8 weights
  and fp8 KV options but a bf16 MoE; whether to compare their FP8 MoE against
  a hypothetical InferX FP8 MoE path is a scope decision, not a benchmark
  detail — default is to record their numbers and defer.
