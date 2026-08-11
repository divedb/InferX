# DeepSeek-V2-Lite performance analysis and optimization plan

## 1. Executive summary

InferX serves DeepSeek-V2-Lite correctly, but the measured path is slower than
vLLM 0.26.0 in every tested serving regime. The gap is modest in steady-state
batch-1 decode (105.7 versus 161.4 token/s, 1.53x) and much larger in prefill
(7.7x at 150 prompt tokens and 14.5x at 513 tokens). At concurrency four,
vLLM produces 210.4 output token/s versus InferX's 58.5 token/s (3.60x).

The source explains this shape. The two optimizations originally named as the
largest DeepSeek gaps -- absorbed MLA decode and device-resident grouped MoE
dispatch -- are already implemented. The remaining bottlenecks are:

1. **Prefill MLA is a correctness path, not a production attention path.** It
   gathers the latent cache, reconstructs per-head K/V with `kv_b`, then uses a
   custom two-pass attention kernel with repeated score computation and a
   synchronization inside the context loop.
2. **The absorbed decode kernels are scalar CUDA kernels.** `MlaAbsorbQ`,
   `MlaLatentAttention`, and `MlaUnabsorbOut` avoid the asymptotically worse
   reconstruction, but do not use tensor cores or a tuned MLA backend.
3. **DeepSeek decode does not replay CUDA graphs.** The server prints that
   graphs are enabled, but the DeepSeek engine arm neither captures nor
   replays them; `DeepseekV2Model::CaptureDecodeGraph` is unimplemented.
4. **Sampling is synchronously CPU-bound at every token.** Each step calls
   `cudaStreamSynchronize`, copies a full 102,400-element BF16 vocabulary row
   to the host for every sequence, converts it to FP32, and performs argmax on
   the CPU. This prevents model/scheduler overlap and adds work proportional to
   `batch * vocab` even though greedy sampling needs only one integer per row.
5. **The BF16 MoE grouped kernel is launch-efficient but not compute-efficient.**
   It uses one warp per output column and scalar BF16-to-FP32 multiply-adds.
   It deliberately does not use tensor cores and its own comment identifies a
   CUTLASS grouped GEMM as the upgrade for larger prefill batches.
6. **Multi-sequence MLA is serialized on the host.** Every layer loops over
   sequence ranges and launches an independent MLA pipeline per sequence. That
   weakens batching precisely where vLLM gains most at concurrency four.

The recommended order is: establish component-level profiles; move argmax and
sampling to the GPU and make the step asynchronous; add graph capture for the
fixed-shape absorbed decode path; replace prefill attention with FlashInfer or
another fused ragged MLA implementation; then replace the custom grouped BF16
MoE kernel with a tensor-core grouped GEMM. The first two decode changes should
close much of the 1.53x steady-state decode gap. Fused prefill MLA and grouped
GEMM are required to address the 7.7-14.5x prefill gap.

## 2. Measurement basis

### 2.1 Environment and workload

- GPU: NVIDIA GeForce RTX 4090, 48 GB
- InferX commit: `6c56642d67e202ac7cd3d3ee44ed6da08dad674b`
- Baseline: vLLM 0.26.0, PyTorch 2.11.0+cu130, Transformers 5.14.1
- Checkpoint: `deepseek-ai/DeepSeek-V2-Lite-Chat`, BF16
- Maximum sequences: 4; maximum sequence length: 1024
- Decode length: 16 tokens; greedy sampling; unique prompts
- Prefill targets: 128 and 512 tokens (actual tokenizer counts 150 and 513)
- Benchmark client: the same `bench/serve_bench.py` socket and SSE path for
  both engines

Raw results are in
`bench-results/remote-deepseek-v2-lite-20260809/{inferx,vllm}-bf16.json`.

### 2.2 Results

| Scenario | InferX | vLLM | vLLM advantage |
|---|---:|---:|---:|
| Batch-1 TTFT, p50 | 331.0 ms | 100.0 ms | 3.31x lower |
| Batch-1 decode after TTFT | 105.7 tok/s | 161.4 tok/s | 1.53x |
| Batch-1 whole-request output rate | 33.3 tok/s | 82.4 tok/s | 2.47x |
| Concurrency 1 output throughput | 31.0 tok/s | 82.6 tok/s | 2.67x |
| Concurrency 2 output throughput | 36.6 tok/s | 57.6 tok/s | 1.57x |
| Concurrency 4 output throughput | 58.5 tok/s | 210.4 tok/s | 3.60x |
| 150-token prefill | 383.6 tok/s | 2,961.3 tok/s | 7.72x |
| 513-token prefill | 595.8 tok/s | 8,627.2 tok/s | 14.48x |

The vLLM concurrency-two point is anomalous: it is slower than both its
concurrency-one and concurrency-four points. Only four requests were sampled
at that setting. It should be rerun with at least 20 measured requests before
using that row for optimization decisions. The batch-one and concurrency-four
results, together with the code-level evidence below, are sufficient to locate
the major gaps.

### 2.3 What the result shape tells us

- InferX's inter-token latency is 10.1 ms at batch one versus vLLM's 6.2 ms.
  This is a meaningful but bounded decode-kernel/launch gap.
- InferX's concurrency-four p99 ITL rises to 39.5 ms while vLLM remains at
  9.5 ms. InferX is batching requests, but the work does not scale efficiently
  with batch size.
- InferX takes 391 ms for a 150-token prefill and 861 ms for 513 tokens. vLLM
  takes only 50.7 and 59.5 ms. The widening ratio with prompt length points to
  the attention and MoE implementations, not HTTP overhead.
- Short 16-token requests amplify TTFT. InferX's whole-request batch-one rate
  is 2.47x behind even though decode alone is only 1.53x behind. Prefill must
  therefore be fixed to improve interactive workloads, not merely bulk
  ingestion.

## 3. Current execution path

For each scheduler step, `DeepseekV2Model::Step` performs:

1. Four host-to-device copies for token IDs, positions, slot mappings, and the
   block table.
2. Embedding lookup.
3. For each of 27 layers, MLA followed by either the initial dense FFN or a
   routed MoE plus shared expert.
4. Final norm and a full vocabulary projection.
5. A stream-wide synchronization.
6. One synchronous device-to-host copy of the full BF16 vocabulary row per
   requested output token, BF16-to-FP32 conversion on the CPU, and CPU argmax.
7. Scheduler commit and text streaming.

Decode chooses the absorbed MLA branch when a sequence contributes exactly one
token. Prefill chooses the unabsorbed branch. This distinction is important:
the old description of every decode step reconstructing full K/V is no longer
true after commit `44eef64`; similarly, the old per-expert cuBLASLt loop and
per-layer offsets readback were removed by commit `1cd79a5`.

## 4. Bottleneck analysis

### 4.1 Prefill attention is the highest-confidence bottleneck

In `src/model/mla.cc`, every non-single-token sequence takes the unabsorbed
branch:

- gather every cached latent into a contiguous buffer;
- split latent and RoPE portions into more scratch buffers;
- multiply all context latents by `kv_b` to reconstruct per-head K and V;
- split that reconstruction again;
- execute `MlaAttention`.

`csrc/mla.cu` explicitly calls `MlaAttentionKernel` a correctness path. It
launches one block per query/head and:

- scans visible context once for the maximum score;
- scans it again for softmax and V accumulation;
- recomputes each score in the second pass;
- synchronizes the block for every context position;
- accumulates in scalar FP32 instead of using a tiled tensor-core design;
- creates substantial context-sized scratch traffic before attention begins.

This gives poor arithmetic intensity and grows much worse with prompt length.
The measurement agrees: when the prompt grows from 150 to 513 tokens, InferX
TTFT grows by 470 ms, while vLLM grows by only 8.8 ms. No transport, tokenizer,
or fixed scheduler overhead can explain that difference.

**Optimization:** integrate FlashInfer's MLA prefill wrapper if it supports the
checkpoint's decoupled RoPE and paged latent layout. If not, implement a fused
ragged prefill kernel that consumes the compressed latent cache directly or
fuses reconstruction with tiled attention. It must avoid materializing
`[context, heads, qk_nope+v]` and avoid the second score computation. Batch all
sequence ranges in one ragged launch rather than looping in C++.

**Acceptance gate:** at actual 512 prompt tokens, reach at least 4,000 token/s
first, then target 80% of matched vLLM. Confirm at 128, 512, 2048, and 4096
tokens; a single short-prompt result can hide an O(context-squared) design.

### 4.2 Absorbed decode removes reconstruction but remains untuned

The decode path correctly applies MLA absorption:

- `MlaAbsorbQ`: `W_UK^T * q_nope`;
- `MlaLatentAttention`: attention directly over the 512-dimensional latent
  plus the shared RoPE key;
- `MlaUnabsorbOut`: apply `W_UV` to the latent attention result.

That is why InferX is only 1.53x behind in batch-one steady-state decode rather
than orders of magnitude behind. However, all three operations are hand-written
scalar CUDA kernels. `AbsorbQKernel` and `UnabsorbOutKernel` implement matrix
products with scalar BF16 loads and FP32 accumulation rather than MMA/tensor
cores. `LatentAttentionKernel` is another correctness-style two-pass softmax
kernel, with one block per head and a full latent output accumulator.

**Optimization:** express absorption/unabsorption as batched tensor-core GEMMs
or fuse them into the MLA attention kernel. Compare three concrete variants:

1. cuBLASLt strided-batched GEMM for absorption and unabsorption;
2. CUTLASS/CuTe kernels specialized for the fixed V2-Lite dimensions;
3. FlashInfer's MLA decode kernel, retaining InferX's paged latent allocator.

The comparison must include batch 1, 2, 4, and 8 and contexts 32 through 4096.
A kernel that wins at batch 8 may lose at batch 1 through launch overhead.

**Acceptance gate:** reduce batch-one device decode time from roughly the
observed 9.5 ms/token to at most 7 ms before graph optimization, without an
argmax mismatch against the current kernel.

### 4.3 No CUDA graph replay on the DeepSeek path

The command line reports `cuda graphs: on`, but this is misleading for
DeepSeek. The DeepSeek branch in `Engine::Create` does not call graph capture,
`DeepseekV2Model::CaptureDecodeGraph` returns `Unimplemented`, and the sync
runner calls `Step` directly on every iteration.

A DeepSeek decode step launches hundreds of operations: norms, copies, query
and latent projections, RoPE/splits, three absorbed MLA kernels, output
projection, routing, three dispatch kernels, gather, two routed-expert kernels,
activation/combine, two shared-expert GEMMs, residual operations, and final
vocabulary work across 27 layers. Even small per-launch costs accumulate.

The comment claiming graph capture needs a fixed-shape MLA path is now stale:
the absorbed single-token path has token-sized scratch and fixed tensor shapes
for a fixed batch size and block-table width. What still blocks capture is the
per-sequence C++ loop and synchronous host sampling, not context-sized decode
scratch.

**Optimization:** first create one batched/ragged absorbed MLA call per layer.
Then capture graphs for each decode batch size from `max_running` down to one,
using the Qwen2 descending-shape allocation rule. Keep prefill eager. Make the
startup log report the number of successfully captured DeepSeek graphs rather
than the requested flag.

**Acceptance gate:** replay graphs for batch sizes 1-8, confirm identical token
output, and demonstrate at least a 15% batch-one ITL reduction. Use an Nsight
Systems trace to verify that launch gaps have collapsed rather than relying
only on end-to-end timing.

### 4.4 Full-vocabulary host sampling serializes every step

`DeepseekV2Model::Step` calls `cudaStreamSynchronize` after the model forward.
It then copies each selected `[vocab]` BF16 row to a host vector, expands all
elements to FP32, and `RunSyncModel` scans them for argmax. With V2-Lite's
102,400-token vocabulary this transfers about 200 KiB per sequence per decode
step, versus four bytes if only a sampled token ID crosses PCIe.

The cost is not just PCIe bandwidth. The stream synchronization prevents CPU
scheduler work from overlapping the GPU tail, repeated pageable host vectors
can introduce staging overhead, and batch size multiplies conversion/argmax
work. This contributes to the poor concurrency scaling.

**Optimization:** reuse the existing device sampler used by Qwen2. Add
`DeepseekV2Model::StepAsync` and `AwaitStep`, run greedy argmax on the GPU, and
copy only one `int32` per logits row through pinned memory. The eventual sampler
should support temperature/top-p without returning logits to the host. For an
additional memory/bandwidth reduction, compute logits only for
`batch.logits_indices` rather than projecting every prefill token if that is
not already guaranteed by the batch layout.

**Acceptance gate:** an Nsight trace must show no full-vocabulary D2H transfer
and no host-wide stream synchronization in the steady-state loop. Measure the
change independently before graph capture so their benefits are attributable.

### 4.5 The custom BF16 grouped MoE kernel leaves tensor cores idle

Replacing 64 per-expert launches with two grouped launches was the right first
step. The current `GroupedGemmBf16Kernel`, however, assigns one warp to an
output column and performs scalar BF16-to-FP32 FMAs. It does not issue MMA
instructions. For decode it binary-searches the expert offsets for every
grouped row/output tile; for prefill it loops over every expert inside a grid
whose y dimension is based on total assignments. Empty and mismatched expert
iterations add control overhead.

DeepSeek-V2-Lite routes each token to six of 64 experts and also runs two shared
experts. Thus every MoE layer combines small irregular routed GEMMs with a
substantial dense shared-expert MLP. vLLM has highly tuned fused MoE kernels,
while InferX's kernel was designed to remove synchronization and launch storms,
not to reach peak BF16 throughput.

**Optimization:** use CUTLASS/CuTe grouped GEMM or a proven fused-MoE backend.
Build the problem descriptors and expert tile schedule on the GPU, use tensor
cores, fuse SiLU multiplication where practical, and consider fusing weighted
combine into the down projection epilogue. Independently tune the shared
expert's cuBLASLt algorithms because it runs for every token and can dominate
when routed expert batches are sparse.

**Acceptance gate:** add a DeepSeek-specific MoE benchmark with real shapes
and routing distributions for 1, 2, 4, 8, 32, 128, and 512 tokens. Report both
kernel time and achieved memory bandwidth/TFLOP/s. Require at least 2x speedup
over the current kernel for 128+ token prefill without regressing batch-one
decode by more than 3%.

### 4.6 Per-sequence MLA launches defeat continuous batching

`RunPagedForward` iterates through `ranges` inside every layer and invokes the
entire MLA pipeline separately for each sequence. With four decoding
sequences, the model therefore launches four sets of query/KV/attention/output
operations per layer instead of one batched ragged operation. The MoE and dense
parts see the combined token batch, but attention does not.

This matches the throughput curve. InferX gains only 1.89x from concurrency one
to four (31.0 to 58.5 token/s), while vLLM gains 2.55x (82.6 to 210.4 token/s).
InferX p99 ITL simultaneously rises from 10.3 to 39.5 ms.

**Optimization:** change the MLA interface to accept sequence lengths,
query-start offsets, and a 2-D block table for the whole batch. One launch must
process all active sequences. This is also a prerequisite for a small set of
batch-size CUDA graphs.

**Acceptance gate:** concurrency-four throughput should exceed 3x the
concurrency-one throughput on the current short-context workload, with p99 ITL
below 20 ms, before attempting scheduler policy tuning.

### 4.7 Excess copies and unfused elementwise work

Each layer copies the hidden state into `resid`, later copies the attention or
FFN result back into `x`, and launches separate norm, split, residual-add,
activation, gather, and combine kernels. These operations are individually
small but become material over 27 layers, especially without graphs.

**Optimization:** after the larger items above:

- fuse residual-add plus RMSNorm, following the Qwen/vLLM pattern;
- write attention and FFN residual outputs directly into alternating hidden
  buffers instead of D2D copying;
- fuse RoPE split and latent-cache append;
- fuse MoE gather into the grouped GEMM input iterator;
- fuse weighted combine and shared-expert addition;
- use persistent pinned buffers for the small scheduler inputs, or update graph
  input nodes without pageable staging.

These changes should be driven by a trace. They are not a substitute for fused
attention, GPU sampling, or tensor-core MoE.

### 4.8 Scheduler and HTTP are secondary for this result

The engine intentionally waits 1 ms to coalesce a newly arrived burst and may
wait up to 2 ms while idle. These values affect TTFT but cannot explain 340-800
ms of additional prefill time. Likewise, both engines used the same HTTP/SSE
benchmark client. Scheduler tuning should wait until GPU execution is fixed;
otherwise it risks trading latency for throughput around a slow kernel path.

After the GPU work, evaluate adaptive coalescing: skip the 1 ms window for an
isolated latency-sensitive request, retain it briefly when a burst is visible,
and expose the delay as a metric.

## 5. Loading and deployment observations

InferX took 53.608 seconds from process start to the HTTP endpoint becoming
available. This measurement was made with an already-used filesystem cache, so
it is a warm-cache startup number, not a disk-cold load. vLLM separately logged
6.77 seconds for weight reads, 9.20 seconds for model construction, and 36.01
seconds for engine initialization including compilation and warmup.

InferX's newer loader contains per-layer and total-stage timing, but the server
log captured during this run did not contain those messages. Before optimizing
startup, rebuild and verify that loader instrumentation is in the deployed
binary, then split startup into checkpoint mmap/read, host stacking, H2D copy,
scratch/KV allocation, and warmup. Weight stacking for 64 experts per MoE layer
is a plausible startup cost, but it must not be blamed without these counters.

Two deployment defects also distorted the first validation attempt:

- the binary resolved Miniconda's old `libstdc++.so.6` and required an
  `LD_PRELOAD` workaround;
- the model registry lists `deepseek-v2-lite-chat`, but completion lookup only
  succeeds with `deepseek-v2-lite-chat@loaded` because the unversioned ID is not
  registered as an alias.

These are correctness/operability issues rather than throughput bottlenecks,
but they should be fixed before automated performance regression runs.

## 6. Prioritized implementation roadmap

### P0: make performance attributable

1. Add CUDA event ranges for embedding, MLA Q/KV projections, absorbed or
   unabsorbed attention, routed MoE, shared expert, final projection, sampling,
   and H2D/D2H work.
2. Add NVTX ranges per layer and phase.
3. Run Nsight Systems for one prefill plus 32 decode tokens at batch 1 and 4.
4. Run Nsight Compute on the MLA and grouped MoE kernels, recording achieved
   occupancy, DRAM throughput, tensor-core utilization, warp stalls, and launch
   count.
5. Expand the serving benchmark to at least 20 requests per concurrency point,
   128 output tokens, and three independent runs. Report median and range.

### P1: remove the synchronous sampling boundary

1. GPU argmax and compact token output.
2. Pinned asynchronous token-ID copy.
3. `StepAsync`/`AwaitStep` integration with the server loop.
4. Project only requested logits rows.

Expected benefit: lower batch-size sensitivity and a measurable reduction in
both ITL and CPU time. This is relatively contained and enables later graphs.

### P2: batch MLA decode and capture graphs

1. Replace the per-sequence MLA loop with a ragged batch interface.
2. Preallocate all decode scratch for `max_running`.
3. Capture one graph per decode batch size, descending from maximum to one.
4. Expose actual capture count and replay hit rate.

Expected benefit: close much of the remaining batch-one 1.53x decode gap and
materially improve concurrency-four scaling.

### P3: replace prefill MLA

1. Integrate and validate FlashInfer MLA prefill if its layout matches.
2. Otherwise implement tiled fused ragged MLA without full K/V materialization.
3. Add chunked prefill so long prompts do not monopolize the scheduler.

Expected benefit: largest TTFT gain; target at least 7x at the measured 512
tokens to approach the current vLLM result.

### P4: tensor-core fused MoE

1. CUTLASS/CuTe grouped BF16 GEMM at real DeepSeek shapes.
2. GPU tile scheduling from expert counts.
3. Activation and combine epilogue fusion.
4. Tune the dense shared expert separately.

Expected benefit: improve both prefill and high-concurrency decode after
attention and launch overhead stop masking the MoE cost.

### P5: fuse memory-bound residual work and tune policy

Only after P1-P4, fuse residual/norm/copy kernels, reduce scheduler input
copies, tune batch coalescing, and consider BF16/FP8 weight or KV variants.
Quantization should be evaluated as a separate accuracy/performance product;
it should not obscure the BF16 comparison used here.

## 7. Required regression matrix

Every optimization should preserve the long-prompt HF logits gate with the
deinterleaved RoPE convention and run this matrix on the same locked host:

| Dimension | Values |
|---|---|
| Prompt length | 32, 128, 512, 2048, 4096 |
| Decode length | 16 for iteration, 128 for final reporting |
| Concurrency | 1, 2, 4, 8 |
| Prefix mode | unique prompts; shared prefix separately |
| Metrics | TTFT p50/p99, ITL p50/p99, output token/s, prefill token/s, GPU time, launch count, GPU utilization, CPU utilization |
| Repetitions | 3 runs, at least 20 measured requests per point |

Compare identical text and server-reported token counts. Run one engine at a
time, retain raw JSON and profiler reports, record clocks and software
versions, and reject a change that improves a mean while materially worsening
p99 latency without documenting the trade.

## 8. Target outcome

The next performance milestone should be explicit:

- batch-one decode at least 145 token/s;
- concurrency-four output throughput at least 170 token/s;
- 512-token prefill at least 4,000 token/s as an intermediate gate and within
  20% of matched vLLM as the final gate;
- batch-one TTFT below 150 ms for the current short prompt;
- no full-vocabulary D2H copies in steady state;
- successful CUDA graph replay for absorbed decode batch sizes 1-8;
- identical greedy continuations and an unchanged DeepSeek reference-logits
  result.

These targets focus effort where the evidence is strongest. Micro-optimizing
HTTP, tokenization, or scheduler bookkeeping before replacing the correctness
prefill kernel and synchronous sampling boundary will not close the measured
gap.
