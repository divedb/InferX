# InferX — Engineering Plan for a Production-Grade LLM Inference System

Status: **Draft 1 (research + architecture plan)** — no production code yet.
Author: engineering (research phase)
Date: 2026-09-18
Working directory: `/home/gc/InferX`

This document is the output of the research phase. It is intended to be sufficient for a
strong engineer (or coding agent) to begin implementation **without repeating the research**.
It deliberately does **not** prescribe a repository file tree; it specifies components,
interfaces, invariants, lifecycle, and milestones. No production code is written in this phase.

---

## 0. Executive summary

We will build **InferX**, a from-scratch, production-grade LLM inference engine and OpenAI-compatible
serving layer. It is **not** a wrapper around vLLM/SGLang/TensorRT-LLM. We will reuse mature
*kernels and libraries* (Eigen/cuBLAS, FlashAttention/FlashInfer, NCCL, safetensors,
tokenizers) but own the
**engine architecture**: request lifecycle, scheduler, KV-cache manager, memory model,
execution loop, distributed orchestration, serving layer, observability, and reliability.

The central architectural bet, validated by the research, is:

> **A single-token-budget scheduler over a paged, content-addressed KV cache, with a tight
> single-writer engine loop separated from the HTTP/tokenization layer by an async IPC
> boundary, and a pluggable attention backend with a correctness-first reference path.**

This is the convergent design of vLLM V1, SGLang, LightLLM, and lmdeploy. We adopt its
fundamentals, document the tradeoffs, and defer features that are not required to be correct
and fast on day one.

The plan is staged so that **every milestone produces a testable, end-to-end artifact**:
the first milestone is a correctness-verified single-request generator; by the end of the
third it is a streaming, continuously-batched server with prefix caching; performance and
distributed features are layered on top and each is gated on measurement.

**Implementation language (revised 2026-09-18):** All InferX code is **C++23**, built with
CMake, formatted to the **Google C++ Style Guide**, and implemented by **prioritizing
well-established third-party libraries** over re-implementing solved problems. We use
libraries for tensor math/BLAS, JSON, safetensors, tokenization, distributed collectives,
and testing; we own the engine architecture (scheduler, KV cache, execution loop, serving
layer) that differentiates this project. See D1.

**Target models (revised 2026-09-18):** support for **Qwen3.8-27B** (dense),
**Qwen3.8-Flash-Next** (hybrid linear attention + MoE + MTP), and **DeepSeek-V4.1-Flash**
(MLA + MoE + MTP, FP8-native) is promoted **ahead of M1**. Their cache and attention
requirements — recurrent state, MLA latent KV, sparse MoE — define the backend ops and
cache-manager interfaces before serving/batching is built. See §1.4, milestone M0.5, and
decisions D19–D24.

### 0.1 Non-negotiable design principles

1. **Correctness first.** A numerically correct reference path (naive Eigen attention,
   explicit KV copy) must always exist and be selectable. Optimizations are opt-in and are
   validated against the reference.
2. **Measurement before optimization.** No optimization is merged without a benchmark
   showing improvement on the target hardware, and a correctness test showing no regression.
3. **Bounded resources.** Every queue, buffer, cache, and in-flight operation has an explicit
   bound and an explicit overload policy. Unbounded waiting queues are a defect.
4. **Fail-safe cache semantics.** Any cache/offload failure degrades to a *miss* or a
   *dropped save*, never to wrong output and never to a stalled engine.
5. **One writer per piece of state.** Scheduler state is owned by the engine loop; per-request
   output state is owned by the frontend. Cross the boundary only with immutable messages.
6. **Extensibility is an interface requirement.** Model architectures, attention backends,
   quantization schemes, schedulers, and KV tiers are behind interfaces from the start, even
   when only one implementation exists.
7. **The dev box is not production.** Hardware-specific claims must be tagged as
   [verified locally] / [requires target GPU] / [assumption to validate].

### 0.2 Target environment (current dev machine) [verified locally]

- GPU: 1× NVIDIA GeForce RTX 4080 (Ada Lovelace, `sm_89`), 16 GB GDDR6X, ~716 GB/s HBM bandwidth, no NVLink.
- CPU: 32 logical cores. Host RAM: 31 GB (20 GB free), 8 GB swap.
- CUDA: toolkit 13.1, driver 591.86.
- Toolchain [verified locally]: GCC 13.3 (`-std=c++23` works), Clang 18.1, CMake 3.28,
  Ninja 1.11, CUDA 13.1 + cuBLAS 13, Rust 1.97/Cargo (for `tokenizers-cpp`), Go 1.27.
- Python 3.12 with NumPy; PyTorch/Transformers not installed but installable (CPU wheel) for
  the reference oracle.
- Same machine hosts reference source trees used for this research: vLLM (`b8b302cd`),
  mini-sglang, LightLLM, lmdeploy, FlashAttention, FlashMLA, LMCache, Mooncake, FlexKV,
  DeepSpeed, DeepEP, DeepGEMM, `divedb/kv-cache`, kvcached, kvpress, aibrix.

**Practical consequence:** 16 GB limits realistic locally-runnable models to roughly
0.5B–8B parameters. Multi-GPU/TP/EP and H100-specific kernels (FA3, FP8 W8A8, DeepGEMM,
FlashMLA) **cannot be validated on this box** and must be developed against a separate target
rig (e.g., 2–8× H100/A100 or cloud). The plan marks these explicitly.

---

## 1. Goals and non-goals

### 1.1 Goals

- **Correctness**: token-for-token parity with the reference HuggingFace implementation for
  greedy decoding on supported models, within defined numerical tolerances for sampling.
- **Throughput**: continuous batching with high GPU utilization; measured tokens/s within a
  defined ratio of a vLLM baseline on identical hardware and workloads.
- **Latency**: predictable TTFT and TPOT/ITL under concurrency, with SLO-oriented scheduling
  and measured tail percentiles (p50/p90/p99).
- **Memory**: paged KV cache with near-zero fragmentation, block-level prefix sharing,
  explicit bounds, and a documented OOM policy.
- **Production serving**: OpenAI-compatible HTTP API, SSE token streaming, cancellation,
  timeouts, admission control/backpressure, health/readiness, graceful shutdown, structured
  logging, Prometheus metrics, and OpenTelemetry tracing.
- **Distributed**: tensor parallelism and (later) pipeline/expert parallelism behind clean
  interfaces, with deterministic correctness tests.
- **Extensibility**: add new dense/MoE/hybrid/MLA architectures, attention backends, quant
  schemes, and KV tiers without changing the scheduler.
- **Target-model support**: correctly execute the three priority models in §1.4 (dense
  Qwen3.8-27B, hybrid/MoE Qwen3.8-Flash-Next, MLA/MoE DeepSeek-V4.1-Flash) before serving
  and batching are frozen.
- **Reproducibility**: a checked-in benchmark harness and result schema so every optimization
  is traceable to numbers.

### 1.2 Non-goals (for v1)

- Training / fine-tuning. Inference only.
- Supporting every architecture. The three priority targets in §1.4 (dense,
  hybrid-linear/MoE, and MLA/MoE families) are in scope before M1; other families and
  multimodal inputs remain later.
- Full multi-node at launch, but interfaces must not preclude it.
- Beating vLLM on raw kernel throughput in the first milestones. Correctness plus
  architecture first; kernel parity later.
- Custom CUDA/CUTLASS kernels in M0–M3. We use Triton and third-party kernels first; write
  our own only where measurement justifies it.
- CPU-only inference as a supported production path (a PyTorch CPU fallback exists only for
  testing kernels/CI).
- A hosted multi-tenant control plane (rate limiting, routing, autoscaling) — that is a
  separate cluster concern (cf. aibrix); we expose the metrics such a plane needs.

### 1.3 Success criteria (falsifiable)

- `M0`: greedy output equals HF `generate()` for ≥3 models on ≥50 prompts and all
  prompt/output lengths tested (exact token id match).
- `M0.5`: the three target architectures in §1.4 pass tiny-checkpoint greedy parity, and
  their hybrid-state/MoE/MLA modules match independent references; cache-state invariants
  hold across chunk boundaries and preemption.
- `M1`: a streaming OpenAI `/v1/chat/completions` server sustains continuous batching with
  ≥2 in-flight requests and a valid SSE stream; no request starvation; clean abort.
- `M2`: prefix-cache hit rate ≥ 95% on a repeated-prefix workload; TTFT on a cache hit
  ≤ 50% of the miss TTFT for a 4k-token prefix (target, to be measured).
- `M3`: decode with CUDA graphs shows ≥10% lower inter-token latency than eager decode at
  batch ≥ 8 for a 3B model (target, to be measured).
- `M4`: sustained overload returns 429/503 within a bounded queue wait instead of unbounded
  latency growth; engine survives 10k-request stress without memory growth (RSS/VRAM).
- `M5`: spec decode yields ≥1.5× tokens/s at equal output distribution on a suitable workload
  (validated by distribution equivalence tests), or it is not enabled by default.
- `M6`: 2-GPU TP produces token-identical greedy output to 1-GPU for a sharded model
  [requires target GPU], and NCCL correctness tests pass.
- Overall: benchmark harness reproduces a documented baseline and reports TTFT/TPOT/E2E/
  throughput/goodput plus GPU metrics.

### 1.4 Target models and revised priority (2026-09-18)

**Priority change:** support for the three models below is promoted **ahead of M1
(serving/batching)**. Their attention, cache, and parallelism requirements are structural —
a hybrid linear-attention state cache or an MLA latent cache cannot be bolted onto a
paged-KV/scheduler design after the fact without rework. They must therefore shape the
backend ops, KV manager, and model-assembly interfaces first. Serving, continuous batching,
and streaming then build on those interfaces rather than preceding them.

The three targets (architecture details are **assumptions to validate**, see E10):

| Priority | Model | Assumed family | Features that drive engine work |
|---|---|---|---|
| P0 | **Qwen3.8-27B** | Dense decoder-only (Qwen3 lineage) | GQA, QK-norm, SwiGLU, RoPE (YaRN/long-context), tied/untied LM head; 27B needs multi-GPU and/or weight-only quant |
| P0 | **Qwen3.8-Flash-Next** | Hybrid linear attention + gated attention, sparse MoE, MTP | Chunked gated-delta-rule linear attention with recurrent **state cache**, hybrid KV+state management, MoE routing/expert parallelism, multi-token prediction |
| P0 | **DeepSeek-V4.1-Flash** | MLA + fine-grained MoE + MTP, FP8-native | Compressed MLA KV latent, (sparse) MLA attention + indexer, DeepSeekMoE all-to-all, FP8/FP4 weights and KV, MTP, very long context |

Design commitments that follow from this priority:

1. **Model assembly is config-driven and composable.** A layer is a composition of modules
   (attention variant, MLP/MoE variant, norm, positional encoding, optional MTP head). New
   architectures are registered as compositions, not forks of the model loop.
2. **The KV manager is hybrid from day one.** It must hold, per layer group: paged full
   attention KV, sliding-window KV, and opaque **recurrent state** (linear-attention /
   convolution / SSM), with per-group block accounting and prefix-cache semantics.
3. **The backend ops interface gains `MoE`, `LinearAttention`/`StateUpdate`, and
   `MlaAttention` primitives**, each with a CPU reference implementation before any GPU
   kernel. MLA and MoE do not get retrofitted into the dense attention op.
4. **Quantization is a load-time concern, not a kernel afterthought.** The weight loader and
   linear/MoE ops must accept FP8/BF16 (and, on target hardware, FP4) checkpoint formats
   with explicit per-tensor/per-block scales.
5. **MTP is a proposer behind the speculative-decoding interface**, reusing the scheduler
   lookahead/rollback machinery from D10 rather than a separate decode path.
6. **Correctness is validated on tiny random checkpoints locally.** Full-scale execution of
   27B and MoE models requires the target rig (§0.2), so parity is established at the
   architecture level first and at scale second.

---

## 2. Research findings

This section records what we studied and what we learned, so later decisions can be traced
to evidence. Sources are listed in §18.

### 2.1 Systems studied (source-level, not just docs) [verified locally]

| System | Commit / location | Why studied |
|---|---|---|
| vLLM V1 | `/home/gc/vllm` `b8b302cd` | Reference for unified scheduler, paged KV, prefix cache, executor, metrics |
| SGLang (prod) | installed `sglang/srt/mem_cache` | RadixAttention prefix cache, eviction policies, HiCache tiers |
| mini-sglang | `/home/gc/mini-sglang` | Compact end-to-end reference: process topology, overlap scheduling, radix cache, CUDA graphs |
| LightLLM | `/home/gc/LightLLM` `eb2f89bc` | Explicit overload/429, token-load admission, TP+SP overlap |
| lmdeploy | `/home/gc/lmdeploy` `309d2b50` | DP-TP hybrid, request-handle backpressure, strong KV quant (INT4/TurboQuant), health monitor |
| TensorRT-LLM | not vendored; referenced via kernels/backends | TRT-LLM attention path used through FlashInfer in mini-sglang |
| FlashAttention | `/home/gc/flash-attention` `c75d019` | FA2/FA3 algorithms, online softmax, warp specialization |
| FlashMLA | `/home/gc/FlashMLA` `9241ae3` | MLA decode, seesaw scheduling, FP8 sparse |
| DeepEP / DeepGEMM | `/home/gc/DeepEP`, `/home/gc/DeepGEMM` | MoE all-to-all, grouped/masked GEMM, Mega MoE |
| LMCache | `/home/gc/LMCache` `d495223b` | Tiered KV offload, layerwise pipelining, MP daemon |
| Mooncake | `/home/gc/Mooncake` `f4f7fd4a` | Disaggregated KV, transfer engine, early rejection under overload |
| FlexKV | `/home/gc/FlexKV` | Three-tier radix KV store, DAG transfer scheduling |
| divedb `kv-cache` | `/home/gc/kv-cache` | Rigorous fail-safe host-offload connector, commit-after-completion |
| kvcached | `/home/gc/kvcached` | CUDA VMM elastic KV (and its risk profile) |
| kvpress | `/home/gc/kvpress` | KV compression/pruning (accuracy-sensitive) |
| aibrix | `/home/gc/aibrix` | Gateway routing, RPM/TPM 429, autoscaling, LoRA-as-CRD |

Primary papers/blogs read: PagedAttention/SOSP'23 [1], Orca [2], Sarathi-Serve chunked
prefill [3], SGLang/RadixAttention [4], DistServe [5], Mooncake [6], vLLM V1 blog [7],
FlashAttention-2 [8]/FA3 [9], LMCache [10], Splitwise [11]. Full list in §18.

### 2.2 Cross-cutting findings

**(a) The convergent engine architecture.** vLLM V1, SGLang, LightLLM, and lmdeploy all
converge on the same skeleton:

```
HTTP/API frontend ──(async IPC)──> engine loop (schedule → execute → update)
                                        ├─ scheduler (token budget, continuous batching)
                                        ├─ KV-cache manager (paged pool + prefix cache)
                                        └─ workers (model forward, attention, sampler)
```

The frontend and tokenizer are separate processes/threads from the engine loop specifically
to keep the per-step CPU overhead near zero, because on fast GPUs the model forward can be
~5 ms while Python/HTTP/detokenization overhead is comparable [7].

**(b) Unify prefill and decode into one token-budget scheduler.** vLLM V1 explicitly removed
the prefill/decode phase distinction and schedules a dict `{request_id: num_new_tokens}`
against a global `max_num_batched_tokens` budget. This single representation subsumes
continuous batching, chunked prefill, prefix caching, and speculative decoding
[7][vLLM `scheduler.py:180-189,198`]. Chunked prefill was shown to remove prefill/decode
stalls and improve throughput-latency Pareto [3].

**(c) Paged, content-addressed KV cache is the foundation.** PagedAttention [1] eliminates
fragmentation by allocating fixed-size blocks through a block table; vLLM V1 additionally
hashes each full block together with its parent's hash so a cache hit on block *k* proves the
entire prefix is present, enabling early-break lookups and zero-copy sharing
[vLLM `kv_cache_utils.py:547-573`, `block_pool.py`]. SGLang's RadixAttention instead keeps a
radix tree and splits nodes to reuse arbitrary page-aligned prefixes, at higher metadata cost
[SGLang `radix_cache.py`]. Both separate the *physical* pool from the *logical* block/page
table consumed by attention.

**(d) Attention is the hot path and must be pluggable.** Prefill is compute-bound and wants
FlashAttention-style tiling over a full/contiguous sequence; decode is memory-bound and wants
paged FlashDecoding/Triton split-K over hundreds of sequences, each with its own block table.
FA3's warp specialization and TMA matter on Hopper; on Ada (`sm_89`) FA2-class kernels are the
practical ceiling. vLLM selects a backend by model, dtype, head size, and GPU arch
[vLLM `selector.py`, `platforms/cuda.py`]. FlashMLA is required for DeepSeek-lineage MLA.

**(e) CUDA graphs are table stakes for decode but need a fallback.** vLLM uses piecewise
CUDA graphs because attention/MoE have data-dependent shapes; it pads decode batches to a
fixed set of captured sizes and dispatches by batch descriptor [vLLM
`cudagraph_dispatcher.py`, `compilation.py`]. DeepEP low-latency and DeepGEMM masked GEMM
exist specifically to keep MoE graph-capturable. Capture adds memory and warmup cost and must
be gated per backend.

**(f) Prefix caching is cheap only if engineered.** vLLM V0's prefix caching caused enough CPU
overhead to be off by default; V1 made it near-zero via intrusive free lists, append-only
block IDs, and minimal Python object creation, then enabled it by default [7].

**(g) Preemption by recompute, not swap.** vLLM V1 deleted GPU↔CPU swap and preempts by
freeing blocks, resetting `num_computed_tokens`, and re-queueing [vLLM `scheduler.py:273-281`].
Recompute is cheap when prefix caching is present. Host offload is an optional tier, not the
default preemption mechanism.

**(h) Admission and backpressure are widely mishandled.** vLLM V1 has **no** 429/503 overload
path and an unbounded waiting deque [vLLM `scheduler.py:334-338`]; overload is externalized to
a router. LightLLM returns 429 from a token-load admission model and lmdeploy bounds in-flight
work with a fixed request-handle pool. For a *production-grade* system we should not copy
vLLM's unbounded queue; we adopt bounded admission (see D14).

**(i) Observability needs event timestamps, not just counters.** vLLM computes TTFT/TPOT/
queue/prefill/decode/E2E from engine-core monotonic events (`QUEUED`, `SCHEDULED`,
`PREEMPTED`) relayed to the frontend [vLLM `stats.py`, `docs/design/metrics.md`]. Metrics
collection is deliberately kept out of the inner loop.

**(j) Distributed is three orthogonal axes with different costs.** TP shards every weight and
communicates per layer (latency-critical, NVLink-class); PP shards layers and adds pipeline
bubbles (overlappable with micro-batches); EP shards experts and uses all-to-all (decode is
latency-critical, hence DeepEP low-latency + masked grouped GEMM). vLLM flattening rules:
EP collapses TP to 1 within MoE; DP-TP hybrids exist in lmdeploy/LightLLM to give MoE more
effective parallelism.

**(k) The cache/offload ecosystem converged on the same essentials**: engine-aligned chained
hashes, CPU pinned memory as the first offload tier, async side-stream copies with completion
polling, pinning/refcounts, and fail-safe degradation to miss. Disaggregated/remote KV pays
off only for long shared prefixes at scale.

**(l) Speculative decoding and constrained decoding are scheduler features, not bolt-ons.**
Spec decode needs lookahead token reservation, draft-token injection, and rollback accounting
in the scheduler [vLLM `scheduler.py:897-912`]. Structured output gates scheduling through an
FSM/bitmask state (`WAITING_FOR_FSM`) and a grammar bitmask applied before sampling.

### 2.3 Fundamental vs advanced (how we classify features)

**Fundamental (must be in the initial architecture):**
- Unified token-budget scheduler; continuous batching; chunked prefill.
- Paged KV cache with a block table and a physical/logical separation.
- Prefix caching with chained content hashes and correct namespace isolation.
- Pluggable attention backend with a naive Eigen reference implementation.
- Process separation of frontend/tokenizer from the engine loop; async IPC.
- Bounded admission and an explicit overload policy.
- Request cancellation, timeouts, and graceful shutdown.
- Per-request event timestamps and Prometheus metrics.
- Deterministic correctness tests against a reference.
- Config/model abstraction and streaming weight loading.

**Fundamental for the §1.4 priority targets (promoted 2026-09-18):**
- MoE routing, grouped/masked expert GEMM, and expert parallelism (both Flash targets).
- MLA latent KV with weight absorption, and the optional sparse indexer (DeepSeek-V4.1-Flash).
- Hybrid linear attention with a recurrent `StateCache` (Qwen3.8-Flash-Next).
- FP8/BF16 (and FP4 where supported) load-time quantization with explicit scales.
- MTP proposers (both Flash targets).
- Config-driven long-context positional scaling (all targets).

**Advanced (introduce only with evidence, behind interfaces):**
- CUDA graphs (introduced in M3 once the decode shape model is stable).
- Quantization (M5; INT4 weight-only first on Ada).
- Speculative decoding (M5; n-gram first, then draft model/EAGLE).
- Structured/constrained generation (M5).
- LoRA / multi-adapter (M5).
- KV offload to host/SSD/remote (M7).
- Prefill/decode disaggregation (M7+).
- Elastic GPU VMM (research spike only).
- Multimodal inputs (later).

### 2.4 Where systems disagree (and we must choose)

1. **Prefix-cache structure: hash-blocks (vLLM) vs radix tree (SGLang).** Hash-blocks are
   simpler, low-overhead, and produce stable append-only block IDs (good for CUDA graphs), but
   only reuse whole blocks. The radix tree reuses finer-grained page-aligned prefixes and has
   pluggable eviction, at higher Python metadata cost and with in-place tree mutation. We
   choose hash-blocks initially and design the interface so a radix implementation can replace
   it (D5).
2. **Preemption: recompute (vLLM V1) vs swap/host offload (LMCache/FlexKV).** Recompute is
   simpler and sufficient when prefix caching exists; offload is an extension for long-context
   and cross-instance reuse. We choose recompute for v1 and offload later (D13).
3. **Admission: unbounded (vLLM) vs bounded/429 (LightLLM/lmdeploy).** We choose bounded.
4. **Execution overlap: two-stream overlap scheduling (mini-sglang) vs async scheduling with
   output placeholders (vLLM V1) vs PP batch queue (vLLM).** We start synchronous, add
   overlap in M3 (D7/D8).
5. **Quantization depth: weight-only (broad) vs W8A8/FP8 (Hopper) vs INT4 KV (lmdeploy).**
   Ada constrains us; we start weight-only INT4 and FP8-KV memory savings (D9).

### 2.5 Assumptions that still need validation

- That a single Triton paged-attention kernel can match FlashAttention/FlashInfer closely
  enough on `sm_89` to be the default optimized path (or that we should depend on FlashInfer).
- That hash-block prefix caching gives high hit rates on realistic traces (vs radix).
- CUDA-graph capture of a mixed prefill+decode batch (vs decode-only) on Ada.
- That FP8 tensor-core ops are usable through the chosen kernel libraries on `sm_89`.
- Realistic target performance ratios vs vLLM on this exact box (must be measured).
- The right default block size (16 vs 32 vs 64) for Ada given 16 GB.
- The exact architectures of the §1.4 priority targets (linear-attention variant, MoE
  granularity, MLA/sparse-indexer details, MTP heads, quantization formats) and their
  availability; E10 resolves this before M0.5 commits to a component list.

These are turned into experiments in §17.

---

## 3. Proposed architecture

### 3.1 Layering (logical)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Serving/frontend process                                                    │
│  - HTTP server (OpenAI-compatible: /v1/chat/completions, /v1/completions,   │
│    /v1/models, /health, /metrics)                                           │
│  - Request validation, sampling-param normalization, templating             │
│  - Per-request output queues; SSE streaming; abort/timeout                  │
│  - Tokenizer/Detokenizer workers (separate threads/processes)               │
└───────────────▲──────────────────────────────┬─────────────────────────────┘
                │ async IPC (ZMQ + msgpack/msgpack-like framing)              │
                │ ADD / ABORT / UTILITY  |  OUTPUT deltas + engine events     │
┌───────────────┴──────────────────────────────▼─────────────────────────────┐
│ Engine-core process (single writer)                                         │
│  Engine loop:  scheduler.schedule() → executor.execute_model() →            │
│                scheduler.update_from_output()                               │
│  - Scheduler: token-budget allocation, admission, preemption, priority      │
│  - KV-cache manager: block pool, prefix hashing, per-request block tables   │
│  - Executor: dispatches to workers (local or multiprocess/Ray)              │
│  - Structured-output manager (FSM/bitmask)                                  │
└───────────────▲──────────────────────────────┬─────────────────────────────┘
                │ per-step command / model output                             │
┌───────────────┴──────────────────────────────▼─────────────────────────────┐
│ Worker(s) — one per (TP,PP,EP) rank                                         │
│  - Model runner: input prep, model forward, attention, sampler              │
│  - KV cache tensors (paged), block tables, CUDA graphs                      │
│  - Distributed collectives (NCCL / custom)                                  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component responsibilities and interfaces

Interfaces are minimal, explicit, and stable. Names below are conceptual; concrete
signatures will be finalized at implementation.

**Frontend / API layer**
- `Server`: ASGI application exposing OpenAI-compatible routes; owns middleware for request
  ids, cancellation, and load accounting.
- `ServeEngine` (protocol): `generate(request) -> async iterator of RequestOutput`;
  `abort(request_id)`; `add_lora`/`remove_lora`; `sleep`/`wake`; `health`; `shutdown`.
- `EngineClient`: async IPC client to the engine core. Sends `EngineCoreRequest` (add/abort/
  utility), receives `EngineCoreOutputs` (per-request delta tokens, finish reason, events).
  Must detect engine death via a sentinel and surface `EngineDeadError`.
- `TokenizerService`: encode text/chat template → token ids; detokenize token-id deltas
  incrementally with correct UTF-8/byte-BPE boundary handling and stop-string detection.
- `OutputProcessor`: converts engine deltas into `RequestOutput` objects; computes finish
  reasons; tracks per-request metrics events.

**Engine core**
- `EngineCore`: owns scheduler + executor + structured-output manager; runs the step loop.
  - `step() -> (outputs, did_work)`: `schedule`; `execute_model`; `update_from_output`.
  - `add_request`, `abort_requests`, `preprocess_add_request` (off-loop in a thread).
- `Scheduler` (interface `SchedulerInterface`): `schedule() -> SchedulerOutput`;
  `update_from_output(sout, mout) -> EngineCoreOutputs`; `add_request`; `finish_requests`;
  `has_requests`; `get_request_counts`; `make_stats`; `update_draft_token_ids`; `shutdown`.
- `KVCacheManager` (interface): `get_computed_blocks(request) -> (blocks, n)` (prefix hit);
  `allocate_slots(request, num_new_tokens, ...) -> blocks | None`; `free(request)`;
  `reset_prefix_cache()`; `get_block_ids(blocks)`; `get_usage()`. Owns a `BlockPool` and a
  coordinator of per-attention-type managers.
- `RequestQueue`: FCFS and priority implementations, both **bounded**.
- `StructuredOutputManager`: grammar compilation/caching; `get_grammar_bitmask(reqs)`.
- `Executor` (interface): `execute_model(sout, non_block) -> ModelRunnerOutput | Future`;
  `max_concurrent_batches`; `get_kv_cache_specs`; `initialize_from_config`; failure callback.

**Workers / model runner**
- `ModelRunner`: `execute_model(sout) -> ModelRunnerOutput`; `_prepare_inputs`;
  `compute_logits`; `sample`; `capture_cudagraphs`; `propose_draft_tokens`.
- `Model` (per-architecture): uniform constructor, `forward(input_ids, positions, attn_meta)`.
- `AttentionBackend` (interface): `get_name`, `get_kv_cache_shape`, `get_supported_dtypes`,
  `validate_head_size`, `build_metadata`, `forward`, `capture/replay` hooks.
- `Sampler`: logits processing pipeline (bad words, penalties, logit processors, grammar
  mask) then temperature/top-k/top-p/greedy.
- `BlockTable`: per-request physical block id mapping; `compute_slot_mapping`; GPU mirror.
- `WeightLoader`: streaming safetensors loader with sharding/fusion, meta-device init.

**Cross-cutting**
- `DistributedCommunicator`: group creation (TP/PP/DP/EP); `all_reduce`, `all_gather`,
  `reduce_scatter`, `send/recv`, all-to-all; graph-capture-aware; NCCL by default.
- `Metrics/Events`: engine emits monotonic events; frontend computes intervals; Prometheus
  exporter; OTel spans using GenAI semantic conventions.
- `Config`: immutable, validated config objects (model, cache, scheduler, parallel,
  observability) with explicit defaults, derived at startup.

### 3.3 Request lifecycle

```
HTTP request
  → validate params, render chat template (frontend)
  → tokenize (TokenizerService)
  → EngineCoreRequest(msgpack) ──IPC──> engine input thread
  → preprocess (build Request; assign arrival/queue timestamps; emit QUEUED)
  → scheduler.add_request → waiting queue
  → schedule(): prefix match → allocate KV blocks → RUNNING, emit SCHEDULED
       (may be chunked: num_new_tokens < prompt length)
  → executor.execute_model(SchedulerOutput)
        worker: prepare inputs → model forward → sample → ModelRunnerOutput
  → scheduler.update_from_output: append tokens, check stop, free KV on finish
  → EngineCoreOutputs(delta ids, finish, events) ──IPC──> frontend
  → detokenize incrementally, match stop strings, stream SSE chunk
  → on stop/EOS/max_tokens/abort/timeout: finish; free KV; emit FINISHED
```

Key states (`RequestStatus`): `WAITING`, `WAITING_FOR_FSM`, `WAITING_FOR_REMOTE_KVS`
(reserved), `RUNNING`, `PREEMPTED`, then terminal `FINISHED_STOPPED`, `FINISHED_LENGTH_CAPPED`,
`FINISHED_ABORTED`, `FINISHED_ERROR`. `is_finished = status > PREEMPTED`.

### 3.4 Execution model (per step)

1. Scheduler computes `SchedulerOutput`:
   - `scheduled_new_reqs` (full data for first sight), `scheduled_cached_reqs` (diff only),
   - `num_scheduled_tokens: {req_id: n}`, `total_num_scheduled_tokens`,
   - `finished_req_ids`, `scheduled_spec_decode_tokens`, `block_tables`, `slot_mapping`,
   - grammar bitmask, KV-connector metadata (later).
2. Executor sends the step to workers; worker updates cached request state, prepares inputs
   (persistent buffers + diffs), runs model under the attention backend, samples.
3. `ModelRunnerOutput` returns sampled ids, logprobs (optional), per-request accepted spec
   counts, and `req_id → index` mapping.
4. Scheduler applies outputs: append, stop-check, free, update spec accounting, produce
   per-client `EngineCoreOutputs`.
5. Frontend maps outputs to request queues; streams; records metrics.

Design rule: **workers cache request state; only diffs cross the process boundary** (vLLM V1
lesson). This keeps IPC payloads small and CPU overhead low.

---

## 4. Major architectural decisions (ADR-style)

Each decision lists: problem, options, evidence, tradeoffs, recommendation, rationale.

### D1 — Implementation language and kernel strategy

- **Problem:** Own the engine but not re-derive solved infrastructure (tensor math, BLAS,
  tokenization, file formats, collectives).
- **Options:** (a) C++23 core + established libraries; (b) Python + PyTorch with
  Triton/CUDA extensions; (c) mixed C++ engine with Python frontend.
- **Evidence:** vLLM V1 keeps the engine in Python and moved Python overhead out of the hot
  loop via process separation [7]; mini-sglang is ~8k Python lines. But the low-level
  production engines that win on latency and deployability are C/C++ (TensorRT-LLM,
  llama.cpp, lmdeploy's turbomind, Mooncake's transfer engine, DeepEP/DeepGEMM, FlashMLA).
  The chosen project constraints mandate C++23 + Google style.
- **Tradeoffs:** C++ gives predictable latency, low per-step overhead, easy deployment, and
  direct access to CUDA; it costs iteration speed and requires more discipline for
  correctness/safety. We mitigate with a reference path, strong testing, and libraries.
- **Recommendation:** **C++23**, built with CMake/Ninja, Google C++ Style Guide, formatted
  with clang-format. Use well-established third-party libraries throughout and write only
  the engine-specific code ourselves:
  - **Tensor math / BLAS:** Eigen 3.4 (CPU) now; cuBLAS/CUTLASS on GPU later. Do not write
    GEMM ourselves.
  - **Attention kernels:** FlashAttention/FlashInfer/CUTLASS or our own Triton/CUDA kernel
    later; a simple Eigen/CUDA reference attention is the correctness oracle.
  - **JSON/config:** `nlohmann/json`.
  - **Weights:** the official HuggingFace `safetensors` C++ header.
  - **Tokenization:** HuggingFace `tokenizers` via `tokenizers-cpp` (Rust core + C++ API),
    behind an interface so the engine can run on raw token ids.
  - **Distributed:** NCCL (via `nccl.h`/`nccl.hpp`), not hand-written collectives.
  - **CUDA:** CUDA runtime/driver + cuBLAS; later Thrust/CUB for primitives.
  - **Testing:** GoogleTest + GoogleBenchmark.
  - **CLI/logging/metrics:** `gflags`/`absl` or `CLI11`; `spdlog` for logging; a Prometheus
    client library for metrics (or a small exporter over our own counters).
- **Rationale:** Satisfies the C++23/Google-style mandate, avoids "not invented here"
  duplication, and keeps our engineering effort on the differentiating engine internals.
- **Revisit:** if a library's license, build complexity, or runtime cost is unacceptable for
  a component, replace it behind its interface (e.g., swap `tokenizers-cpp` for a
  `sentencepiece` backend), not by rewriting the engine.
- **Pinning:** all third-party dependencies are version-pinned (CMake `FetchContent` with
  `GIT_TAG`/releases) and recorded in the lock/version file.

### D2 — Unified token-budget scheduler (no prefill/decode phases)

- **Problem:** Prefill and decode have opposite compute/memory profiles; naive batching
  causes stalls and coupling.
- **Options:** (a) phase-separated scheduling (classic); (b) mixed/continuous batching with
  phases; (c) unified token-budget (vLLM V1).
- **Evidence:** vLLM V1 represents work as `{req_id: n_tokens}` against a budget and notes it
  is "general enough" for chunked prefill, prefix caching, and spec decode
  [vLLM `scheduler.py:180-189`; 7]. Chunked prefill removes prefill/decode stalls [3].
- **Tradeoffs:** Unified is simpler and composable; it gives up explicit phase-level controls
  (e.g., strict decode-first), which can be reintroduced as a scheduling policy on top.
- **Recommendation:** Adopt the unified token-budget scheduler as the core abstraction.
  Provide a policy layer for phase preference (e.g., prefill-first vs decode-first) without
  changing the representation.
- **Rationale:** This is the single biggest simplification and the enabling abstraction for
  later features. Retrofitting it is painful (vLLM V0→V1).
- **Alternative kept:** if measurement shows decode SLOs suffer, add a decode-reserve budget
  (a fraction of the token budget reserved for running decode requests), which
  mini-sglang approximates with `reserved_size`.

### D3 — Admission, preemption, and scheduling policy

- **Problem:** Under KV pressure and high concurrency, what runs, what waits, and what is
  evicted?
- **Options:** preemption: recompute vs swap/host-offload; admission: bounded vs unbounded;
  priority: FCFS vs priority.
- **Evidence:** vLLM V1 recomputes and has no swap [vLLM `scheduler.py:273-281`]; prefix
  caching makes recompute cheap. vLLM's waiting queue is unbounded [vLLM
  `scheduler.py:334-338`]; LightLLM returns 429 from token-load admission; lmdeploy bounds
  in-flight work with a request-handle pool. Mooncake uses prediction-based early rejection
  for overload [6].
- **Tradeoffs:** Recompute is simple and cache-friendly but can amplify load; host offload
  preserves progress but adds transfer latency/complexity. Bounded admission protects latency
  and protects the engine but reduces acceptance under bursts. Priority improves SLO handling
  but risks starvation without aging.
- **Recommendation:**
  - Preempt by **recompute** in v1 (free blocks, reset computed tokens, prepend to waiting).
  - **Bound the waiting queue** by both request count and estimated token/KV demand; reject
    with 429/503 when full (see D14).
  - Implement **FCFS and priority** queues; priority preempts the lowest-priority running
    request; add optional aging to prevent starvation.
- **Rationale:** Recompute + bounded admission gives predictable latency and bounded memory,
  which the production requirements demand and which vLLM's design lacks by itself.

### D4 — KV cache paging model

- **Problem:** Variable-length sequences cause fragmentation and limit batch size.
- **Options:** contiguous per-request allocation; paged blocks (PagedAttention); virtual-memory
  elastic; hybrid (paged + sliding window).
- **Evidence:** PagedAttention gives near-zero external fragmentation and 2–4× throughput
  [1]. vLLM allows block sizes {1,8,16,32,64,128}, default 16 on CUDA [vLLM `config/cache.py`,
  `platforms/cuda.py`]. mini-sglang stores a token-level page table and derives block tables
  per backend, decoupling page size from kernel requirements [mini-sglang].
- **Tradeoffs:** Small blocks → higher hit rate and less internal fragmentation but more
  metadata and gather overhead; large blocks → fewer gathers and required by some MLA kernels.
- **Recommendation:** Paged pool, **configurable block size with default 16** for dense
  attention on Ada (validate 16 vs 32 vs 64 in E1). Keep a token-level logical mapping that
  backends translate to their native block tables, so page-size policy is decoupled from the
  attention backend.
- **Rationale:** Paging is fundamental and proven; mini-sglang's decoupling is a good
  extensibility pattern.

### D5 — Prefix caching structure

- **Problem:** Reuse KV across requests/sessions to cut TTFT and recompute.
- **Options:** (a) hash-chained immutable blocks (vLLM V1); (b) radix tree (SGLang);
  (c) no cross-request cache.
- **Evidence:** vLLM's chained hash `(parent_hash, token_ids, extra_keys)` proves prefix
  presence, allows early-break lookup, and — with intrusive free lists and append-only block
  IDs — adds <1% overhead at 0% hit rate, so it is on by default [7][vLLM
  `kv_cache_utils.py:547-573`, `block_pool.py`]. SGLang's radix tree reuses arbitrary
  page-aligned prefixes and supports pluggable eviction (LRU/LFU/SLRU/priority) at higher
  metadata cost [SGLang `radix_cache.py`, `evict_policy.py`].
- **Tradeoffs:** Hash-blocks are simpler/faster with coarse (block) granularity; radix is
  finer-grained and policy-rich but mutates a Python tree per step.
- **Recommendation:** Implement **hash-chained block prefix caching** first, behind a
  `PrefixCache` interface. Namespace the hash with model id, dtype, KV layout, TP/PP rank,
  LoRA id, multimodal hashes, and a tenant `cache_salt`. Add a radix implementation later if
  E2 shows material hit-rate gains.
- **Rationale:** Correctness, low overhead, and stable block IDs (needed for CUDA graphs)
  dominate early; radix is an optimization with a higher complexity budget.

### D6 — Attention backend strategy

- **Problem:** Attention dominates both prefill compute and decode memory traffic; kernels
  are hardware- and model-specific.
- **Options:** pure PyTorch reference; FlashAttention (v2/v3); FlashInfer; Triton paged;
  FlashMLA; framework-specific (TRT-LLM).
- **Evidence:** vLLM selects backends by arch/model/dtype [vLLM `selector.py`,
  `platforms/cuda.py`]; Ada (`sm_89`) tops out at FA2-class, FA3 is Hopper; FlashMLA is
  required for MLA; mini-sglang uses FA3 prefill + FlashInfer decode on Hopper and FlashInfer
  on Ada [mini-sglang `attention/`].
- **Tradeoffs:** Third-party kernels are fast and battle-tested but add dependencies and a
  narrow contract (head size, dtype, layout, page size). A self-written Triton kernel gives
  full control and portability but will lag in raw performance initially.
- **Recommendation:** Define a backend registry and implement, in order: (1) a **naive
  PyTorch/CUDA reference** (correctness oracle, always available), (2) a **Triton/CUDA paged
  attention** (prefill + decode + split-KV) as our own optimized path, (3) **FlashInfer
  and/or FlashAttention** via optional adapters for maximum speed on supported hardware.
  Default selection by arch/dtype/head-size with a documented support matrix; fall back to
  reference when unsupported.
- **Rationale:** Owning a correct reference plus our own optimized kernel satisfies the
  "not a thin wrapper" goal; third-party adapters keep us competitive. This mirrors how
  mature systems keep a fallback path.
- **Risks:** Triton kernel bugs are subtle; mitigated by differential tests vs the reference
  (see §13).

### D7 — Process/IPC model and concurrency

- **Problem:** Python/HTTP/detokenization overhead stalls the GPU; shared state invites races.
- **Options:** (a) single process with threads/asyncio; (b) frontend process + engine-core
  process(es) + tokenizer workers; (c) C++ engine.
- **Evidence:** vLLM V1 moves AsyncLLM/tokenization/detokenization into separate
  processes/threads and caches request state on workers, transmitting only diffs
  [7][vLLM `core.py`, `core_client.py`]; mini-sglang spawns API/tokenizer/detokenizer/
  per-rank scheduler processes with typed ZMQ messages [mini-sglang `server/launch.py`].
- **Tradeoffs:** Multi-process adds IPC complexity and a death-handling burden; single
  process is simpler but cannot hide Python overhead.
- **Recommendation:** **Frontend process + one engine-core process + tokenizer/detokenizer
  workers**, connected by ZMQ with length-framed, versioned messages (msgpack or custom
  binary). Start with **one engine process and an in-process executor**; support a
  multiprocess executor and Ray later. Include a death sentinel, backpressure on the input
  socket, and a handshake state machine.
- **Rationale:** Proven design that keeps per-step overhead low and scales to DP/TP
  topologies; interface-first so transport can change.

### D8 — CUDA graphs and CPU/GPU overlap

- **Problem:** Kernel-launch and Python overhead dominate decode at small batch; collectives
  are latency-sensitive.
- **Options:** eager; full CUDA graph; piecewise CUDA graphs; overlap scheduling; async
  scheduling with output placeholders; PP micro-batching.
- **Evidence:** vLLM V1 uses piecewise graphs because attention is data-dependent, padding to
  captured decode sizes [vLLM `cudagraph_dispatcher.py`; 7]. mini-sglang overlaps CPU
  scheduling with GPU compute on two streams, processing results one iteration late
  [mini-sglang `scheduler.py:83-131`]. vLLM async scheduling uses output placeholders to
  schedule ahead [vLLM `async_scheduler.py`].
- **Tradeoffs:** Graphs add memory, warmup, and capture constraints; overlap adds complexity
  and one-step-late semantics that complicate cancellation/abort.
- **Recommendation:** In order: (1) eager correctness; (2) **decode-only full CUDA graphs**
  padded to a fixed set of batch sizes; (3) **piecewise graphs** if mixed prefill+decode
  graphs are needed; (4) **two-stream overlap scheduling** with a kill-switch. Keep the
  one-step-late result pipeline guarded against double-free.
- **Rationale:** Decode graphs are the highest ROI; overlap is the next. Piecewise graphs
  and async scheduling are more invasive and only with evidence.

### D9 — Quantization

- **Problem:** 16 GB constrains model size/context; weights and KV dominate memory.
- **Options:** FP16/BF16 baseline; weight-only INT4 (AWQ/GPTQ/Marlin); W8A8 INT8; FP8 W8A8;
  KV-cache quantization (INT8/INT4/FP8).
- **Evidence:** Mature in vLLM/lmdeploy: AWQ, GPTQ/Marlin, INT8 W8A8, FP8 W8A8; FP8-KV is
  primarily a memory/throughput win because dequant is not fused into attention
  [vLLM quantization docs]. lmdeploy offers INT4/INT8 per-head-per-token KV and TurboQuant.
  Ada (`sm_89`) lacks the Hopper FP8 GEMM ecosystem maturity.
- **Tradeoffs:** Weight-only INT4 is broadly supported and reduces weight memory ~2× with
  modest accuracy loss; activation quantization needs calibration and hardware support;
  KV quantization directly extends context but can hurt long-context accuracy.
- **Recommendation:** Baseline BF16. Add **weight-only INT4 (AWQ/GPTQ via a dequant-GEMM or
  Marlin-class kernel)** as the first quant path on Ada. Add **FP8 weight/activation** only
  on Hopper+ [requires target GPU]. Treat **KV-cache FP8/INT8** as a memory feature with
  explicit accuracy validation; keep BF16 KV as default. Do not attempt custom quant kernels
  until a library path is insufficient.
- **Rationale:** Maximizes local usefulness (fit larger models in 16 GB) with the least
  risk; defers hardware-specific work.

### D10 — Speculative decoding

- **Problem:** Decode is memory-bound; each token costs a full weight read.
- **Options:** n-gram/prompt-lookup (no model); draft model; EAGLE/EAGLE3/Medusa/MTP; none.
- **Evidence:** vLLM supports n-gram (CPU, numba) and EAGLE/draft with hidden-state reuse,
  plus scheduler rollback accounting [vLLM `spec_decode/`, `scheduler.py:897-912`]; lmdeploy/
  LightLLM ship MTP/EAGLE variants. DeepSeek MTP is model-native.
- **Tradeoffs:** N-gram is free and helps repetitive/structured text but has low acceptance on
  open generation; draft/EAGLE gives higher acceptance but adds memory and complexity and is
  model-specific.
- **Recommendation:** Implement **n-gram first** (scheduler-integrated: `num_lookahead_tokens`,
  draft injection, rollback). Add **draft-model/EAGLE later** behind a `Proposer` interface.
  Validate statistically that output distribution is preserved; default off until measured.
- **Rationale:** N-gram is a low-risk way to build the scheduler hooks correctly; it also
  forces the right abstraction for later proposers.

### D11 — Structured/constrained generation

- **Problem:** JSON/schema/regex-constrained output is a common production requirement.
- **Options:** FSM/bitmask over vocabulary (outlines/xgrammar-style); grammar-guided decoding
  without bitmask; post-hoc validation (rejectable).
- **Evidence:** SGLang uses compressed FSMs and reports large structured-decoding speedups
  [4]; vLLM gates scheduling on `WAITING_FOR_FSM` and applies a grammar bitmask before
  sampling [vLLM `structured_output/`, `scheduler.py:357-364`].
- **Tradeoffs:** Bitmask construction and application cost CPU/GPU per step but guarantee
  validity; a bad FSM can block progress.
- **Recommendation:** Implement a `StructuredOutputManager` with compiled FSM/bitmask and a
  grammar cache; integrate as a pre-sampling logit mask and a `WAITING_FOR_FSM` gate.
  Defer to M5.
- **Rationale:** Cleanly separable; correct scheduler gating is the important part.

### D12 — Distributed inference

- **Problem:** Models/context exceed one GPU; need lower latency/ more throughput.
- **Options:** TP, PP, DP, EP, combinations; NCCL vs custom collectives; disaggregation.
- **Evidence:** TP is per-layer all-reduce (latency-critical), PP is stage send/recv with
  bubbles, EP is all-to-all (DeepEP). vLLM groups TP/PP/DP/EP and collapses TP within MoE
  [vLLM `parallel_state.py`, `fused_moe/config.py`]. PP micro-batching uses a batch queue of
  size `pp_size` [vLLM `core.py:300-352`]. lmdeploy DP-TP and LightLLM TP+SP overlap hide
  collective latency; collective fusion into GEMMs is the frontier.
- **Tradeoffs:** TP needs NVLink-class bandwidth; PP needs micro-batching and adds latency;
  EP needs load balancing; all add failure modes.
- **Recommendation:** Make parallelism a **config + group abstraction** from day one
  (`ParallelConfig`, `GroupCoordinator`). Implement **TP first** [target GPU], then **PP**
  [target GPU], then **EP/MoE** [target GPU]. Use NCCL; make collectives graph-capture-aware.
  Keep DP as independent engine replicas behind an external router initially. Disaggregation
  is M7+.
- **Rationale:** Interfaces first, implementations staged, because the dev box cannot
  validate multi-GPU.

### D13 — KV cache offload and disaggregation

- **Problem:** Prefix reuse/capacity beyond a single GPU's HBM; long-context economics.
- **Options:** host DRAM offload; SSD; remote/shared pool; software VMM elasticity;
  compression.
- **Evidence:** LMCache/Mooncake/FlexKV/divedb converged on chained block hashes, CPU pinned
  memory as L1, async side-stream copies, pinning, and fail-safe miss semantics; layerwise
  pipelining hides transfer under compute; Mooncake reports KV transfer at 4.2% of 32K TTFT;
  divedb reports 2.17–2.25× TTFT on prefix reuse with ~1.3% ITL; kvcached gives elastic VMM
  but with fragmentation/race/monkey-patch risks.
- **Tradeoffs:** Offload trades bandwidth and complexity for avoided recompute; worthwhile
  only for long, reused prefixes; remote/SSD adds operational cost.
- **Recommendation:** v1 uses in-engine prefix caching + recompute preemption. Add **host
  DRAM offload** as the first tier (M7) with an async, fail-safe `KVConnector` interface.
  Remote/SSD/disaggregation later. Treat kvcached-style VMM as a **research spike only**.
- **Rationale:** Highest benefit/cost first; matches the consensus essentials.

### D14 — Serving, admission control, and backpressure

- **Problem:** Overload must not destroy latency or memory; multi-tenant fairness.
- **Options:** unbounded queue (vLLM); bounded queue + 429 (LightLLM); semaphore/handle pool
  (lmdeploy); gateway-level rate limiting (aibrix).
- **Evidence:** vLLM has no 429 and an unbounded waiting deque; LightLLM uses token-load
  admission and returns 429; lmdeploy bounds work with a handle pool; aibrix enforces
  RPM/TPM at the gateway.
- **Tradeoffs:** Rejecting loses requests but protects SLOs; queueing improves acceptance but
  risks timeout storms.
- **Recommendation:** **Bounded admission at the API layer and engine**, with configurable
  `max_waiting_requests`, `max_concurrent_requests`, and a request timeout. Return 429 with
  `Retry-After` when over capacity and 503 when the engine is unhealthy/draining. Expose
  queue depth and load so a gateway can do smarter routing/rate limiting.
- **Rationale:** Explicit overload behavior is a core production requirement and an area
  where vLLM is weak; we choose the LightLLM/lmdeploy pattern.

### D15 — Observability

- **Problem:** Debuggability and SLO measurement cannot be added later.
- **Options:** counters only; Prometheus + structured logs; OpenTelemetry tracing; full.
- **Evidence:** vLLM relays engine-core monotonic events to the frontend and computes TTFT/
  TPOT/queue/prefill/decode/E2E, exposing Prometheus histograms and GenAI OTel spans
  [vLLM `stats.py`, `loggers.py`, `tracing.py`]. lmdeploy/LightLLM add API-waiting gauges and
  health monitors.
- **Recommendation:** Engine emits timestamped events (`QUEUED`, `SCHEDULED`, `PREEMPTED`,
  `FIRST_TOKEN`, `FINISHED`); frontend computes intervals and exports Prometheus metrics;
  OTel spans use `gen_ai.*` conventions; structured JSON logs with request ids; admin
  endpoints for prefix-cache reset and profiling. Keep metrics work off the engine hot path.
- **Rationale:** Directly mirrors the proven design and satisfies the observability goals.

### D16 — Weight loading and model abstraction

- **Problem:** Large checkpoints must load without a giant host state dict; TP sharding and
  fusion should be uniform.
- **Options:** full state dict in RAM; shard-on-load streaming; mmap; torch.distributed
  checkpoints.
- **Evidence:** mini-sglang streams safetensors directly to GPU, sharding and fusing
  QKV/gate-up on the fly with O(1) host memory [mini-sglang `models/weight.py`]. vLLM
  initializes on meta device and loads per-rank shards.
- **Recommendation:** Meta-device init + streaming, sharding, fusing loader; a uniform
  `Model`/`LinearBase` abstraction with TP-aware linear/embedding/lm-head layers; validate
  shapes/dtypes and reject unexpected keys.
- **Rationale:** Controls memory and makes TP correct by construction.

### D18 — Kernel/backend abstraction (CPU, CUDA, HIP, ...)

- **Problem:** Hardware-specific code must not leak into the model, scheduler, or KV-cache
  logic; InferX must run on CPU (correctness/CI), CUDA, HIP, and future accelerators without
  forking the engine.
- **Options:** (a) direct device code in the model; (b) a low-level *kernel HAL*
  (one virtual call per elementary op); (c) a **tensor/device + fused-op + stream backend**
  abstraction (ggml-backend/ATen-ExecutionProvider style); (d) a graph compiler/IR
  (TVM/IREE/ONNX-Runtime style).
- **Evidence:** llama.cpp's `ggml_backend` runs CPU/CUDA/HIP/Metal/Vulkan/SYCL behind one
  interface; PyTorch dispatches at tensor-op granularity; vLLM keeps attention backends and
  KV layouts behind per-backend classes and selects by capability. A kernel-only HAL does not
  address allocation, layout, streams, or graph capture, and a full graph IR is premature.
- **Tradeoffs:** Too fine-grained dispatch adds per-op overhead that hurts small-batch decode;
  too coarse prevents mixing backends and blocks the CPU oracle. A graph IR adds large
  complexity before there is evidence it is needed.
- **Recommendation:**
  1. Introduce `Device`, `DataType`, an owning device-tagged `Tensor` with a `Storage`
     abstraction, and an allocator/copy/synchronize surface.
  2. Define `Backend` at the granularity of **fused transformer primitives**
     (`Embedding`, `Linear`, `RmsNorm`, `SiluMul`, `Add`, `Attention`, `ComputeLogits`) plus
     `FromHost`/`ToHost`/`Copy`/`Zero`/`Synchronize`.
  3. Keep a mandatory **`CpuBackend`** (Eigen/BLAS) as the correctness oracle and CI path.
  4. Share one `GpuBackend` implementation between CUDA and HIP via a compile-time platform
     template; wrap cuBLAS/rocBLAS and FlashInfer/CK; do not hand-write GEMM.
  5. `KvCache` and attention layouts are backend-owned, exposed through the logical block
     interface; a `Capabilities` descriptor drives selection and fallback.
  6. Virtual dispatch only at fused-op granularity; use templates/CRTP inside kernels. Add
     `Capture`/`Replay` hooks when CUDA/HIP graphs land (M3).
  7. **Do not** build a graph IR/compiler until graph capture and cross-backend partitioning
     create a demonstrated need.
- **Rationale:** This is the smallest abstraction that removes hardware from the engine while
  preserving the CPU oracle, enabling CUDA/HIP later, and leaving room for graph capture.
- **Plan impact:** `tensor.h`/`ops.h` become the CPU kernel implementation behind
  `CpuBackend`; the model, KV cache, and generator talk only to `Backend`. This is the S1
  refactor recorded in §19a.

### D17 — Config and extensibility

- **Problem:** Divergent defaults and hidden coupling cause production surprises.
- **Options:** scattered CLI args; a single validated config object; plugin registries.
- **Evidence:** vLLM's `VllmConfig` is the engine-wide global; provider registries exist for
  models/backends/quant.
- **Recommendation:** Immutable, validated `Config` objects with explicit precedence
  (defaults → file/env → CLI), plus registries for models, attention backends, quant
  schemes, KV tiers, and schedulers. Fail fast on invalid combos (e.g., backend vs head
  size).
- **Rationale:** Reproducibility and maintainability; avoids the V0 feature-coupling failure.

### D19 — Hybrid linear attention and recurrent state cache

- **Problem:** Qwen3.8-Flash-Next is assumed to mix linear attention (chunked gated delta
  rule / Gated DeltaNet) with periodic gated full attention. Linear layers keep a fixed-size
  recurrent state instead of a growing KV cache, which the paged-KV-only model cannot
  express. [assumption to validate]
- **Options:** (a) force full attention over a growing KV (wrong if the checkpoint's layers
  are recurrent, and wasteful); (b) a dedicated recurrent **state cache** with chunked
  kernels; (c) integrate an external linear-attention/SSM kernel library.
- **Evidence:** Qwen3-Next uses Gated DeltaNet plus gated attention; Flash-Linear-Attention,
  Mamba2, and causal-conv kernels are established. vLLM models hybrid state through per-layer
  cache types and dedicated managers (sliding-window / Mamba), confirming that one cache type
  is insufficient.
- **Tradeoffs:** Recurrent state is O(1) memory per request, which is excellent for long
  context, but the kernels are less mature, chunked prefill must carry state across chunks,
  and prefix reuse requires storing state checkpoints rather than sharing KV blocks.
- **Recommendation:** Add a `StateCache` group (convolution state + recurrent state) to the
  composable cache manager; implement a naive CPU recurrence first as the oracle; add a
  chunked/scan kernel when measured. Disallow naive prefix reuse of state unless checkpoints
  are stored.
- **Rationale:** It is structural to the model family and must exist in the cache interface
  before M1 freezes batching.

### D20 — Mixture-of-experts execution and expert parallelism

- **Problem:** Both Flash models are assumed to be sparse MoE, which changes weight loading,
  the FFN op, parallelism, and decode batching.
- **Options:** (a) dense approximation (incorrect and prohibitive); (b) TP-sharded experts;
  (c) expert parallelism with all-to-all dispatch/combine and grouped/masked GEMM; optionally
  redundant experts / EPLB.
- **Evidence:** vLLM's modular `FusedMoE` (dispatch, permute, grouped experts, unpermute),
  DeepEP high-throughput and low-latency dispatch/combine, and DeepGEMM grouped/masked GEMM
  are the production shape; lmdeploy/LightLLM add EP and redundant experts.
- **Tradeoffs:** EP maximizes capacity but needs all-to-all and load balancing; TP-only is
  simpler but does not scale expert count; decode requires fixed-size buffers and masked GEMM
  to stay CUDA-graph compatible.
- **Recommendation:** A CPU reference MoE (router, top-k gating, per-expert MLP) in M0.5; the
  GPU path becomes EP with DeepEP + DeepGEMM in the distributed milestone, using fixed-size
  decode buffers for graph capture. Load balancing and EPLB follow only after measurement.
- **Rationale:** MoE must be an op and a parallel axis from the start; retrofitting it into a
  dense linear path is the V0 failure mode.

### D21 — MLA and sparse attention

- **Problem:** DeepSeek-V4.1-Flash is assumed to use multi-head latent attention (MLA), likely
  with a sparse-attention indexer, which changes both the KV representation and the decode
  kernel.
- **Options:** (a) materialize per-head K/V (memory blowup); (b) MLA with weight absorption
  and a compressed latent KV cache; (c) additionally a sparse indexer that selects top-k
  latent blocks.
- **Evidence:** DeepSeek-V2/V3 MLA, FlashMLA (including FP8 sparse decode), and vLLM's MLA
  and sparse-MLA backends show latent caching plus absorption is the production design.
- **Tradeoffs:** The latent cache is compact and fast but needs specialized kernels and
  careful weight absorption; sparse attention adds an indexer and a data-dependent read
  pattern that complicates CUDA graphs.
- **Recommendation:** A latent KV layout with MLA ops and absorption, CPU reference first, and
  the sparse indexer behind a capability flag. Pin block size to the kernel contract.
- **Rationale:** The latent representation is the model's memory strategy; treating it as a
  backend detail would force a cache-manager rewrite.

### D22 — Low-precision execution (FP8/FP4, native checkpoints)

- **Problem:** DeepSeek and likely Qwen Flash checkpoints are FP8-native; the dev box (Ada)
  has no viable FP8 GEMM path, while the target is Hopper/Blackwell.
- **Options:** (a) dequantize to BF16 at load (portable, more memory and bandwidth); (b)
  native FP8 W8A8 with explicit scales; (c) FP4/MXFP4 on Blackwell.
- **Evidence:** vLLM FP8 and modelopt-FP4 paths, lmdeploy blocked FP8/FP4, DeepGEMM FP8/FP4/
  MXFP8 kernels. Accuracy depends on scale granularity (per-tensor vs per-block/per-token).
- **Tradeoffs:** Dequantization is correctness-preserving and testable locally but loses the
  memory/bandwidth win; native FP8/FP4 needs target hardware and accuracy validation.
- **Recommendation:** Load FP8/BF16 (and FP4 where supported) with explicit scales; default
  to dequantize-to-BF16 on unsupported devices so parity is testable on the dev box; enable
  native low-precision kernels on the target rig behind a capability flag.
- **Rationale:** The checkpoints ship quantized; the loader and linear/MoE ops must understand
  that from day one, while keeping a portable correctness path.

### D23 — Multi-token prediction (MTP) as a proposer

- **Problem:** The Flash models are assumed to ship multi-token prediction heads, which are
  intended to be used as built-in drafters.
- **Options:** (a) ignore the heads; (b) use them as a `Proposer` inside the speculative
  decoding machinery; (c) train/attach a separate draft model.
- **Evidence:** DeepSeek MTP; EAGLE/draft proposers and vLLM's scheduler lookahead/rollback
  accounting already provide the required hooks.
- **Tradeoffs:** MTP raises acceptance and tokens/s but adds a head, draft-state management,
  and verification accounting; it must preserve the output distribution.
- **Recommendation:** Implement MTP behind the D10 `Proposer` interface, reusing lookahead
  reservation and rollback; validate distribution equivalence and acceptance rate before
  enabling by default.
- **Rationale:** The heads exist specifically to accelerate decode; the scheduler hooks from
  D10 make this an additive proposer rather than a new decode path.

### D24 — Long-context positional extension

- **Problem:** The targets ship very long context (128k–1M) using scaled RoPE and, for
  DeepSeek, sparse attention.
- **Options:** YaRN, NTK/dynamic scaling, linear interpolation; sliding-window layers; sparse
  attention indexers.
- **Evidence:** Major open models configure RoPE scaling explicitly in `config.json`;
  long-context quality depends on matching the reference scaling exactly.
- **Tradeoffs:** Correct scaling is required for parity; sparse/sliding patterns save memory
  and compute but change which KV is read and complicate graph capture.
- **Recommendation:** Make RoPE scaling fully config-driven (`rope_theta`, `rope_scaling`
  type/factor/original_max_positions) and validate parity at long context; combine with the
  hybrid/state and sparse-attention strategies above.
- **Rationale:** Long context is a first-class product requirement for these models and a
  common source of silent parity errors.

---

## 5. Memory and KV-cache strategy (detailed)

### 5.1 Memory budget and startup profiling

At startup, after weights are loaded, the engine must determine how many KV blocks it can
allocate without risking OOM during decode or CUDA-graph capture. Procedure:

1. Load weights (meta-init, then stream shards) and measure `mem_used_after_weights`.
2. Query free HBM; subtract a **headroom reserve** for: CUDA context/kernels, NCCL buffers,
   activation workspace (scales with `max_num_batched_tokens` and hidden size), CUDA-graph
   pools, and allocator fragmentation.
3. If TP > 1, all-reduce free memory across ranks and use the minimum; error if ranks are
   imbalanced beyond a threshold (mini-sglang raises at >2 GiB imbalance).
4. Compute `bytes_per_block = block_size * num_kv_heads_local * head_dim * dtype_size * 2 (K+V)
   * num_layers` (with per-backend layout exceptions, e.g., MLA compressed latent).
5. `num_blocks = floor(available_bytes / bytes_per_block)`; clamp to a reserved minimum.
6. Allocate one contiguous, pre-sized KV tensor (or per-layer tensors with a uniform stride)
   so block IDs are stable and addressable by kernels and CUDA graphs. Include a sentinel
   **null block** (block id 0) for positions that are "computed but not stored" (sliding
   window/prefix gaps).

**Invariant:** block IDs are **append-only and never renumbered**. Duplicate cached content
is tolerated rather than compacted, because compacting invalidates block tables and captured
graphs (vLLM lesson).

### 5.2 Paged block pool

- `BlockPool` owns a fixed `list[Block]` (metadata) and a free list.
- `Block` metadata: `block_id`, `ref_cnt`, write-once `block_hash`, intrusive `prev/next`
  free pointers, `is_null`.
- **Free list**: an intrusive doubly-linked queue over block objects (no per-step Python
  allocation), supporting O(1) middle removal (needed when a cached block is hit and pinned).
  Order: LRU from the front; when a request frees, free tail blocks first so shared prefixes
  survive longest.
- **Hash map**: `BlockHashWithGroupId → Block`, where the group id separates KV-cache groups
  (full attention, sliding window, etc.). No de-duplication by design.
- **Prefix hit**: walk the request's chained block hashes, stop at the first miss; cap the hit
  length at `num_tokens - 1` so the last token can be recomputed for logits. Under spec
  decoding, drop the last matched block so hidden states are recomputed.
- **Pinning**: a hit `touch`es blocks (removes from free queue if ref_cnt 0, increments ref).
  A block is evictable only at ref_cnt 0.
- **Caching**: only **full** blocks are hashed/cached; partial trailing blocks are never
  shared.
- **Eviction**: allocating from the free list evicts any cached hash for the blocks taken
  (resetting the hash), after which the blocks are reused.

### 5.3 Hash construction and namespace isolation

The block hash is `H(parent_hash, token_ids_tuple, extra_keys)` where `H` is a stable hash
(default SHA-256 over canonical CBOR; optionally a faster 64-bit hash with collision
detection). The root's parent is a process-wide random `NONE_HASH`. `extra_keys` **must**
include everything that changes the meaning of the KV:

- model identity + revision, weight dtype, KV dtype/layout,
- attention type/sliding-window parameters, layer range,
- TP/PP/EP rank and parallel config,
- LoRA adapter id (or a default),
- multimodal feature hashes and offsets (later),
- tenant `cache_salt` (only on the first block) for isolation.

Getting this wrong silently corrupts attention. A dedicated unit test enumerates the keys and
asserts that changing any of them changes the resulting hash/cache-hit behavior.

### 5.4 Per-attention-type KV managers

The manager is composable from the start, because the priority targets combine several cache
types under one page/block accounting:

- `FullAttentionManager`: linear hash-chain scan; prefix caching enabled.
- `SlidingWindowManager`: reuse only with `ceil((window-1)/block_size)` contiguous hits,
  searching right-to-left; free out-of-window blocks; fill gaps with the null block.
- `LatentMlaManager`: treats the MLA compressed latent as the cached element (no split K/V);
  prefix caching applies to latent blocks.
- `StateCacheManager`: per-request convolution/recurrent state for hybrid linear attention;
  prefix reuse requires stored state checkpoints (D19).
- `ChunkedLocalAttentionManager`, `MambaManager`, `CrossAttentionManager`: as needed by E10,
  with prefix caching disabled where semantics forbid it.

A `KVCacheCoordinator` aggregates groups; for hybrid models it intersects cache hits across
groups and truncates to the shortest. This mirrors vLLM's coordinator design and avoids a
painful retrofit when adding the Flash-target cache types.

### 5.5 Block table and slot mapping

- Each request has a block table: `block_table[logical_block] = physical_block_id`.
- The worker mirrors block tables on GPU. `slot_mapping` for a newly computed token at
  position `p` is `block_table[p // block_size] * block_size + (p % block_size)`.
- `reshape_and_cache` (or the attention backend's store path) scatters new K/V into the pool
  via the slot mapping.
- The scheduler passes block tables and slot mappings in `SchedulerOutput`; workers update
  only diffs.

### 5.6 KV allocation algorithm (per scheduled request)

Given a request with `num_computed_tokens` already materialized, a prefix hit of `H` tokens,
and `n` new tokens to compute:

1. Determine `num_required_blocks = cdiv(H + n, block_size)` plus lookahead (spec) blocks.
2. Count already-owned blocks; compute needed vs evictable cached blocks.
3. If `needed > free_blocks`, return `None` → scheduler preempts (recompute) or the request
   stays waiting.
4. Touch pinned prefix blocks; save/allocate new blocks; record the computed-block count.
5. Cache newly **full** blocks (up to `min(computed+new, request.num_tokens)`), never caching
   rejected speculative tokens.

### 5.7 KV cache data types and layouts

- Default BF16. Optionally FP8-E4M3 or INT8 KV on supported hardware, stored with per-tensor
  or per-head/per-token scales; dequant happens in the attention kernel.
- Layout must match the active backend (e.g., `[2, num_blocks, block_size, kv_heads, head_dim]`
  for FlashAttention-style; `[num_blocks, block_size, kv_heads, head_dim]` for some Triton
  paths; MLA uses a compressed latent). The backend declares its required layout and block
  alignment; the cache manager allocates accordingly.
- **Validation:** a backend must reject unsupported (dtype, head_size, block_size) at config
  time, not at first forward.

---

## 6. Scheduling, admission, and backpressure (detailed)

### 6.1 Scheduling algorithm (per step)

```
token_budget = max_num_batched_tokens
running = list of running requests (ordered)
waiting = bounded FCFS/priority queue

# 1. Running requests (decode + continued prefill)
for req in snapshot(running):
    n = req.num_tokens_with_spec - req.num_computed_tokens     # tokens still needed
    if chunked and n > 0: n = min(n, token_budget, long_prefill_threshold)
    n = min(n, token_budget, max_model_len - 1 - req.num_computed_tokens)
    if n == 0: continue            # allow lower-priority requests to proceed
    blocks = kv.allocate_slots(req, n)      # incl. prefix hit bookkeeping
    if blocks is None:
        preempt(lowest-priority running request)   # free, reset, prepend to waiting
        if preempted is req: break
        continue
    schedule req for n tokens; token_budget -= n

# 2. Waiting requests (admit only if nothing was preempted this step)
while token_budget > 0 and len(running) < max_num_seqs and waiting:
    req = waiting.peek()
    if req needs FSM and not ready: skip/keep waiting
    hit = kv.get_computed_blocks(req)       # prefix cache
    n = min(req.num_prompt_tokens - hit_len, token_budget)
    if not chunked and n < full_prompt: keep waiting (cannot fit this step)
    blocks = kv.allocate_slots(req, n, prefix_hit)
    if blocks is None: break                # no KV; stop admitting
    pop waiting; mark RUNNING; token_budget -= n; emit SCHEDULED
```

Properties that must hold (tested as invariants):
- `sum(num_scheduled_tokens) <= max_num_batched_tokens`.
- `len(running) <= max_num_seqs`.
- A request's `num_computed_tokens` never exceeds its `num_tokens`.
- Every running request has KV blocks for all computed tokens.
- `waiting` size ≤ configured bound.
- No request is both running and waiting.

### 6.2 Chunked prefill

- Enabled by default for generation tasks. A long prompt is split across steps so decode
  requests are never stalled by a giant prefill [3].
- Controls: `max_num_batched_tokens` (global), `long_prefill_token_threshold` (cap a single
  prefill chunk), `max_num_partial_prefills` (how many prompts may be mid-prefill).
- Multimodal item chunking can be disabled (needs an encoder cache to hold embeddings).
- A request mid-prefill must not sample; it enters decode only when fully prefilled. The
  `Request` object tracks `num_computed_tokens`; sampling/stop checks apply only when
  `num_computed_tokens >= num_prompt_tokens`.

### 6.3 Admission control (bounded)

Two limits, both configurable:
- **API/ingress limit**: `max_concurrent_requests` (semaphore or handle pool) bounds work
  admitted from the frontend. Excess → 429 with `Retry-After`.
- **Engine waiting limit**: `max_waiting_requests` and an optional estimated-KV bound. When
  exceeded, the engine rejects (frontend maps to 429) rather than growing memory.

Additional knobs: `max_num_seqs` (running cap), `max_num_batched_tokens` (compute cap).

**Overload behavior:** the system must reach a steady state where queue depth and latency are
bounded. If requests time out while waiting, they are removed and their resources freed.
A request timeout (wall clock from arrival) is enforced by the frontend.

### 6.4 Preemption and fairness

- When KV allocation fails for a running request, preempt the lowest-priority running request
  (FCFS: the newest). Free its blocks, reset `num_computed_tokens = 0`, set `PREEMPTED`,
  prepend to the waiting queue, and emit a `PREEMPTED` event.
- Prefix caching makes recompute of the shared prefix cheap.
- Priority queue orders by `(priority, arrival_time)`. Add **aging** (decrement effective
  priority over time or a max-wait boost) to prevent starvation.
- Consider a small **decode reserve**: reserve a fraction of the token budget for running
  decodes so a burst of prefills cannot inflate ITL. Validate in E3.

### 6.5 Continuous batching and batch construction

- Each step builds one padded batch from `SchedulerOutput`. Decode batches are padded to a
  captured CUDA-graph size when graphs are enabled.
- Persistent input buffers are reused across steps; only changed rows are written (the
  "persistent batch" technique) to minimize CPU work.

---

## 7. Concurrency model

- **Frontend**: asyncio event loop. One task per request handles the streaming response;
  a single background task drains engine outputs and fans them out to per-request queues.
  Cancellation (`asyncio.CancelledError`) and `http.disconnect` trigger abort.
- **Tokenizer/detokenizer**: worker threads/processes; CPU-bound tokenization must not block
  the event loop. Detokenization is incremental and per-request stateful.
- **Engine core**: one dedicated thread/process owns the scheduler and the step loop (single
  writer). Socket I/O runs on separate daemon threads: an input thread decodes/adds requests
  (including `preprocess_add_request`) and an output thread encodes/pushes outputs. The step
  loop only blocks on input when idle.
- **Workers**: model forward is launched from the engine thread; with overlap scheduling the
  launch and result processing run on separate streams and results are consumed one iteration
  late.
- **Synchronization**: request state has exactly one owner at a time. Cross-boundary data is
  immutable and versioned. Avoid locks in the hot loop; prefer queues and single-writer
  ownership.
- **GPU streams**: default compute stream; separate H2D/D2H copy streams for weight/KV
  transfer; a stream for overlap scheduling (launch next batch while processing previous
  results). Correct event synchronization is a tested invariant.

---

## 8. GPU execution strategy

### 8.1 Step anatomy

```
schedule() ──► prepare inputs (persistent buffers + diff) ──► copy to GPU
   ──► model forward (attention backend reads KV pool via block tables)
   ──► compute logits (last PP rank) ──► apply grammar mask ──► sample
   ──► copy sampled ids to CPU ──► build ModelRunnerOutput ──► update scheduler
```

### 8.2 Prefill vs decode kernels

- **Prefill** (many query tokens per request): compute-bound. Use a tiled FlashAttention-style
  kernel; support causal masking, GQA/MQA, optional sliding window, and paged KV for the
  cached prefix portion. Chunked prefill passes only the current chunk as queries.
- **Decode** (one query token per request): memory-bound. Use a paged attention kernel with
  split-KV across blocks and a reduction (FlashDecoding style), or a backend-provided paged
  decode kernel. Batch size is the primary utilization lever.
- **Mixed batches** (both at once) are supported by the unified scheduler; the backend must
  dispatch per request or use a unified kernel. This is where FA3's flexibility helps on
  Hopper; on Ada, a unified Triton kernel or two sub-batches may be necessary.

### 8.3 Reference path

A pure C++/Eigen attention implementation that materializes/unpacks paged KV into contiguous
tensors and computes scaled dot-product attention directly is the
**correctness oracle**. It is slow and never the default in production, but every optimized
backend is differentially tested against it.

### 8.4 CUDA graphs

- Capture **decode-only** graphs for a fixed set of padded batch sizes (e.g., 1,2,4,8,...
  up to `max_num_seqs`), sharing a single graph memory pool.
- Use a dummy padding request/pages so padded rows are valid and do not corrupt state.
- Dispatch by a batch descriptor; if no graph matches, fall back to eager.
- Capture after warmup; support `cuda_graph_max_batch_size = 0` to disable (for debugging and
  for backends that cannot capture).
- Piecewise capture is a later option if mixed batches must be graph-replayed. Attention,
  MoE dispatch, and any op with data-dependent shapes stay outside graphs unless the backend
  provides fixed-size buffers (DeepEP low-latency pattern).

### 8.5 Sampling and logits

Pipeline order (matching established semantics): capture logprobs (if requested) → cast to
float32 → allowed-token mask → bad-word mask → non-argmax-invariant logit processors →
penalties (repetition/presence/frequency) → grammar bitmask → temperature/top-k/top-p/greedy.
Greedy short-circuits when all requests are greedy. Sampling runs on GPU; only selected token
ids are copied to CPU. Sampling parameters are per-request and normalized by the frontend.

### 8.6 Determinism

- Greedy decoding with a fixed backend and batch composition must be deterministic.
- Batching can change floating-point reduction order and thus sampled tokens near ties; this
  is an inherent property. Correctness tests therefore compare against a reference **under the
  same batching conditions** where exactness is required, and use distributional/statistical
  tests for sampling.
- Document any nondeterminism (e.g., split-KV reductions, atomics) and provide a
  deterministic mode where feasible.

---

## 9. Distributed inference strategy

This entire section is **[requires target GPU]** for validation; the dev box has one GPU and
no NVLink. Interfaces and correctness tests are planned now.

### 9.1 Parallel config and groups

- `ParallelConfig`: `tp_size`, `pp_size`, `dp_size`, `ep_size`, `enable_ep`,
  `sequence_parallel`, plus derived group topology.
- `GroupCoordinator`: creates TP/PP/DP/EP process groups once; exposes collectives
  (`all_reduce`, `all_gather`, `reduce_scatter`, `send/recv`, all-to-all) as graph-capture-aware
  wrappers. NCCL default; gloo fallback for CPU/control.

### 9.2 Tensor parallelism

- Megatron-style column/row parallel linears, vocab-parallel embedding and LM head, with GQA
  KV-head replication when `tp_size > num_kv_heads`.
- Sharding applied at weight-load time; all-reduce after row-parallel projections; all-gather
  only when a consumer needs a full activation; final logits gathered to the output rank.
- Optimizations later: sequence parallelism (reduce-scatter/all-gather around norms),
  dual-batch/micro-batch overlap to hide collective latency, and collective+GEMM fusion.

### 9.3 Pipeline parallelism

- Layers split by stage; embedding on the first stage, LM head + sampler on the last.
- The engine keeps `pp_size` batches in flight via a batch queue (async execute), hiding
  bubbles GPipe-style.
- Pipeline-aware scheduler: account for outputs arriving later than scheduling; handle aborts
  mid-pipeline safely.

### 9.4 Expert parallelism / MoE (prioritized by §1.4)

Required by both Flash targets. CPU reference MoE lands in M0.5; the parallel GPU path lands
in the distributed milestone (M6/M7).

- All-to-all dispatch/combine per MoE layer (DeepEP); grouped/masked GEMM over experts
  (DeepGEMM); load balancing, and optionally redundant experts/EPLB.
- Decode path must be CUDA-graph friendly: use fixed-size dispatch buffers and a masked
  grouped GEMM (DeepEP low-latency + DeepGEMM masked pattern) or an AG-RS alternative.
- Shared/routed expert split and granularity are checkpoint-driven (E10).

### 9.5 Failure handling

- Rank failure detection; collective timeouts; engine-wide fatal error if a remote rank dies
  (recovery is by the orchestrator restarting the replica).
- Deterministic correctness tests: 1-GPU vs N-GPU greedy parity; per-layer output comparison.

---

## 10. Reliability and production considerations

### 10.1 Failure modes and responses

| Failure | Detection | Response |
|---|---|---|
| CUDA OOM in a step | Exception from forward | Catch, dump step metadata/stats, free/preempt the lowest-priority request and retry the step once; if still OOM, fail the engine and mark unhealthy (do not corrupt state) |
| Engine loop crash | Sentinel frame / watchdog | `EngineDeadError`; `/health` → 503; watchdog exits the server unless configured to keep alive; orchestrator restarts |
| Worker/rank crash | Process sentinel, collective timeout | Fail the engine replica; report; no partial output |
| Client disconnect | ASGI disconnect / cancellation | Abort request, free KV, remove from queues; idempotent abort |
| Stop string detected mid-stream | Detokenizer | Abort that request in the engine; emit final chunk |
| Request timeout | Frontend timer | Abort, return a timed-out error/finish reason |
| Engine input queue full | Bounded queue | 429/503; never block the HTTP event loop indefinitely |
| KV cache exhausted | `allocate_slots` returns `None` | Preempt (recompute) or wait; bounded by admission |
| Prefix cache reset while requests run | Admin call | Only reset when pool usage indicates no pinned blocks; otherwise reject/queue |
| Graceful shutdown | Signal | Stop accepting, drain in-flight (bounded grace), cancel remainder, free resources, exit |
| Host offload transfer failure (M7) | Event/timeout | Treat as miss; never serve partial KV; drop the save |
| Distributed rank loss | Collective/NCCL error | Fail replica; surface to orchestrator |

### 10.2 Resource bounds

Every resource is bounded and observable: waiting queue, running set, token budget, KV
blocks, host offload bytes, in-flight transfers, output queue depth per request (with
coalescing), and number of CUDA graphs. Overload returns 429/503 rather than growing memory
or latency without limit.

### 10.3 Request isolation

- Per-request state is keyed by a unique id; aborts are idempotent and safe under races.
- Tenant isolation in the prefix cache via `cache_salt` so one tenant cannot observe another's
  KV. (Encryption-at-rest for offloaded KV is a later concern.)
- Sampling parameters and max token limits are validated per request; a malformed request
  cannot affect others.

### 10.4 Graceful degradation

- If an optimized attention backend is unavailable for a shape/dtype, fall back to the
  reference path (slower but correct) rather than failing.
- If GPU memory is tight, reduce `max_num_seqs`/token budget dynamically (a documented policy)
  rather than OOM.

---

## 11. Observability strategy

### 11.1 Events and derived metrics

The engine emits monotonic, same-process timestamps for: `QUEUED`, `SCHEDULED` (first schedule,
ignoring preemptions), `PREEMPTED`, `FIRST_TOKEN`, `LAST_TOKEN`, `FINISHED{reason}`. The
frontend computes per-request: queue time, prefill time, decode time, TTFT, mean TPOT/ITL,
E2E, and token counts. Doing interval math in the frontend keeps the engine hot path clean
(vLLM lesson).

### 11.2 Prometheus metrics (minimum set)

- Gauges: `num_requests_running`, `num_requests_waiting`, `kv_cache_usage_ratio`,
  `prefix_cache_hit_rate`, `host_offload_usage` (M7), `gpu_utilization`,
  `gpu_memory_used_ratio`, `in_flight_transfers`.
- Counters: `prompt_tokens_total`, `generation_tokens_total`, `requests_total{reason}`,
  `num_preemptions_total`, `requests_rejected_total{reason}`, `prefix_cache_queries_total`,
  `prefix_cache_hits_total`.
- Histograms: `time_to_first_token_seconds`, `inter_token_latency_seconds`,
  `e2e_request_latency_seconds`, `request_queue_time_seconds`,
  `request_prefill_time_seconds`, `request_decode_time_seconds`,
  `request_prompt_tokens`, `request_generation_tokens`.
- Info: model/cache/parallel config labels.

### 11.3 Tracing and logging

- OpenTelemetry spans per request with `gen_ai.*` attributes; W3C `traceparent` propagation
  from the ingress. One span per request, with child spans for prefill/decode only when
  detailed tracing is enabled.
- Structured JSON logs with request id, stage, and correlation ids; no secrets, no prompt
  logging by default.

### 11.4 Admin/debug endpoints

`/health`, `/ready`, `/metrics`, `/server_info`, `/reset_prefix_cache`, `/start_profile`,
`/stop_profile`, and (later) `/sleep` / `/wake_up`. For security, admin endpoints are
disabled or token-gated by default.

---

## 12. Testing strategy

### 12.1 Test pyramid

1. **Unit tests** (fast, CPU-runnable where possible):
   - Hash/namespace construction and invariance under config changes.
   - Block pool: allocate/free/evict, intrusive free-list ordering, refcount pinning,
     null-block behavior, prefix hit detection, prefix cache reset safety.
   - Scheduler: token-budget accounting, chunked prefill splitting, admission bounds,
     preemption ordering, priority/aging, invariant assertions.
   - Sampling: top-k/top-p/temperature, penalties, grammar mask, greedy short-circuit.
   - Tokenizer/detokenizer: incremental UTF-8/byte-BPE boundary handling, stop strings.
   - Config validation and backend support-matrix rejection.
2. **Kernel correctness tests** (GPU; skip on CPU): every attention backend and quant
   kernel differentially compared against the C++/Eigen reference across shapes, dtypes,
   page sizes, GQA ratios, head sizes, sliding-window settings, and block-table layouts.
   Include CPU fallbacks so CI can run without a GPU (mini-sglang-experiment pattern).
3. **Model parity tests**: for each supported model, greedy output token ids equal the
   HuggingFace reference implementation on a fixed prompt suite across prompt/output lengths.
   For the §1.4 targets without a Transformers implementation, an independent NumPy reference
   is the oracle. Architecture-specific modules also get focused differential tests:
   - QK-norm and config-driven RoPE scaling at short and long context (D24, E16).
   - Gated-delta-rule recurrence and conv state across chunk boundaries and preemption
     (D19, E11).
   - MoE router/top-k gating and per-expert MLP against a reference (D20, E12).
   - MLA absorption equivalence to explicit attention, and sparse-indexer selection (D21,
     E13).
   - FP8/FP4 load-time dequantization against the BF16 reference (D22, E14).
   - MTP distribution equivalence and rollback accounting (D23, E15).
4. **Scheduler/integration tests**: bounded concurrency with many simultaneous requests;
   assert invariants, no starvation, no double-free, correct finish reasons.
5. **Server tests**: OpenAI request/response schema conformance, SSE framing and `[DONE]`,
   streaming vs non-streaming equivalence, request ids, usage accounting, abort behavior,
   timeout behavior, malformed request handling.
6. **Concurrency/stress tests**: long-running mixed workloads; assert bounded memory, no
   leaks (VRAM/RSS), no deadlocks, correct behavior under abort storms and overload.
7. **Reliability tests**: simulated OOM (reduce budget), engine crash and health reporting,
   graceful shutdown under load, client disconnect during prefill and during decode.
8. **Distributed tests** [requires target GPU]: NCCL collective correctness, TP/PP parity
   vs single-GPU, rank-failure handling.

### 12.2 Properties and invariants (assertion-tested)

- `num_computed_tokens ≤ num_tokens`; KV allocated for all computed tokens.
- `Σ scheduled tokens ≤ max_num_batched_tokens`; `len(running) ≤ max_num_seqs`.
- No request simultaneously in running and waiting; terminal requests are removed and freed
  exactly once.
- Prefix-hit tokens are never recomputed incorrectly; changing any namespace field prevents a
  false hit.
- Padded CUDA-graph rows and the null block never affect real outputs.
- Abort is idempotent; aborting a finished request is a no-op.

### 12.3 Race and leak detection

- Run tests under a debug allocator / `torch.cuda.memory_summary()` diffs to detect VRAM
  leaks; assert free blocks return to the initial count after a workload drains.
- Use sanitizers for any C++/CUDA extension; `compute-sanitizer` for kernels.
- Stress with randomized abort timing and request rates to shake out races.

### 12.4 Regression protection

- A **golden outputs** corpus (model + prompt + params → expected token ids) checked in, plus
  performance baselines recorded with the benchmark harness. Any change that regresses
  correctness or performance beyond a threshold fails CI (or requires an explicit,
  documented exception).

---

## 13. Benchmarking methodology

### 13.1 Metrics (definitions)

- **TTFT** = time of first streamed token − request arrival (client-side wall clock).
  Includes tokenization, queueing, prefill, and first decode.
- **TPOT/ITL** = (end − first_token) / (output_tokens − 1) for a request; report both the
  derived mean and the per-token ITL distribution.
- **E2E latency** = last token time − arrival; report mean and p50/p90/p99.
- **Throughput** = completed requests/s and total output tokens/s over the measurement window.
- **Goodput** = requests/s that satisfy a TTFT and TPOT SLO.
- **GPU metrics** = SM utilization, memory used, power, and (if available) achieved memory
  bandwidth, sampled via NVML/DCGM during the run.
- **Queue/prefill/decode decomposition** = derived from engine events.

### 13.2 Harness requirements

- An async client that streams SSE and timestamps each chunk with a monotonic clock.
- Configurable arrival process (closed-loop concurrency vs open-loop Poisson/gamma request
  rate), input/output length distributions, and dataset replay (sharegpt-like, random,
  prefix-repetition).
- Percentiles via `numpy.percentile` over per-request metrics; report `--metric-percentiles`
  (default p50/p90/p99).
- Peak concurrency and peak token-rate bins.
- Machine-readable results (JSON) with environment metadata (GPU, driver, CUDA, versions,
  config) for reproducibility and regression tracking.
- Separate **offline** benchmarks (in-process, for kernel/model tuning) from **online**
  serving benchmarks (through the HTTP API).

### 13.3 Baselines

- **HuggingFace Transformers** `generate` (batch 1 and static batching) as a latency and
  correctness baseline.
- **vLLM** on the identical machine/model/workload as the throughput/latency baseline. This
  is the primary competitive baseline and is installed/available in the environment.
- Optional: SGLang/LightLLM/lmdeploy if feasible on the target hardware.
- Our own previous milestone (regression baseline).

### 13.4 Reference workloads

- Single request, short prompt (128) + short output (128): latency sanity.
- Batch of 32, 1k input / 256 output: throughput sanity.
- Concurrency sweep (1,2,4,...,max) at fixed lengths: latency vs throughput curve.
- Long prompt (4k/8k/16k) with chunked prefill: TTFT behavior.
- Prefix-repetition (shared 2k/4k prefix): cache hit-rate and TTFT speedup.
- Realistic mixed trace (replayed): goodput under SLO.
- Overload: request rate above capacity; assert bounded latency and correct 429/503.
- Spec decode: acceptance rate, tokens/s, and distribution equivalence.

### 13.5 Targets

Targets are **measured baselines**, not absolute claims. For each target we record
`(hardware, model, dtype, workload, baseline value, our value, ratio)`. Initial engineering
targets (to be confirmed in E0/E1):

- M1: within 2× of vLLM tokens/s on the batch-32 workload (architecture correct, no kernels).
- M3: within 1.5× of vLLM on the concurrency sweep after Triton attention + CUDA graphs.
- M4: no unbounded latency growth under overload; p99 bounded by the configured queue+timeout.
- M5: spec-decode tokens/s ≥ 1.5× baseline on repetitive workloads, with distribution tests
  passing, else disabled by default.
- M6: TP scaling efficiency ≥ 0.7 per added GPU on NVLink-class hardware [requires target GPU].

### 13.6 Profiling

- Per-step CPU/GPU timing breakdown (schedule, input prep, H2D, forward, sample, D2H).
- `torch.profiler`/Nsight Systems for kernel-level analysis; Nsight Compute for hot kernels.
- Always compare against a baseline before/after; store traces as artifacts for the hot
  changes.

---

## 14. Staged implementation milestones

Each milestone is independently testable and ends with a demoable artifact. No milestone
should leave significant unvalidated code. Milestones M5+ are explicitly gated on the earlier
ones being correct and measured.

Legend: **[local]** runnable on the dev box (RTX 4080); **[target]** requires a multi-GPU or
Hopper+ rig.

**Priority revision (2026-09-18):** **M0.5** (target-model architecture support) is inserted
before M1. It delivers the backend ops, model-assembly registry, and hybrid cache interface
that M1 and later milestones depend on. M1+ are otherwise unchanged but now assume M0.5's
interfaces.

### M0 — Correct single-request inference (foundations)

- **Goal:** Load a model from a HuggingFace checkpoint and generate correct greedy output for
  a single request, end to end, with a reference attention path.
- **Components:** config objects; weight loader (streaming, meta-init, safetensors); tokenizer
  wrapper; model abstraction + Llama/Qwen dense implementations; naive paged-less attention
  (or simple KV cache); greedy sampler; offline `generate` API.
- **Key interfaces:** `Model.forward`, `KVCache` (even if simple), `Sampler`, `WeightLoader`.
- **Dependencies:** C++23 toolchain (GCC 13+/Clang 18+), CMake, Eigen, safetensors,
  nlohmann/json, GoogleTest; a tokenizer backend (`tokenizers-cpp`/sentencepiece); PyTorch +
  Transformers in a Python venv for the reference oracle only.
- **Correctness criteria:** exact token-id parity with HF `generate` for ≥3 models, ≥50
  prompts, temperature 0, across short/medium/long prompts.
- **Tests:** unit tests for weight sharding/fusion, RoPE, RMSNorm, attention vs reference;
  model parity suite.
- **Benchmarks:** baseline single-request latency (not optimized).
- **Performance:** correctness only; record numbers for regression tracking.
- **Risks:** subtle RoPE/attention/quantization bugs; tokenizer chat-template differences.
  Mitigate with differential tests at each layer.
- **Exit criteria:** parity tests green; a `generate(prompt) -> text` CLI works.

### M0.5 — Target-model architecture support [local correctness, target scale]

- **Goal:** Load and correctly execute the three priority models from §1.4 — Qwen3.8-27B
  (dense), Qwen3.8-Flash-Next (hybrid linear attention + MoE + MTP), and
  DeepSeek-V4.1-Flash (MLA + MoE + MTP, FP8-native) — establishing architecture-level
  correctness on tiny random checkpoints locally, with real-checkpoint parity on the target
  rig.
- **Components:**
  - **Config-driven model assembly.** A registry maps `architectures`/`model_type` to a
    composition of modules (attention variant, MLP/MoE variant, norm, positional encoding,
    optional MTP head) rather than per-arch model loops.
  - **Dense extensions.** QK-norm, optional attention bias, tied/untied LM head, GQA/MQA
    variants, config-driven RoPE scaling (D24).
  - **Hybrid linear attention (D19).** Chunked gated-delta-rule recurrence; conv/SSM state;
    `StateCache` group with per-request state and chunk-boundary carry.
  - **MoE (D20).** Router + top-k gating + per-expert MLP as a first-class op; CPU reference
    now, EP/grouped-GEMM later.
  - **MLA + sparse attention (D21).** Compressed latent KV layout, weight absorption, MLA
    attention op; sparse indexer behind a capability flag.
  - **Low-precision loading (D22).** FP8/BF16 (and FP4 where supported) safetensors parsing
    with explicit scales; dequantize-to-BF16 fallback on unsupported devices.
  - **MTP proposer (D23).** Multi-token heads exposed through the speculative-decoding
    `Proposer` interface.
  - **Tokenizer/chat templates.** Qwen and DeepSeek BPE tokenizers, chat templates, and
    reasoning/think-tag parsing.
  - **Hybrid KV manager.** Per-layer-group cache types (full / sliding / latent / recurrent)
    under one block/page accounting and prefix-cache policy.
- **Key interfaces:** `Backend::{Moe, MlaAttention, LinearAttention/StateUpdate}`;
  `ModelRegistry`; `CacheManager` with multiple group types; `Proposer`.
- **Dependencies:** M0 complete; per-architecture reference oracles (Transformers where
  available, else an independent NumPy reference); [target] GPU and the actual released
  configs/weights; a long-context parity harness.
- **Correctness criteria:** for each architecture, greedy token ids match the reference on
  tiny random checkpoints across short/medium/long contexts; MoE routing/gating matches;
  chunked linear-attention state matches a full-sequence recurrence across chunk boundaries;
  MLA absorption is numerically equivalent to explicit attention; MTP preserves the output
  distribution; RoPE scaling matches at context extremes.
- **Tests:** per-module differential tests (QK-norm, RoPE scaling, router/top-k, delta rule,
  MLA absorption, FP8 dequant); cache-state parity across chunk boundaries and preemption;
  per-architecture end-to-end tiny-checkpoint parity; tokenizer/chat-template round trips.
- **Benchmarks:** tiny-checkpoint correctness and step cost; [target] real-checkpoint TTFT/
  TPOT and memory; MLA/state cache footprint vs full KV.
- **Performance:** correctness-gated. Targets recorded, not asserted: state/MLA caches should
  reduce per-token KV traffic at long context; MoE decode is latency-sensitive.
- **Risks:** model availability and architecture assumptions (R15); hybrid-state correctness
  across chunked prefill and preemption; MoE all-to-all and graph-capture complexity; FP8/FP4
  kernel availability; local hardware cannot run full-scale models.
- **Exit criteria:** all three architectures load and pass tiny-checkpoint parity locally;
  hybrid/MLA/MoE cache semantics covered by invariant tests; real-checkpoint parity
  demonstrated on the target rig for at least Qwen3.8-27B and one Flash model.

### M1 — Continuous batching + serving + streaming [local]

- **Goal:** A streaming OpenAI-compatible server with a continuous-batching scheduler and a
  paged KV cache; multiple concurrent requests share GPU efficiently.
- **Components:** paged `BlockPool` + `KVCacheManager`; `Request`/state machine; unified
  token-budget `Scheduler` (FCFS); executor + `ModelRunner`; frontend/asyncio server with
  `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`; SSE streaming;
  incremental detokenizer; per-request output queues; abort on disconnect.
- **Key interfaces:** `SchedulerInterface`, `KVCacheManager`, `Executor`, `EngineClient`,
  `ServeEngine`.
- **Dependencies:** ZMQ or an equivalent IPC; FastAPI/uvicorn; msgpack.
- **Correctness criteria:** multi-request outputs match single-request outputs (with fixed
  batching conditions); OpenAI schema conformance; SSE framing; abort frees resources.
- **Tests:** scheduler invariants; block pool unit tests; server schema/streaming tests;
  an integration test with ≥8 concurrent requests.
- **Benchmarks:** concurrency sweep 1→max; throughput/latency vs HF static batching.
- **Performance:** expect to be behind vLLM; target within 2× on batch-32.
- **Risks:** scheduler/KV bugs under concurrency; IPC framing; cancellation races.
- **Exit criteria:** continuous batching demonstrably improves throughput over static
  batching; no leaks after a 1k-request run; clean shutdown.

### M2 — Chunked prefill, prefix caching, admission control [local]

- **Goal:** Long prompts do not stall decodes; repeated prefixes are reused; the system has a
  bounded, explicit overload policy.
- **Components:** chunked prefill in the scheduler; hash-chained prefix cache + eviction
  (LRU) + namespacing; preemption by recompute; bounded waiting queue; API-level and
  engine-level admission; request timeouts; 429/503 responses with `Retry-After`.
- **Correctness criteria:** chunked-prefill output equals non-chunked; prefix-cache hits
  produce identical output to a full recompute; namespace fields prevent false hits; overload
  returns errors rather than unbounded latency; preemption does not corrupt output.
- **Tests:** prefix-cache correctness/reset tests; namespace-invariance tests; scheduler
  preemption/admission tests (property-based where possible); overload stress test.
- **Benchmarks:** prefix-repetition workloads (hit rate, TTFT speedup); long-prompt TTFT
  with/without chunked prefill; overload latency curve.
- **Performance:** cache-hit TTFT ≤ 50% of miss TTFT for a 4k prefix (target).
- **Risks:** prefix-cache correctness (silent corruption), preemption thrash, admission
  tuning.
- **Exit criteria:** all invariants hold under randomized concurrency; overload steady state
  bounded; prefix-cache hit-rate benchmark recorded.

### M3 — Optimized attention, CUDA graphs, overlap scheduling [local]

- **Goal:** Close the performance gap with vLLM on the dev box.
- **Components:** Triton paged attention (prefill/decode/mixed, GQA, sliding window, optional
  split-KV); backend registry + selection matrix; FlashInfer/FlashAttention adapters;
  decode-only CUDA graphs with padded batch sizes; persistent input buffers; two-stream
  overlap scheduling with a kill-switch; per-step profiling.
- **Correctness criteria:** every backend passes differential tests vs the reference across
  the full shape matrix; graph replay outputs equal eager outputs; overlap does not change
  outputs or leak/free twice.
- **Tests:** kernel differential tests; CUDA-graph equivalence; stress with overlap on/off.
- **Benchmarks:** TTFT/TPOT/throughput concurrency sweep vs vLLM; per-step CPU/GPU breakdown;
  kernel microbenchmarks.
- **Performance:** within 1.5× of vLLM tokens/s on the sweep (target); ≥10% ITL improvement
  from graphs at batch ≥8.
- **Risks:** Triton kernel bugs on edge shapes; graph capture failures with certain
  backends/attention metadata; overlap-induced races.
- **Exit criteria:** benchmark results recorded with a clear before/after; fallback to
  reference verified.

### M4 — Production resource management and reliability [local]

- **Goal:** The system behaves predictably and observably under sustained/abnormal load.
- **Components:** full Prometheus metrics + event-based latency accounting; OTel tracing;
  structured logging; health/readiness/watchdog; engine death sentinel; OOM handling policy;
  graceful shutdown/drain; admin endpoints; dynamic capacity adjustment.
- **Correctness criteria:** metrics match a reference computation from events; OOM path frees
  correctly and either succeeds on retry or fails cleanly; shutdown drains in-flight work
  within the grace period; engine death is detected and surfaced as 503.
- **Tests:** metrics unit tests; fault-injection (OOM, crash, disconnect, timeout); soak test
  (10k+ requests) with leak checks; shutdown-under-load test.
- **Benchmarks:** sustained throughput/latency run with GPU utilization and memory curves;
  regression vs M3.
- **Risks:** metric overhead, watchdog false positives, subtle shutdown races.
- **Exit criteria:** bounded resources demonstrated; no leaks; dashboards/graphs produced
  from exported metrics.

### M5 — Advanced inference optimizations [local, partially target]

- **Goal:** Reduce memory and improve tokens/s with quantization, speculative decoding,
  structured output, and LoRA.
- **Components:** weight-only INT4 path (AWQ/GPTQ via a library kernel); optional FP8/KV quant
  [target]; n-gram spec decode with scheduler rollback, then a draft/EAGLE proposer behind a
  `Proposer` interface; `StructuredOutputManager` (FSM/bitmask + grammar cache);
  multi-LoRA loading/routing/unloading.
- **Correctness criteria:** quantized outputs within tolerance of BF16 on a held-out set;
  spec decode passes distribution-equivalence tests and never emits rejected tokens;
  grammar-constrained output always validates against the grammar; LoRA routing selects the
  right adapter and batched multi-LoRA matches single-adapter outputs.
- **Tests:** quant accuracy suite; spec-decode distribution/statistics tests; grammar
  conformance tests; LoRA differential tests.
- **Benchmarks:** tokens/s and memory with/without each feature; spec-decode acceptance rate;
  structured-decoding overhead.
- **Risks:** accuracy regressions, acceptance-rate surprises, grammar dead-ends, adapter
  correctness.
- **Exit criteria:** each feature is off by default unless it wins on the target workload,
  with recorded measurements.

### M6 — Multi-GPU execution [target]

- **Goal:** TP (then PP) inference with correct, measured scaling.
- **Components:** `ParallelConfig` + `GroupCoordinator`; TP-aware layers/weights; multiprocess
  executor (one worker per rank); PP stages + batch queue; NCCL integration; graph-capture of
  collectives; TP/PP correctness and scaling benchmarks.
- **Correctness criteria:** N-GPU greedy output equals 1-GPU within defined tolerance (exact
  for deterministic ops); NCCL collective tests; failure of a rank fails the replica cleanly.
- **Tests:** collective unit tests; layer-wise parity; end-to-end TP/PP parity; rank-failure
  test.
- **Benchmarks:** scaling efficiency and latency vs 1 GPU; TP vs PP tradeoff on the target
  topology.
- **Risks:** collective correctness in graphs, load imbalance, PP bubble tuning.
- **Exit criteria:** target scaling-efficiency threshold met; documented topology guidance.

### M7 — Cache tiering, MoE/EP, and disaggregation [target]

- **Goal:** Extend capacity and reuse across nodes; support MoE models.
- **Components:** host-DRAM KV offload behind a fail-safe `KVConnector` (async side-stream
  copies, pinning, commit-after-completion); optional SSD/remote tier; MoE + expert
  parallelism (all-to-all, grouped/masked GEMM); prefill/decode disaggregation with KV
  transfer; model routing metrics.
- **Correctness criteria:** offload hits produce identical outputs; transfer failure degrades
  to a miss; MoE output parity; disaggregated P/D output parity; no partial-KV reads.
- **Tests:** connector correctness with fault injection; tier eviction; MoE parity; P/D
  end-to-end.
- **Benchmarks:** TTFT/ITL with/without offload on long shared prefixes; MoE throughput; P/D
  goodput.
- **Risks:** transfer latency on the critical path, metadata consistency, multi-node
  operational complexity.
- **Exit criteria:** offload is enabled only when it wins on the target workload; documented
  deployment topology.

### M8 — Performance hardening and reproducibility [local + target]

- **Goal:** Lock in performance and regression protection.
- **Components:** finalized benchmark suite and result schema; CI performance gates; profile-
  guided tuning (block size, token budget, graph sizes, sampling fusion); documentation of
  bottlenecks and tuning guidance.
- **Exit criteria:** reproducible benchmark report; regression gates active; a
  production-readiness review checklist completed.

---

## 15. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Prefix-cache correctness bugs (silent wrong KV) | Med | High | Chained-hash namespace tests; differential tests vs no-cache; invariant assertions |
| R2 | Scheduler/KV bugs under concurrency (double-free, starvation) | Med | High | Property/invariant tests; stress with randomized aborts; single-writer design |
| R3 | Triton attention kernel edge-shape bugs | High | Med | Full shape matrix differential tests; reference fallback |
| R4 | CUDA-graph capture incompatibility / corruption | Med | Med | Decode-only capture; padding; kill-switch; eager fallback; equivalence tests |
| R5 | IPC complexity / engine death handling | Med | Med | Sentinel + watchdog; versioned framing; integration tests |
| R6 | Performance below vLLM (architecture overhead) | Med | High | Per-step profiling; overlap; persistent buffers; measure early (M1) |
| R7 | OOM under load destroys requests/engine | Med | High | Startup profiling + headroom; bounded admission; OOM policy; dynamic budget |
| R8 | Distributed correctness/scaling [target] | Med | High | Interface-first; NCCL tests; parity tests; stage TP before PP/EP |
| R9 | Quantization/spec-decode accuracy or distribution regressions | Med | Med | Statistical + conformance tests; off by default until measured |
| R10 | Scope creep (too many features at once) | High | High | Fundamental-vs-advanced split; milestone gates; feature flags |
| R11 | Dev box underpowered (16 GB, one GPU) hides/creates issues | High | Med | Tag claims [local]/[target]; cloud/target rig for distributed; small models locally |
| R12 | Dependency churn (FlashInfer/Triton/PyTorch APIs) | Med | Med | Pin versions; isolate behind adapters; CI matrix |
| R13 | Structured output FSM dead-ends/overhead | Low | Med | Grammar validation tests; timeout/dead-end handling; cache compiled grammars |
| R14 | Long-context numerical drift | Med | Med | Reference parity at long context; deterministic mode; document tolerances |
| R15 | Target-model architecture assumptions in §1.4 are wrong or checkpoints are unavailable | High | High | Checkpoint-driven "model spec" extraction (E10); config-driven, swappable modules; no hard-coded per-arch loops |
| R16 | Hybrid recurrent-state correctness across chunked prefill, preemption, and prefix reuse | Med | High | Full-sequence recurrence oracle; chunk-boundary and preemption parity tests; forbid state prefix reuse without checkpoints |
| R17 | MoE all-to-all, expert imbalance, and decode/graph-capture latency | Med | High | CPU reference first; fixed-size decode buffers + masked grouped GEMM; EPLB only after measurement [target] |
| R18 | FP8/FP4 loader/kernel availability and accuracy on target hardware | Med | Med | Explicit scales; dequantize-to-BF16 fallback; accuracy validation vs reference; capability gating |
| R19 | MTP acceptance or distribution regression | Med | Med | Distribution-equivalence tests; off by default until measured |
| R20 | Local hardware cannot run 27B/MoE at scale, so scale bugs surface late | High | Med | Architecture parity on tiny checkpoints locally; secure target rig before M0.5 scale exit; tag claims [local]/[target] |

---

## 16. Open questions and unresolved decisions

1. **Block size default on Ada.** 16 vs 32 vs 64: hit rate vs gather overhead. Decide in E1.
2. **Own Triton kernel vs FlashInfer default.** Is our Triton kernel within an acceptable
   margin? Decide in E3; if not, keep it as the reference optimized path and ship FlashInfer.
3. **Mixed-batch graph capture.** Can we capture mixed prefill+decode, or decode-only? E4.
4. **FP8 on `sm_89`.** Verify actual library support and accuracy before committing. E5.
5. **Unified vs reserved-decode budget.** Does a decode reserve materially improve p99 ITL?
   E3.
6. **Radix vs hash prefix cache.** Does radix materially improve hit rate on our traces? E2.
7. **Overlap scheduling at small models.** On a 4080 with small models, is Python overhead
   significant enough to justify overlap? E6.
8. **Host offload threshold.** What prefix length/hit rate makes offload a net win? E7.
9. **Multi-node transport.** RDMA vs TCP vs NVLink-only; depends on the target cluster. E8.
10. **Reference tokenizer semantics.** Which chat templates/parser behavior to support and how
    to avoid template drift across model releases.
11. **Exact target architectures.** The §1.4 feature lists are assumptions: which linear
    attention (gated delta rule vs gated attention vs Mamba2), MoE granularity and shared
    experts, MLA/absorption details, sparse indexer design, and MTP head count must be read
    from the released configs/weights (E10).
12. **Target hardware and parallelism for the priority models.** How to serve 27B dense and
    the two MoE targets: TP/PP/EP split, minimum GPU count and interconnect, and whether
    FP8/FP4 is required to fit.
13. **Hybrid cache semantics.** Whether recurrent state can be prefix-cached (checkpointed)
    and how it interacts with preemption and chunked prefill (E11).
14. **Quantization formats and scales.** Which FP8/FP4 recipes and scale granularities the
    checkpoints use, and the accuracy budget for dequantize-to-BF16.
15. **MTP availability.** Whether the released checkpoints include MTP heads, and the
    expected acceptance rate on target workloads (E15).

---

## 17. Experiments and prototypes required before committing

These are small, time-boxed spikes that inform decisions. They are **not** milestones and
should not ship.

- **E0 (baseline):** Install PyTorch/CUDA stack; run vLLM and HF on the 4080 with 1–3B models;
  record TTFT/TPOT/throughput and GPU utilization. Produces the benchmark baseline.
- **E1 (block size):** Microbenchmark paged attention gather at block sizes 8/16/32/64 with
  representative batch/lengths; measure hit rate on a prefix-repeat trace. Decide default.
- **E2 (prefix structure):** Prototype hash-block vs a minimal in-memory radix tree on the
  same trace; compare hit rate, CPU overhead, and memory. Decide cache structure.
- **E3 (attention backend):** Implement the pure-PyTorch reference and a Triton paged kernel;
  compare against FlashInfer/FlashAttention for prefill/decode shapes on Ada (speed +
  exactness). Decide default and support matrix.
- **E4 (CUDA graphs):** Prototype decode-only graph capture with padding; measure ITL gain and
  capture memory; test mixed-batch capture feasibility. Decide graph mode.
- **E5 (FP8/INT4):** Verify library support and accuracy for FP8 and weight-only INT4 on
  `sm_89`; measure memory and speed. Decide first quantization path.
- **E6 (overlap):** Prototype two-stream overlap and measure its effect on step time / ITL at
  small batch. Decide whether overlap is needed early.
- **E7 (offload):** Simulate host-DRAM offload of KV for a long shared prefix; measure
  transfer latency vs recompute cost. Decide the offload threshold and validity of the
  design.
- **E8 (distributed) [target]:** On a multi-GPU rig, prototype TP all-reduce placement and
  measure scaling/collective overhead; validate NCCL + graph capture. Decide TP-first scope.
- **E9 (structured output):** Prototype FSM/bitmask construction and per-step cost for JSON
  schemas; confirm the scheduler gating design.
- **E10 (target-model specs) [priority]:** Obtain and inspect `config.json`, `tokenizer.json`,
  and (if accessible) weights for the three §1.4 models. Produce a per-model "model spec"
  (layers, attention types, MoE config, MLA dims, RoPE scaling, MTP heads, quant format) and
  diff it against the assumptions. This experiment gates the M0.5 component list and must run
  first.
- **E11 (hybrid state):** Prototype the gated-delta-rule recurrence and carry state across
  chunk boundaries; compare against a full-sequence reference; test preemption/restart.
  Decide the `StateCache` interface.
- **E12 (MoE):** CPU reference router/top-k/expert MLP vs the checkpoint's expected routing;
  [target] spike a grouped/masked GEMM and all-to-all path (DeepGEMM/DeepEP). Decide the EP
  scope and decode buffer strategy.
- **E13 (MLA):** Prototype latent KV with weight absorption and verify equivalence to
  explicit per-head attention; decide layout and block size; evaluate the sparse indexer.
- **E14 (FP8/FP4):** Verify loader correctness with explicit scales and kernel availability/
  accuracy on target hardware; decide the default precision per model.
- **E15 (MTP):** Measure acceptance rate and verify distribution equivalence for the MTP
  proposer; decide whether it is enabled by default.
- **E16 (long context):** Validate config-driven RoPE scaling against the reference at
  32k/128k (and beyond where supported); decide the context ceiling per model.

Each experiment ends with a one-page written decision recorded as an update to the relevant
ADR in §4.

---

## 18. Sources and references

Primary and secondary sources used. Local source paths are given for reproducibility.

**Papers / conferences**
1. Kwon et al. "Efficient Memory Management for LLM Serving with PagedAttention." SOSP 2023.
   arXiv:2309.06180. https://arxiv.org/abs/2309.06180
2. Yu et al. "Orca: A Distributed Serving System for Transformer-Based Generative Models."
   OSDI 2022. (Continuous batching; iteration-level scheduling.)
3. Agrawal et al. "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve."
   arXiv:2403.02310. https://arxiv.org/abs/2403.02310 (Chunked prefill, stall-free scheduling.)
4. Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs."
   arXiv:2312.07104. https://arxiv.org/abs/2312.07104 (RadixAttention, compressed FSM.)
5. Zhong et al. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM
   Serving." OSDI 2024. arXiv:2401.09670. https://arxiv.org/abs/2401.09670
6. Qin et al. "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving."
   arXiv:2407.00079. https://arxiv.org/abs/2407.00079
7. vLLM Team. "vLLM V1: A Major Upgrade to vLLM's Core Architecture." 2025-01-27.
   https://blog.vllm.ai/2025/01/27/v1-alpha-release.html (Unified scheduler, zero-overhead
   prefix caching, piecewise CUDA graphs, process split.)
8. Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."
   arXiv:2307.08691.
9. Shah et al. "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low
   Precision." arXiv:2407.08608.
10. "LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference."
    arXiv:2510.09665.
11. Patel et al. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting."
    ISCA 2024. arXiv:2311.18677.
12. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." / EAGLE-3.
13. Leviathan et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
14. Qwen Team. "Qwen3 Technical Report" and the Qwen3-Next architecture notes
    (Gated DeltaNet + gated attention hybrid, ultra-sparse MoE, multi-token prediction).
    To be pinned to the exact model card at release.
15. DeepSeek-AI. "DeepSeek-V2: A Strong, Economical, and Efficient
    Mixture-of-Experts Language Model." arXiv:2405.04434 (Multi-head Latent Attention).
16. DeepSeek-AI. "DeepSeek-V3 Technical Report." arXiv:2412.19437 (MLA + DeepSeekMoE + MTP,
    FP8-native training and weights).
17. Yang et al. "Gated Linear Attention Transformers with Hardware-Efficient Training" and
    "Gated Delta Networks: Improving Mamba2 with Delta Rule" (chunked gated-delta recurrence).
18. Gu and Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces."
    arXiv:2312.00752 (recurrent state-cache reference).
19. Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated
    Mixture-of-Experts Layer." ICLR 2017 (routing fundamentals).
20. DeepSeek/Moonshot model cards and released `config.json` files for the §1.4 targets
    (authoritative architecture, RoPE scaling, quantization, and MTP-head definitions).

**Official docs / engineering posts**
- vLLM docs: design docs on prefix caching, hybrid KV cache manager, paged attention,
  multiprocessing, metrics; quantization feature docs. Local: `/home/gc/vllm/docs/`.
- SGLang docs, HiCache design. Local: `/home/gc/Mooncake/docs/source/design/hicache-design.md`.
- LMDeploy KV quantization docs. Local: `/home/gc/lmdeploy/docs/`.
- TensorRT-LLM, Hugging Face TGI documentation.

**Source trees inspected (local)** — see §2.1 table. Key files/lines are cited inline
throughout §2 (e.g., vLLM `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/block_pool.py`,
`vllm/v1/core/kv_cache_utils.py`, `vllm/v1/engine/core.py`, `vllm/v1/worker/gpu_model_runner.py`,
`vllm/v1/metrics/`, `vllm/entrypoints/openai/`; SGLang `srt/mem_cache/radix_cache.py`;
mini-sglang `python/minisgl/`; LightLLM `lightllm/router/`; lmdeploy `lmdeploy/pytorch/`;
DeepEP `deep_ep/buffers/`; DeepGEMM `deep_gemm/`; FlashMLA `flash_mla/`; LMCache
`lmcache/v1/`; `kv-cache/docs/design.md`; kvcached `kvcached/`; aibrix `pkg/`).

**Note on citations:** file:line references in §2 are from the checked-out commits listed in
§2.1 and may drift in upstream repositories; they are recorded to make claims auditable.

---

## 19. Critical review of this plan

Before finalizing, we reviewed the plan against the required questions.

- **Are major architectural decisions supported by evidence?** Yes. Each ADR cites either
  source code from a production system, a primary paper, or explicit measurements to be
  taken. The fundamental choices (unified scheduler, paged KV, hash prefix cache, process
  split, pluggable attention) are the convergent design of multiple independent systems.
- **Are important tradeoffs documented?** Yes — each ADR lists alternatives and why they were
  not chosen. Where systems disagree (hash vs radix, recompute vs offload, bounded vs
  unbounded admission), the disagreement is stated in §2.4 rather than hidden.
- **Are milestones independently testable?** Yes. M0 and M1 are end-to-end demoable; every
  later milestone has explicit correctness criteria, tests, benchmarks, and exit criteria.
- **Are correctness and performance measurable?** Yes. Correctness is defined against HF
  parity and differential kernel tests; performance has a defined harness, metric
  definitions, baselines (HF and vLLM), and engineering targets to be confirmed.
- **Are production failure modes considered?** Yes — §10 enumerates OOM, engine crash,
  disconnect, timeout, overload, KV exhaustion, offload failure, and distributed failure,
  each with a response.
- **Are we prematurely committing to uncertain details?** The genuinely uncertain choices
  (block size, own vs third-party kernel, FP8 support, mixed graphs, radix vs hash, offload
  threshold) are explicitly deferred to §17 experiments rather than hard-coded. Where we do
  commit early (scheduler representation, paged KV, process split), the evidence is strong and
  retrofitting would be expensive.

### 19.1 Revisions made after review

- **Target-model priority (2026-09-18):** promoted support for Qwen3.8-27B,
  Qwen3.8-Flash-Next, and DeepSeek-V4.1-Flash ahead of M1. Added §1.4 (target models and
  rationale), milestone **M0.5**, decisions **D19–D24**, risks **R15–R20**, open questions
  **11–15**, experiments **E10–E16**, and architecture-parity tests. The change is
  structural, not additive: hybrid recurrent state, MLA latent KV, and MoE must shape the
  backend ops and cache manager before serving/batching are designed.
- **Language revision (2026-09-18):** switched the implementation from Python+PyTorch to
  **C++23 + established third-party libraries** per explicit project constraints; rewrote D1,
  the reference-path wording (D6/§8.3), and M0 dependencies accordingly. The architecture is
  unchanged; only the realization language and library choices changed.
- Added explicit bounded-admission and overload semantics (D14) rather than inheriting
  vLLM's unbounded queue, because production requirements demand it.
- Added the reference attention path as a first-class requirement (D6/M0) so every
  optimization has an oracle.
- Added namespace isolation for the prefix cache as a correctness invariant (§5.3), since
  silent cross-tenant/model cache hits are a severe failure.
- Added the "dev box cannot validate distributed/FP8/MLA" constraint prominently and tagged
  claims accordingly, so the plan does not overstate local verifiability.
- Moved CUDA graphs, quantization, spec decode, structured output, and offload behind
  interfaces but out of the critical path, to avoid the vLLM-V0-style feature-coupling trap.
- Made the benchmark methodology precede implementation and define baselines and targets, per
  the "performance optimizations should not be accepted without measurement" requirement.

### 19.2 Known weaknesses / follow-ups

- Performance targets are provisional until E0 establishes real baselines on the 4080.
- The distributed plan is necessarily interface-heavy and unvalidated locally; a target rig
  must be secured before M6.
- Observability and reliability details will firm up during M4 when real failure traces exist.
- The tokenizer/chat-template compatibility matrix is under-specified and is tracked as an
  open question.

---

## 19a. M0 implementation record (2026-09-18)

M0 was implemented in C++23 on the dev box. This records what exists, the evidence gathered,
and the deliberate deviations from the pre-implementation plan.

### 19a.1 What was built

A C++23 library (`inferx_core`) and a CLI (`inferx_generate`) implementing a correct,
single-request, greedy decoder-only transformer:

- `ModelConfig` — HuggingFace `config.json` parsing (Llama/Qwen/Mistral-style fields,
  explicit or derived `head_dim`, GQA head counts, optional QKV bias).
- `SafeTensors` — reader for the safetensors format (single file and sharded directories),
  converting F32/F16/BF16/F64 tensors to float32.
- `Model` — embedding, RMSNorm, RoPE, grouped-query causal attention with a dense KV cache,
  SwiGLU MLP, final norm, and LM head (explicit or tied).
- `Generator` — greedy prefill + decode loop with EOS and sequence-length handling.
- `Tokenizer` — facade with a tokenizers-cpp (HuggingFace `tokenizers`) implementation.
- Reference primitives (`Linear`, `RmsNorm`, `SiLU`, `ApplyRotary`) as the correctness oracle.
- GoogleTest suites for ops, the tokenizer, and end-to-end model correctness.

### 19a.2 Third-party libraries used (and deferred)

- **Abseil** (vendored submodule): `Status`/`StatusOr`, `flat_hash_map`, spans, flags, strings.
- **Eigen 3.4** (FetchContent): dense CPU linear algebra. No hand-written GEMM.
- **nlohmann/json v3.11.3** (FetchContent): config and safetensors-header parsing.
- **tokenizers-cpp / HuggingFace `tokenizers`** (vendored submodule, Rust core): tokenization.
- **GoogleTest v1.15.2** (FetchContent): tests.
- **Deferred**: Folly (futures/JSON/executors), Boost.Beast (HTTP server), and FlashInfer
  (attention kernels) are vendored but not yet needed; they land in M1/M3.

Deliberate exception: there is **no actively maintained C++ safetensors library** upstream
(the old `bindings/cpp` no longer exists), so the small, well-specified format is read
directly. This is format plumbing, not a reinvented algorithm.

Build conflict resolved: tokenizers-cpp bundles sentencepiece, which bundles a second copy of
Abseil and collides with the top-level Abseil targets. InferX builds only the self-contained
HuggingFace JSON tokenizer binding (`src/huggingface_tokenizer.cc`) against the Rust
`tokenizers_c` core, avoiding the collision; SentencePiece support is deferred.

### 19a.3 Correctness evidence

- A tiny random Llama checkpoint is generated with a NumPy reference forward pass
  (`python/make_testdata.py`).
- The NumPy reference was cross-checked against **HuggingFace Transformers** (CPU):
  maximum absolute logit difference **1.155e-07**, greedy tokens identical.
- The C++ engine matches the HF-validated fixture:
  - full-prefill logits for every prompt position within `1e-3` absolute,
  - per-step decode logits (validating the KV cache path) within `1e-3`,
  - greedy tokens exactly: `[77, 5, 50, 97, 2]`,
  - primitive unit tests (Linear/RMSNorm/SiLU/RoPE) and a tokenizer round trip.
- `ctest`: 3/3 suites pass. Code is formatted with clang-format (Google style) and compiles
  with `-std=c++23` on GCC 13.3.

### 19a.4 How to build and run

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/tools/inferx_generate --model tests/testdata/tiny_llama \
    --prompt_tokens "1,5,10,20,30" --max_new_tokens 8
python/.venv/bin/python python/make_testdata.py   # regenerate + HF cross-check
```

### 19a.5 Known limitations and deviations

- The plan (D1) originally selected Python + PyTorch; per the revised project constraints the
  implementation is **C++23 + established libraries**, and D1/§8.3 were updated accordingly.
- M0 uses a **dense KV cache** and a naive Eigen attention path; paged KV, continuous
  batching, the HTTP server, CUDA graphs, and FlashInfer are explicitly later milestones.
- The CLI text path requires a tokenizer whose vocabulary matches the model; the tiny fixture
  uses raw token ids. SentencePiece tokenizers are not enabled.
- Larger/slower models are constrained by the 16 GB RTX 4080; M0 correctness is validated on
  a synthetic model so the check is fast and hermetic.

---

## 20. Appendix A — Global invariants

1. A request's KV is allocated for every computed token; no gaps except the null block in
   sliding-window/chunked-local regions.
2. `Σ scheduled_tokens ≤ max_num_batched_tokens`; `len(running) ≤ max_num_seqs`;
   `len(waiting) ≤ max_waiting_requests`.
3. Cached KV blocks are immutable once hashed; block IDs are never renumbered.
4. A block is evictable only at `ref_cnt == 0`; a cached block under an active request is
   pinned.
5. Every terminal request is freed exactly once; aborts are idempotent.
6. Prefix-cache hits are sound: a hit implies the full token prefix is present and was
   produced under identical namespace parameters.
7. Only full blocks are cached; the last token is always recomputed for logits.
8. Optimized backends are differentially equivalent to the reference on their supported
   shape matrix; unsupported shapes fall back.
9. Overload is bounded: queue depth, latency, and memory reach a steady state; excess is
   rejected with 429/503.
10. The engine hot loop never blocks on network I/O or tokenization.

## 21. Appendix B — Glossary

- **TTFT / TPOT / ITL / E2E**: time to first token / time per output token / inter-token
  latency / end-to-end request latency.
- **Continuous batching**: mixing requests at token granularity every step.
- **Chunked prefill**: splitting a long prompt across steps.
- **Paged KV**: fixed-size KV blocks addressed via a block table.
- **Prefix caching / RadixAttention**: reuse of KV for shared prefixes.
- **TP / PP / DP / EP**: tensor / pipeline / data / expert parallelism.
- **MLA**: multi-head latent attention (DeepSeek); compressed KV latent.
- **GQA / MQA**: grouped / multi-query attention.
- **CUDA graph**: recorded GPU launch sequence replayed with fixed addresses.
- **Spec decode**: draft-and-verify generation.
- **Goodput**: throughput meeting latency SLOs.
- **Null block**: sentinel block for positions not physically stored.
- **KVConnector**: interface for external KV storage tiers.


