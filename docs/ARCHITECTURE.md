# InferX — System Architecture

Status: **draft v0.1** — foundational design, pre-implementation.
Target: a C++20 LLM inference server with throughput/latency comparable to vLLM and SGLang.

---

## 1. Goals and non-goals

### Goals

| # | Goal | Measure |
|---|---|---|
| G1 | Match or beat vLLM/SGLang throughput on a single GPU for dense models | tokens/s at fixed p99 TBT |
| G2 | Low and *stable* inter-token latency | p99 TBT < 2× median under mixed load |
| G3 | No Python, no libtorch in the serving path | single static-ish binary, < 200 MB |
| G4 | CPU never the bottleneck | per-step CPU work < 30% of GPU step time at batch 256 |
| G5 | Tensor parallelism from day one | linear-ish scaling to 8 ranks, correct at TP=1 |
| G6 | Three attention/FFN families | dense GQA, MoE, MLA |

### Non-goals (v1)

- Training, fine-tuning, backward pass.
- Vision/multimodal encoders (interfaces should not *preclude* them; nothing is built).
- Pipeline parallelism, disaggregated prefill/decode across nodes. Single-node only.
- Speculative decoding in v1 — but the scheduler and sampler interfaces must not assume one token per sequence per step (see §11, T9).

### Target envelope

Development box: 1× RTX 4080 SUPER (sm_89 Ada, 16 GB, no NVLink), CUDA 12.0, GCC 13.3 / Clang 18, 32 cores, 31 GB host RAM.

This shapes v1 hard:

- **16 GB** means quantized weights are the default path, not an afterthought. A 7B at FP16 leaves ~1 GB for KV; at W4A16 it leaves ~11 GB. Quantization is a first-class part of the model layer, not a plugin.
- **sm_89** has FP8 tensor cores (e4m3/e5m2) but no TMA and no warp-specialized async pipelines — those are Hopper. Kernels target the SM80-style `cp.async` + `mma.sync` pipeline; CUTLASS SM89 FP8 GEMM uses the SM80 collective builders.
- **No second GPU on this box.** TP is designed in but cannot be perf-tested here. See §7.4 for how we keep it honest instead of building it blind.

---

## 2. The core thesis

Python inference engines pay a structural tax that C++ does not, and the architecture should be built to collect the refund rather than to imitate their shape.

1. **No GIL ⇒ no process-per-rank.** vLLM and SGLang run one *process* per TP rank and broadcast each step's batch metadata over shared memory or ZMQ, because Python threads cannot execute the model concurrently. We run **one process, one thread per rank**, sharing the scheduler's output by pointer. Per-step IPC serialization cost goes to zero, and there is no multi-process failure/restart choreography.
2. **No interpreter ⇒ the CPU can run far ahead.** The per-step CPU budget (schedule, build block tables, gather sampling params, launch) is where Python engines burn 1–3 ms and why they need CUDA graphs so desperately. Our budget is tens of microseconds, which buys back the option to *not* graph-capture some paths.
3. **Real ownership semantics.** KV blocks, request state, and weight buffers get RAII lifetimes and single-owner rules that are checkable at compile time, rather than refcount-by-convention.

Everything below follows from these three.

---

## 3. Component map

```
                          ┌─────────────────────────────────────────┐
   HTTP/SSE               │           I/O layer  (N threads)        │
   ──────────────────────▶│  Boost.Beast  ·  OpenAI-compatible API  │
                          │  request parse · SSE stream out         │
                          └───────────────┬─────────────────────────┘
                                          │  Request (owned)
                          ┌───────────────▼─────────────────────────┐
                          │      Tokenizer pool  (M threads)        │
                          │  encode · incremental detokenize        │
                          └───────────────┬─────────────────────────┘
                                          │  MPSC queue
        ┌─────────────────────────────────▼─────────────────────────┐
        │                  Scheduler thread  (exactly 1)            │
        │   waiting queue · running set · admission · preemption    │
        │   chunked prefill mixing · KV block allocation            │
        │   prefix-cache lookup (radix tree)                        │
        └───────────┬───────────────────────────────────▲───────────┘
                    │ BatchPlan (shared_ptr, read-only) │ StepResult
        ┌───────────▼───────────────────────────────────┴───────────┐
        │              Executor  (one thread per TP rank)           │
        │   ┌──────────┐  ┌──────────┐        ┌──────────┐          │
        │   │ rank 0   │  │ rank 1   │  ...   │ rank N-1 │          │
        │   │ stream   │  │ stream   │        │ stream   │          │
        │   └────┬─────┘  └────┬─────┘        └────┬─────┘          │
        │        └─────────────┴───── NCCL ────────┘                │
        └───────────┬───────────────────────────────────────────────┘
                    │
        ┌───────────▼───────────────────────────────────────────────┐
        │  Model layer:  LlamaLike · MoE · MLA   (torch-free)       │
        │  Kernel layer: FlashInfer attn · CUTLASS GEMM · custom    │
        │  Memory:       KV pool · caching device allocator         │
        └───────────────────────────────────────────────────────────┘
```

### 3.1 Component responsibilities

| Component | Owns | Never does |
|---|---|---|
| **I/O layer** | Sockets, HTTP framing, SSE encoding, backpressure | Touch GPU state, tokenize |
| **Tokenizer pool** | Vocab, BPE/SPM encode, incremental decode state per request | Block; must be pure CPU |
| **Scheduler** | All request lifecycle state, KV block table, prefix radix tree, all policy | Touch CUDA, allocate device memory, block on GPU |
| **Executor** | CUDA context/streams, weight buffers, CUDA graphs, forward pass | Make policy decisions, allocate KV blocks |
| **Model layer** | Layer composition, parallelism sharding, quantization schemes | Know about requests, batching, or scheduling |
| **Kernel layer** | Raw device kernels and their launch config | Anything above a single kernel |
| **KV pool** | The one giant device allocation, block free-list | Decide *who* gets blocks (that's scheduler) |

The line that matters most: **the scheduler is the only component with policy, and it never touches CUDA.** The executor is the only component that touches CUDA, and it has no policy. This makes the scheduler unit-testable on a machine with no GPU, which is the single highest-leverage testability decision in the design.

---

## 4. Data flow — the life of a request

```
 1. accept        I/O thread: parse JSON → ServingRequest {prompt, SamplingParams, stream?}
 2. tokenize      Tokenizer pool: prompt → vector<TokenId>. Reject if > max_model_len.
 3. enqueue       Push into scheduler's MPSC intake queue. I/O thread returns to epoll.
 4. admit         Scheduler: prefix-match against radix tree → (matched_blocks, new_tokens).
                  Reserve KV blocks for new_tokens. If insufficient → stay in waiting queue.
 5. plan          Scheduler builds BatchPlan for step N: a set of prefill chunks + decode slots,
                  a flat block-table tensor, position ids, and a sampling-param SoA.
 6. execute       Executor rank threads consume the same BatchPlan pointer, run forward,
                  all-reduce where required, produce logits for the last token of each seq.
 7. sample        On-GPU sampler: penalties → temperature → top-k/top-p/min-p → multinomial.
                  Output stays on device. A single async D2H copy is issued, not waited on.
 8. commit        Scheduler (step N+1): reads step N's token ids (now landed), appends them
                  to sequences, checks stop conditions, frees KV of finished sequences.
 9. detokenize    Finished/updated sequences → tokenizer pool → incremental UTF-8-safe text.
10. emit          I/O thread: SSE chunk or final JSON. On disconnect, cancel propagates back
                  to the scheduler, which frees KV at the next step boundary.
```

Step 8 is the crux: **results from step N are consumed at the start of step N+1**, never waited on inside step N. This is what keeps the GPU from ever idling on a D2H sync. See §5.2.

---

## 5. Execution model

### 5.1 Threading

| Thread | Count | Model |
|---|---|---|
| I/O | `min(8, ncpu/4)` | Boost.Asio `io_context`, C++20 coroutines (`co_await` on Beast awaitables) |
| Tokenizer | `ncpu/4` | Work-stealing pool, pure functions over per-request state |
| Scheduler | **1** | Run-to-completion loop, owns all state, **no locks on its own data** |
| Executor rank | `TP_size` | SPMD; pinned to NUMA-local cores of their GPU |
| Detokenizer | shares tokenizer pool | — |

Single-writer scheduler state is a deliberate constraint, not laziness. The scheduler mutates block tables, radix tree nodes, and request states thousands of times per step; making that lock-free-by-topology is far cheaper than making it lock-free-by-atomics. All cross-thread communication is through queues:

- I/O → Scheduler: `folly::MPMCQueue<RequestPtr>` (multi-producer).
- Scheduler → Executor: a single `BatchPlan` handoff + a rank barrier (no queue; see §5.3).
- Executor → Scheduler: `StepResult` written into a double-buffered slot.
- Scheduler → Detokenizer: `folly::MPMCQueue<OutputChunk>`.

### 5.2 The overlap pipeline

The decode loop must never contain a device synchronization. Structure:

```
 GPU:   [ step N-1 forward+sample ][ step N forward+sample ][ step N+1 ... ]
 CPU:   [ plan N ][ launch N ][ commit N-1 ][ plan N+1 ][ launch N+1 ] ...
                                     ▲
                                     └── reads token ids from step N-1's D2H copy,
                                         which completed while step N was launching
```

Concretely, the scheduler runs one step ahead of the results it consumes. It must therefore **speculate** on the token count when planning step N without knowing step N-1's output. That is safe because:

- Each running sequence produces exactly one token per step (or `k` with spec-decode — a compile-time-known bound), so KV growth is predictable.
- The only thing unknown is *whether a sequence finished*. We plan as if none finished; a sequence that hit EOS at step N-1 executes one wasted step N and is retired at commit. Waste is bounded by `finished_seqs_per_step / batch_size`, typically < 1%.

This is the same trick as SGLang's overlap scheduler, but without a separate process boundary to cross.

**Escape hatch:** the pipeline depth is a config knob (`overlap_depth ∈ {0, 1}`). Depth 0 (synchronous, sync after each step) must remain correct and is the reference for differential testing. Every scheduler bug will first be diagnosed by flipping this to 0.

### 5.3 Rank coordination (SPMD)

All rank threads execute **identical control flow**. Divergence deadlocks NCCL, so control flow may never depend on rank-local data.

```cpp
// Executor rank loop — sketch
void RankWorker::run() {
  cudaSetDevice(rank_);
  for (;;) {
    const BatchPlan* plan = barrier_.wait_for_plan();   // all ranks see the same pointer
    if (!plan) break;                                    // shutdown is also collective
    forward(*plan);                                      // may contain ncclAllReduce
    if (rank_ == 0) sample_and_copy_out(*plan);          // only rank 0 owns the sampler
    barrier_.arrive();
  }
}
```

Two rules make this tractable:

1. **The BatchPlan is immutable and shared by pointer.** Zero serialization, zero copies. It is built by the scheduler, published with a release store, and read by all ranks. This is the concrete payoff of thesis point 1.
2. **Only rank 0 samples.** Logits are all-gathered (or the LM head is replicated) so rank 0 has full vocab logits. Sampling on every rank with a shared seed also works and avoids a gather; we default to rank-0 sampling for determinism and revisit if the gather shows up in profiles.

The barrier is a custom sense-reversing barrier with a spin-then-yield policy, not `std::barrier` — we need the spin phase to keep launch latency in the microsecond range.

### 5.4 CUDA streams

Per rank: a **compute stream**, a **copy stream** (H2D of new token ids, D2H of sampled ids), and a **comm stream** for NCCL. Events, not syncs, order them. NCCL on a separate stream lets independent compute overlap with the all-reduce tail — worth ~3-5% at TP=4 and free to set up now.

---

## 6. Memory

### 6.1 The allocation model

One `cudaMalloc` at startup carves nearly all free VRAM into three arenas:

```
┌──────────────┬───────────────────────────────┬──────────────┐
│   Weights    │          KV block pool        │  Workspace   │
│  (static)    │        (paged, dynamic)       │ (activations)│
└──────────────┴───────────────────────────────┴──────────────┘
```

- **Weights**: allocated once from a bump allocator, never freed. Loaded via `mmap`'d safetensors + `cudaHostRegister` on the mapped range, then async H2D. Sharded per rank at load time, so each rank only ever materializes its own shard on device.
- **KV pool**: sized as `free_vram × gpu_util_frac − weights − workspace`. Divided into fixed-size blocks. Never freed to the driver.
- **Workspace**: a caching sub-allocator with stream-ordered semantics for transient activations. Peak is bounded by `max_num_batched_tokens`, which makes it computable ahead of time rather than discovered by OOM.

There are no `cudaMalloc`/`cudaFree` calls in the steady state. Any one of them would implicitly synchronize the device and blow the overlap pipeline.

### 6.2 Paged KV cache

Alignment for anything a kernel reads is **`kTensorAlignment = 128`** — the L1/L2 cache line, so a row start maps a warp's 128 B request onto exactly four 32 B sectors. Nothing on sm_89 asks for more; 256 B is the legacy texture-binding number. The hard floor is 16 B (widest vectorized load, and CUTLASS drops to slower tiles below it).

Two consequences worth stating, because both were wrong at first:

- **Alignment is a per-call parameter, not a property of the allocator.** `Allocator::Allocate(bytes, alignment)` — the requirement belongs to the caller: `kPageAlignment` for pinned memory destined for `cudaHostRegister`, `kTensorAlignment` for tensor data. `kTensorAlignment` is the *default*, supplied by a non-virtual one-argument overload rather than a default argument on the virtual (which would resolve from the static type and could silently diverge in an override).
- **Device base allocations are not padded.** `cudaMalloc` suballocates from large VA reservations and returns far coarser alignment than we would request — measured at **1 MiB** on this hardware — while applying its own size granularity. Rounding a `cudaMalloc` size up to any alignment of ours buys nothing. The CUDA allocator therefore *verifies* the returned alignment rather than controlling it; a driver that ever returned less would corrupt vectorized loads silently.
- **A sub-allocator's backing buffer must be at least as aligned as the granularity it hands out.** A weaker base forces `BumpArena`/`CachingAllocator` to skip bytes realigning, which shifts every offset and shrinks usable capacity. This is why host allocations use `kTensorAlignment` too rather than something cheaper — host-backed `Storage` holds real tensor data anyway.

Block size: **16 tokens** default, configurable. Layout per layer:

```
kv_pool[layer][block_id][k|v][block_size][num_kv_heads_local][head_dim]
```

`num_kv_heads_local = num_kv_heads / tp_size` for GQA — KV memory shrinks with TP. **This is not true for MLA** (§7.3), and the abstraction must not assume it.

The block table is a `[max_batch, max_blocks_per_seq]` int32 device tensor living in a **fixed, pre-allocated buffer** so CUDA-graph-captured decode steps can read it without re-capture. The scheduler writes updates into a host-pinned staging copy and issues a partial async H2D each step.

### 6.3 Prefix caching

A **radix tree** over token sequences, keyed by token id, with each node owning a run of KV blocks and a refcount.

Chosen over vLLM-style block-hashing because the shared-prefix workloads we care about (system prompts, few-shot, multi-turn chat, agent scratchpads) match at *token* granularity, and a radix tree finds the longest match in one descent without hashing every block boundary. The cost is a more complex structure with node splits on partial match — acceptable because it is single-writer (§5.1) and therefore lock-free.

Eviction: LRU over refcount-zero leaf nodes, triggered on allocation failure. Because the scheduler is the only mutator, eviction is a synchronous tree walk with no coordination.

### 6.4 KV quantization

FP8 (e4m3) KV cache is a **v1 feature, not a stretch goal**, because 16 GB forces it. Per-block scale factors, dequant fused into the attention kernel's K/V load. Doubles effective context capacity for ~0.1% accuracy cost on most models.

---

## 7. Model and parallelism layer

### 7.1 Shape

No dynamic graph, no autograd, no operator dispatcher. A model is a struct of layers; `forward` is a straight-line function over tensors.

Tensors come in **two layers**, and which one a piece of code uses is a real design decision, not a stylistic one:

```
Tensor (8 B handle) ──▶ TensorImpl (refcounted) ──▶ Storage (refcounted) ──▶ bytes
                                                         ▲
TensorView (80 B POD, non-owning) ───────────────────────┘
```

**`Tensor`** is a lightweight owning handle, PyTorch-style: one pointer, copied with a single atomic increment, never copying data. Slice, reshape, and bitcast produce new `TensorImpl`s over the *same* `Storage`, so views are cheap and the bytes live exactly as long as the last handle. `Storage` is either allocated (freed through an `Allocator` on destruction) or **borrowed** — pointing at memory owned elsewhere and freeing nothing, which is the normal case for weights, since loading a 7B model must not mean 400 separate `cudaMalloc`s.

**`TensorView`** is a trivially-copyable non-owning view. It exists because it must: `__global__` parameters have to be trivially copyable, so a handle containing an `IntrusiveRefCntPtr` can never be passed to a kernel. PyTorch has the same split (`at::Tensor` vs `TensorAccessor`).

Trivial copyability is necessary but **not sufficient**, and the difference is easy to miss because it type-checks either way. Copyability is what gets a view *into* a kernel; reading one once it arrives additionally requires every accessor the kernel touches to be callable from device code. Unannotated inline members are host-only to nvcc, and these are not `constexpr`, so `--expt-relaxed-constexpr` does not reach them either — a kernel could receive a view by value and then do nothing with it. The accessors a kernel needs therefore carry `INFERX_HOST_DEVICE` (`__host__ __device__` under `__CUDACC__`, nothing otherwise); the ones it cannot use deliberately do not. Which side a member lands on follows from what its body can reach:

| | Members | Why |
|---|---|---|
| **Device-callable** | `Data`, `Bytes`, `DataAs`, `GetDataType`, `Rank`, `Dim`, `Numel`, `NBytes`, `IsDefined`, `IsEmpty`, `Device`, `IsCpu`, `IsCuda`, default ctor | Pointer arithmetic, plus the `constexpr` dtype and `DeviceId` helpers that `--expt-relaxed-constexpr` makes reachable |
| **Host-only** | `Dims`, `GetShape`, `ToString`, and everything returning `StatusOr` — `Create`, `Slice`, `Reshape`, `Bitcast` | `absl::Span` is third-party and its device-callability is not ours to control; `GetShape` materializes a `Shape` and can allocate; `StatusOr` is a host error idiom |

So a kernel indexes with `Rank()`/`Dim()`/`Numel()` and reads through `Data()`, and anything it cannot express that way belongs on the host side of the launch — a kernel wanting a subrange gets a view sliced before the launch, not a `Slice()` call inside it. The annotation is a promise the compiler checks: adding it to a member that calls a host-only function is an nvcc error, not a silent host-side fallback. The claim is kept honest by a launch rather than an assertion — `tests/kernel/smoke_test.cc` passes a rank-2 view to a real kernel and checks the arithmetic, so a regression that made the accessors host-only again would fail the build rather than the reasoning.

Refcounting comes from LLVM's `IntrusiveRefCntPtr`, vendored as `include/inferx/core/intrusive_ref_cnt_ptr.h` (Apache-2.0 WITH LLVM-exception, reparented into `namespace inferx`). Both `Storage` and `TensorImpl` derive from **`ThreadSafeRefCountedBase`**, never the non-atomic `RefCountedBase` — owning handles cross the scheduler/executor boundary, so the plain base would be a silent data race. Static assertions in the tests pin that choice. The CRTP base deletes through `Derived*`, so neither type carries a vtable: `TensorImpl` is 104 B, `Storage` 40 B, `Tensor` and `StoragePtr` 8 B each.

**Why both layers carry their own refcount.** `Storage` and `TensorImpl` are refcounted independently because they count *different* sharing relationships, which only shows up once you separate the two operations that mint a new tensor handle. *Copying* a `Tensor` (`Tensor b = a`) increments only `TensorImpl`'s count — the `IntrusiveRefCntPtr` copy retains the impl, not the storage — so two handles share one impl while that impl's `Storage` count stays at 1. *Slicing, reshaping, or bitcasting* (`a.Slice(...)`) instead builds a **new** `TensorImpl` over the **same** `Storage`, so many impls share one storage while each impl's own count is 1. The two counts therefore move independently — copy raises the impl count, view raises the storage count — and folding one into the other loses that distinction: merging `Storage` into `TensorImpl` double-frees shared bytes, merging `TensorImpl` into `Storage` forces every view to copy its bytes. `Storage`'s refcount is load-bearing: it is the unit of byte-level sharing.

`TensorImpl`'s refcount is the one worth questioning, since `TensorImpl` is immutable — no setters, all fields set at construction — so its refcount buys no copy-on-write, only `sizeof(Tensor) == sizeof(void*)` (a single-pointer handle that is cheap to move, return, and store in containers) and a `Tensor` copy costing one atomic rather than copying inline metadata. The legitimate alternative is to fold `(offset, dtype, shape)` into `Tensor` by value and keep only `Storage` refcounted — viable *because* the impl is immutable, at the cost of growing `Tensor` from 8 B to ~96 B and losing the single-pointer-handle invariant. That trade is declined (T21): the hot path already runs on `TensorView`, so the `Tensor`/`TensorImpl` layer is the ownership layer where cheap handles and cheap copies pay off, and where the one-per-tensor `TensorImpl` allocation is paid at admission rather than per step.

**`Shape`** is generic and imposes **no maximum rank**. It is backed by `absl::InlinedVector<int64_t, Shape::kDefaultRank>` with `kDefaultRank = 8`, where that constant is *inline capacity only* — a performance property, not a semantic limit. Any shape is representable; unusual ones simply allocate. A future model with an odd weight layout needs no change here.

The engine's real rank limit lives on **`TensorView::kMaxRank`**, because that is where it is genuinely forced: `TensorView` stores its extents as a POD `std::array` so it stays trivially copyable, which is required of a `__global__` parameter. Putting the limit there makes it a consequence of a hard constraint rather than an invented cap, and `ValidateTensorLayout` enforces it once for every tensor factory. The two constants are tied (`kMaxRank == kDefaultRank`) so that any shape a tensor accepts is also allocation-free on the host.

Because `InlinedVector` has a user-provided copy constructor and destructor, `Shape` is not trivially copyable — hence the POD extents in `TensorView`, with `shape()` materializing a `Shape` on demand. Sizes today: `Tensor` 8 B, `Shape` 72 B, `TensorView` 80 B, `TensorImpl` 104 B, `Storage` 40 B.

The rule: **ownership is established once — at weight load or request admission — as a `Tensor`; everything downstream passes `TensorView`.** That keeps atomic refcount traffic and per-tensor heap allocation off the decode path, where neither is affordable, while still getting real lifetime safety everywhere lifetime is actually in question.

Deliberately absent from `TensorImpl`, relative to PyTorch: no strides (contiguous row-major only, §7.1 rationale below), no autograd metadata, no version counter, no dispatch key set, no weak references — the ownership graph is acyclic by construction. Those omissions are what keep the impl small and its destructor trivial.

Layers are compile-time-composed where it pays (attention variant, quant scheme as template params) and virtual where it does not (the top-level `Model` interface, loaded by name). Avoid the trap of templating the entire model on 6 axes — compile time explodes and the binary gains nothing below the kernel boundary.

### 7.2 Tensor parallelism

Standard Megatron sharding:

| Layer | Shard | Comm |
|---|---|---|
| QKV proj | column | none |
| Attention | by head | none |
| O proj | row | all-reduce |
| Gate/Up proj | column | none |
| Down proj | row | all-reduce |
| Embedding | vocab | all-reduce |
| LM head | vocab | all-gather (for rank-0 sampling) |

Two all-reduces per transformer layer. Comms go through a `Communicator` interface, not raw NCCL calls (§7.4).

### 7.3 The three model families

**Dense GQA (Llama/Qwen/Mistral)** — the baseline. RoPE, RMSNorm, SwiGLU. Everything else is measured against this path.

**MoE (Mixtral/Qwen-MoE/DeepSeek-MoE)** — the FFN becomes router → top-k select → grouped GEMM. Two parallelism choices:
- *TP over experts' weights*: every rank holds a slice of every expert. Simple, reuses the dense comm pattern, but every rank does work for every activated expert.
- *Expert parallelism (EP)*: each rank owns whole experts; tokens are all-to-all'd to their expert's rank and back.

v1 implements **TP-over-experts** (fits the existing comm pattern, no all-to-all). The `FfnLayer` interface is designed so EP can be added as a second implementation without touching the attention path or scheduler. On 16 GB, MoE is only reachable at int4 for small models — it will be built and correctness-tested here, throughput-tuned elsewhere.

**MLA (DeepSeek-V2/V3)** — the important one for the KV abstraction. MLA caches a *compressed latent* per token per layer (plus a small decoupled RoPE key), not per-head K and V. Two consequences that must be baked in now:

1. The KV pool layout `[k|v][heads][head_dim]` **does not apply**. The KV manager must be parameterized by a `KvLayout` describing bytes-per-token-per-layer, with the per-head split being one instantiation and the MLA latent another. Getting this wrong means rewriting the block allocator later.
2. **The MLA latent is replicated across TP ranks, not sharded.** KV memory therefore does *not* shrink with TP for MLA. Any capacity-planning code that assumes `kv_bytes / tp_size` is wrong for MLA. This must be a property queried from the model, never computed by the scheduler.

### 7.4 Making blind TP honest

TP cannot be perf-tested on this box. Three mitigations, all cheap now and expensive to retrofit:

1. **`Communicator` is an interface.** Backends: `NcclComm` (real), `SingleRankComm` (no-op, always exercised at TP=1), and `HostSimComm` — a CPU shared-memory backend that runs N rank threads on *one* GPU or even on CPU tensors. `HostSimComm` lets us correctness-test all sharding logic and SPMD control flow here, today, at TP=4.
2. **Sharded weight loading is validated by reconstruction.** A test loads a model at TP=4 into host buffers and asserts the concatenated shards equal the TP=1 weights bit-for-bit. This catches ~80% of TP bugs without a second GPU.
3. **Numerical differential test.** TP=1 vs TP=4-under-`HostSimComm` logits must match within FP tolerance for a fixed prompt.

Real NCCL perf work is explicitly deferred until multi-GPU hardware exists, and the doc should say so rather than pretend otherwise.

---

## 8. Scheduling policy

### 8.1 Continuous batching with chunked prefill

Every step builds a **mixed batch**: some decode slots (1 token each) plus prefill chunks capped by a global `max_num_batched_tokens` (default 8192).

Chunking prefill is what keeps p99 TBT flat. Without it, a 32k-token prefill monopolizes a step and every decoding sequence stalls for hundreds of milliseconds. The cost is that a chunked prefill is slightly less GEMM-efficient than a monolithic one, and each chunk re-reads prior KV. We take that trade because G2 (stable TBT) is a stated goal and raw prefill throughput is not.

Priority within a step: **decode first, then prefill fills the remaining token budget.** This bounds TBT by construction.

### 8.2 Preemption

When KV allocation fails for a running sequence:

- **Recompute** (default): drop the sequence's KV, return it to the waiting queue head. Its prefix is likely still in the radix tree, so re-admission is cheap. Wastes compute, frees memory instantly, no host bandwidth.
- **Swap to host**: D2H the blocks. Preserves work but costs PCIe bandwidth on a contended bus and adds a copy the overlap pipeline must hide.

Default to recompute; swap exists behind a flag for long-context workloads where recompute cost dominates. With prefix caching enabled, recompute is nearly always the right call — this is a change from vLLM's original defaults and is worth measuring rather than assuming.

### 8.3 Admission and fairness

FCFS with a running-batch cap. Explicitly *not* doing shortest-job-first or priority classes in v1 — they interact badly with prefix caching (reordering destroys cache locality) and the interface should stay simple until we have real traces.

---

## 9. Kernel layer and third-party dependencies

| Concern | Choice | Notes |
|---|---|---|
| Attention (prefill, ragged) | **FlashInfer** core templates | Torch-free headers under `include/flashinfer/attention/` |
| Attention (decode, paged) | **FlashInfer** | Needs fixed-shape wrappers for CUDA graph capture |
| MLA attention | **FlashInfer** MLA wrapper | Newer FlashInfer only — drives the CUDA upgrade (§12, R1) |
| GEMM (FP16/BF16) | **cuBLASLt** | Heuristic cache keyed on shape, warmed at startup |
| GEMM (**FP8 e4m3 W8A8**) | **cuBLASLt** | Also cuBLASLt, which was not the original plan — see §15 Q2. Native e4m3 on sm_89, scales bound per call as device pointers. Measured at ~2× FP16 |
| GEMM (W4A16, grouped/MoE) | **CUTLASS** | Still no alternative for *mixed-input* and grouped GEMM. Deferred to M8/M9, since FP8 did not need it |
| RMSNorm, RoPE, SiLU-mul, quant/dequant, sampling, block copy | **Custom** | Small, fusable, and the place where a torch-free design wins |
| Collectives | **NCCL** | Behind `Communicator` (§7.4) |
| HTTP/SSE | **Boost.Beast** | With Asio + C++20 coroutines |
| Containers, sync, status | **Abseil** | `flat_hash_map`, `InlinedVector`, `Status`/`StatusOr` |
| Host allocation | **mimalloc** | Backs `HostAllocatorImpl` via `mi_malloc_aligned`. `MI_OVERRIDE` off, so it never replaces global `malloc` and stays out of ASan's way |
| Concurrent queues | **Folly** (narrow) | `MPMCQueue`, `ProducerConsumerQueue` only |
| Tokenizer | **HF `tokenizers`** via C FFI | Rust static lib; the only sane way to match HF behavior exactly |
| JSON | **simdjson** (parse) + custom writer | Parsing is on the request hot path |
| Logging/metrics | **spdlog** + Prometheus text endpoint | — |

### A note on Folly

Folly was named as an approved dependency, and we are using it — but scoped to two queue types. Building all of Folly pulls glog, gflags, double-conversion, libevent, fizz-adjacent bits, and a large fraction of Boost, for a multi-minute build and a meaningful binary size cost. The recommendation is to consume Folly as a submodule but build only the target providing the lock-free queues, and to revisit vendoring `moodycamel::ConcurrentQueue` (single header) if the build cost becomes a development-velocity problem. Flagging this now because it is much easier to keep the dependency narrow than to narrow it later.

### A note on FlashInfer

FlashInfer's *wrapper* layer is increasingly torch- and JIT-coupled; its *kernel template* layer is not. We use the templates directly and write our own AOT-compiled wrappers with our own `Tensor` type. This means pinning a FlashInfer commit and accepting that upgrades are manual merges rather than version bumps. That is the correct trade for a torch-free system, but it is real ongoing cost and should be budgeted.

---

## 10. C++20 usage

What we actually use, and where:

- **Concepts** — constrain kernel-dispatch and layer templates. Error messages at these boundaries are otherwise unreadable.
- **Ranges** — scheduler-side batch construction. Not in kernel launch paths (views can obstruct vectorization).
- **Coroutines** — **I/O layer only.** The engine core uses an explicit state machine. Coroutines in the per-step decode path mean heap allocations and frame indirection at exactly the point where we need predictable microsecond behavior; an explicit `RequestState` enum is debuggable in a core dump and coroutine frames are not.
- **`std::span`, designated initializers, `constexpr`** — everywhere, freely.
- **`std::jthread` + `stop_token`** — all long-lived threads, for correct shutdown.
- **`std::format`** — available in GCC 13; use it over `fmt` where possible to shed a dependency.
- **Modules** — **no.** GCC 13's implementation is not production-ready and CMake 3.28 support is early. Headers, with strict include discipline. Revisit at GCC 15+.
- **`std::expected`** — C++23, unavailable in GCC 13. Use `absl::StatusOr` uniformly rather than mixing three error idioms.

---

## 11. Key trade-offs — decision register

Each entry is a decision we should be able to revisit deliberately.

| # | Decision | Alternative | Why | Revisit when |
|---|---|---|---|---|
| T1 | Torch-free | libtorch | Binary size, control of the decode loop, no allocator/stream conflicts. Costs ~3-4 wks on quantized GEMM + loading. | Never for the core; offline tooling may use it |
| T2 | One process, thread-per-rank | Process-per-rank | No GIL ⇒ no reason to pay IPC. Zero-copy BatchPlan sharing. Risk: one rank crash kills all. | If per-rank fault isolation becomes a production requirement |
| T3 | Single-writer scheduler | Sharded/locked scheduler | Lock-free by topology beats lock-free by atomics for thousands of small mutations per step | If one core cannot keep up at batch > 512 |
| T4 | Radix-tree prefix cache | Block-hash (vLLM) | Token-granular longest-match, no per-block hashing. More complex; safe because single-writer. | If tree walk shows in profiles |
| T5 | Chunked prefill, decode-priority | Prefill-priority / separate phases | Bounds p99 TBT by construction (G2). Costs some prefill throughput. | With 2+ GPUs, evaluate disaggregated P/D |
| T6 | Overlap depth 1 | Synchronous stepping | Hides all CPU work behind GPU. Costs < 1% wasted steps on finished seqs. | Depth 0 stays as the correctness reference |
| T7 | Recompute preemption | Swap to host | Prefix cache makes re-admission cheap; no PCIe contention | Long-context workloads where recompute dominates |
| T8 | Rank-0 sampling | All-rank sampling w/ shared seed | Determinism, simpler RNG. Costs a logits all-gather. | If the gather appears in profiles |
| T9 | No spec-decode in v1, but interfaces admit it | Assume 1 token/seq/step | Retrofitting variable tokens-per-step into the scheduler and KV allocator is a rewrite; admitting it costs a bound parameter | v1.1 |
| T10 | Block size 16 | 32 or 64 | Less internal fragmentation, finer prefix sharing; larger block tables and slightly worse kernel efficiency | Benchmark once attention is real |
| T11 | KV layout parameterized by model | Fixed `[k|v][heads][dim]` | MLA's latent cache does not fit the fixed layout; and MLA KV does not shard with TP | Immediately — this is a v1 constraint, not future-proofing |
| T12 | TP-over-experts for MoE | Expert parallelism | Reuses the dense comm pattern, no all-to-all | When multi-GPU hardware exists and MoE is a real target |
| T13 | Headers, not modules | C++20 modules | GCC 13 modules are not production-ready | GCC 15+ |
| T14 | Explicit state machine in engine core | Coroutines throughout | Predictable, allocation-free, debuggable in core dumps | Never for the decode path |
| T15 | Two tensor layers: owning `Tensor` + POD `TensorView` | One type for both | A refcounted handle cannot be a `__global__` parameter, so the POD is mandatory; a single owning type would also put atomics and heap allocation on the decode path | If profiling shows the split costs more in complexity than it saves |
| T16 | Contiguous-only, no strides in `TensorImpl` | PyTorch-style strided views | Every kernel would otherwise handle the strided case or silently assume it away. Slicing is dim-0 only; anything else is an explicit copy | If a real layer needs a transposed view badly enough to justify it |
| T17 | `Storage` may be borrowed | Always owning | Weights are one arena allocation with hundreds of tensors over it, not hundreds of allocations | Never — this is the weight-loading path |
| T18 | `Shape` on `InlinedVector`; `TensorView` keeps POD extents | One shape type everywhere | Container ergonomics for host code without making `TensorView` non-trivially-copyable (illegal as a kernel parameter) or putting an allocating copy on the step path | If `TensorView::shape()` materialization ever shows in profiles |
| T19 | `Shape` has **no** maximum rank; the limit is `TensorView::kMaxRank` | A cap on `Shape` itself | The container has no inherent limit once it is a vector, so a cap there would be invented. The POD kernel-boundary type has a real one. Keeps `Shape` generic for model layouts we have not met | Never for `Shape`; `kMaxRank` grows if a real layout needs it |
| T20 | `kDefaultRank = 8` | 6, sized to today's layouts | Headroom for unknown future layouts, at 16 B on `TensorView` (64→80). Costs nothing measurable now and avoids a migration later | If kernel-launch overhead ever shows parameter size mattering |
| T21 | Both `Storage` and `TensorImpl` refcounted | Flatten `TensorImpl`'s immutable `(offset,dtype,shape)` into `Tensor` by value; keep only `Storage` refcounted | The two refcounts count different fan-outs — copy raises the impl count, slicing raises the storage count — so neither is redundant; `Storage`'s count is the byte-sharing unit and `TensorImpl`'s buys the single-pointer handle. Hot path runs on `TensorView`, so the per-tensor `TensorImpl` allocation is off the step path | If `TensorImpl` allocation or handle indirection dominates profiles *and* `sizeof(Tensor)==sizeof(void*)` stops mattering |

---

## 12. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | **CUDA 12.0 is too old.** CUTLASS 4.x (FP8 + grouped GEMM) and the FlashInfer MLA wrapper both need a current toolkit, and 12.0–12.3 cannot compile against libstdc++ 13 at all. | High | **Resolved in M0:** toolkit floor set to CUDA 13.0, enforced at configure time. `scripts/install-cuda.sh` performs the upgrade. |
| R2 | TP is designed but untestable for performance here | High | §7.4 — `HostSimComm`, shard-reconstruction tests, numerical differential tests. Accept that perf tuning is deferred. |
| R3 | Torch-free quantized GEMM (FP8, W4A16, MoE grouped) is the largest single unknown | ~~High~~ **Retired for FP8 at m ≥ 128; Medium for W4A16** | **FP8 e4m3 W8A8 lands at ~2× FP16** on the 4080 SUPER: 214–216 TFLOP/s vs 107–111, the Ada FP8 tensor-core peak, across all four 7B GEMMs. Reproducible to two decimals between independent runs for every shape at m ≥ 128. **Decode shapes (m ≤ 32) are not yet measurable here** — with unlocked clocks the same shape has read 4.03× and 5.77× on consecutive runs, and `gate_up` at m=32 has read 2.04× and 0.32×. FP8 halves the weight read so decode should benefit at least as much, but that is reasoning, not measurement, until the clocks are locked. Torch-free quantized GEMM is demonstrated for prefill; W4A16 remains open and is the part that still needs CUTLASS. |
| R4 | 16 GB constrains what can be validated end-to-end | Medium | Target 7B-class at W4A16 for the main loop; MoE/MLA validated for correctness at toy sizes |
| R5 | FlashInfer pinned-commit drift | Medium | Pin, wrap behind our own interface, and write attention conformance tests against a naive reference kernel |
| R6 | Folly build weight slows iteration | Low | Narrow target build; moodycamel fallback |
| R7 | Tokenizer exactness vs HF | Medium | FFI to the real Rust `tokenizers`; conformance test over a corpus rather than reimplementation |

---

## 13. Proposed repository layout

```
inferx/
├── CMakeLists.txt
├── cmake/                     # FindNCCL, CUDA arch config, submodule glue
├── third_party/               # submodules: abseil, googletest, mimalloc today;
│                              #   cutlass, flashinfer, boost, simdjson, folly,
│                              #   spdlog, tokenizers-cpp as their milestone lands
├── include/inferx/            # public headers
├── src/
│   ├── core/                  # Tensor, DType, Status, arena allocators
│   ├── api/                   # Beast server, OpenAI schemas, SSE
│   ├── tokenizer/             # FFI wrapper, incremental detokenizer
│   ├── scheduler/             # Request, BatchPlan, policy, radix tree, block mgr
│   ├── executor/              # RankWorker, barrier, CUDA graphs, streams
│   ├── model/
│   │   ├── layers/            # attention (GQA|MLA), ffn (dense|moe), norm, rope
│   │   ├── quant/             # W4A16, FP8, KV quant
│   │   ├── loader/            # safetensors mmap, sharded load
│   │   └── arch/              # llama.cpp-style per-arch definitions
│   ├── kernels/               # .cu — custom kernels + FlashInfer/CUTLASS wrappers
│   ├── comm/                  # Communicator: Nccl, SingleRank, HostSim
│   └── observability/         # metrics, tracing
├── tests/
│   ├── unit/                  # scheduler tests run with NO GPU
│   ├── kernel/                # vs naive reference implementations
│   └── e2e/                   # golden-output generation tests
├── bench/                     # throughput/latency harness, ShareGPT replay
└── docs/
```

`tests/unit` running without a GPU is a hard requirement, and it is only achievable because of the §3.1 scheduler/executor split.

---

## 14. Milestones

| M | Deliverable | Proves |
|---|---|---|
| **M0** | CUDA 13.x toolchain floor; CMake + submodules build; `Tensor`, arenas, Status | Toolchain risk R1 retired |
| **M1** | ✅ **FP8 e4m3 W8A8 GEMM vs cuBLASLt FP16 baseline**, with `bench/` harness and CI. Delivered on cuBLASLt, *not* CUTLASS (§15 Q2) | R3 retired for FP8 — ~2× at prefill, reproducibly. Decode pending a clock lock |
| **M2** | Safetensors loader + Llama forward, FP16, batch 1, no cache. Logits match HF reference. | Model layer correctness |
| **M3** | Paged KV + FlashInfer attention + naive scheduler, TP=1, synchronous stepping | The engine exists |
| **M4** | Beast server, tokenizer FFI, SSE streaming, OpenAI API | End-to-end serving |
| **M5** | Continuous batching, chunked prefill, radix prefix cache, preemption | Competitive throughput |
| **M6** | Overlap pipeline (depth 1) + CUDA graphs for decode | G4: CPU off the critical path |
| **M7** | `Communicator` + TP sharding, validated via `HostSimComm` + reconstruction tests | G5 as far as this hardware allows |
| **M8** | W4A16 + FP8 KV quant in the serving path | Fits real models in 16 GB |
| **M9** | MoE (TP-over-experts) and MLA layers | G6 |
| **M10** | Benchmark harness; head-to-head vs vLLM/SGLang on this box | G1 |

M1 before M2 is deliberate: the quantized GEMM story is the largest risk in a torch-free design, and discovering it is intractable *after* building the model layer would be expensive.

---

## 15. Open questions

1. **Which arch first — Llama or Qwen?** Qwen2.5-7B at int4 is a better fit for 16 GB and has broader current relevance; Llama has more reference material. Leaning Qwen2.5-7B-Instruct-AWQ as the M2 target.
2. ~~**Quantization format for v1** — AWQ, GPTQ, or FP8?~~ **Resolved: FP8 e4m3 first, AWQ second.** M1's job is to retire R3, not to ship the final quantization story, and FP8 gets there on the shortest path: sm_89 has native e4m3 tensor cores, CUTLASS's SM89 FP8 GEMMs are well-trodden against the SM80 collective builders, and `kFloat8E4M3FN` is already in `DType`. There is no packing and no dequant in the mainloop, so a wrong number is a kernel bug rather than a layout bug. The cost is real and accepted: 2× compression leaves a 7B with roughly 5–6 GB for KV where W4A16 would leave ~11 GB, so **FP8 is the de-risking vehicle, not the endpoint** — AWQ still has to land for §1's 16 GB envelope to work, and it follows in M8 against a harness that already exists.

   **Delivered, and it did not need CUTLASS.** cuBLASLt implements FP8 e4m3 matmul natively on sm_89, so M1 shipped against a dependency the project already had. Two things made that free rather than lucky: cuBLASLt's FP8 path is only implemented for the TN layout, which is exactly what the row-major `y = x·wᵀ` mapping already produces, so storing weights `[out, in]` as checkpoints do turned out to be what makes FP8 reachable at all; and the scales are bound as device pointers per call, so they never round-trip to the host. CUTLASS is consequently **not an M1 dependency** and moves to M8/M9, where mixed-input W4A16 and grouped MoE GEMM genuinely have no alternative. That is roughly a week of integration removed from the critical path, and one fewer 500 MB submodule until it earns its place.
3. **Structured output (JSON schema / grammar)** — a differentiator and a scheduler-visible feature (the logit mask must be computed per step per sequence, on CPU, in the overlap window). Not in the milestone list; needs a decision on whether it is v1 scope.
4. **Determinism guarantee** — do we promise bitwise-reproducible output for a fixed seed and batch composition? This constrains kernel selection (split-k reductions, atomics) and is much cheaper to promise now than to retrofit.
