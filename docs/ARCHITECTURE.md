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

### 5.2a Not built, and why — the measurement that retired it

`bench/overlap_bench.cc` splits a step into the four phases the host can see and asks how much of it is host work sitting serially in front of the device. That is the ceiling on what this section could ever be worth, and on this hardware it is **1.6–2.1%**.

| batch | prepare | issue | await | commit | step | host % |
|---|---|---|---|---|---|---|
| 1 | 0.007 | 0.167 | 11.036 | 0.002 | 11.211 | 1.6% |
| 8 | 0.015 | 0.180 | 11.650 | 0.003 | 11.848 | 1.7% |
| 32 | 0.033 | 0.225 | 13.735 | 0.006 | 14.000 | 1.9% |

`prepare` and `commit` are the scheduler — 0.02–0.04 ms, **0.2% of a step**. All of §8's machinery, admission and chunking and preemption and the radix descent, costs a fortieth of a millisecond. The rest of the host time is `issue`, and a depth-1 pipeline cannot remove that: issuing is what the pipeline does with the time it saves, and it cannot overlap with itself.

Two things this does *not* say. It is not an argument that the pipeline is a bad design — the same measurement launch-by-launch puts the host at **20–25%**, which is what §5.2 was written against and what it would in fact recover. And it is not confined to pure decode: a mixed workload with prompts arriving into a decoding batch measures 1.9% overall, with the rare prefill-carrying step at 3.7% of a 94 ms step.

What happened is that **M6's other half already collected M6's win**. CUDA graphs cut `issue` from 2.9 ms to 0.18 ms, and there is no longer enough host time left for an overlap pipeline to hide. The remaining 1.9% would cost speculative scheduling — planning step N+1 without knowing whether a sequence finished at N — plus a permanent second code path, since this section itself requires depth 0 to stay as the differential-testing reference. That is a poor trade against a scheduler that has just absorbed chunked prefill, preemption and prefix caching.

Revisit if any of the premises move: a much faster step (W4A16 at M8 would take the bf16 11.2 ms toward 5), a much larger batch than 32, TP where rank coordination joins the host path, or a workload that defeats graph capture. The benchmark is the thing to re-run, not the reasoning.

One caveat, discovered while writing it: the first mixed-workload measurement read 20.3% host, because the benchmark captured a graph for one sequence count while a mixed workload's count changes on every arrival and completion. The server captures every count from `max_running` down to 1; the benchmark now does the same. A graph that does not match is a graph that is not there.

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

**Delivered.** `PrefixCache` in `src/scheduler/`, on by default, matched at admission (§4 step 4) and offered back when a sequence retires *or is preempted*. Four things the section did not settle:

- **Complete blocks only.** A block still being written cannot be shared, so every run in the tree is a whole number of blocks and a sequence's partial tail is never offered. That is also what makes sharing safe without copy-on-write: a request that matches reads those blocks and writes only from the match onward, which is a block boundary by construction. The other half of the safety argument is that a key depends on the whole prefix — which is exactly what the path through the tree encodes.
- **A match is always one token short of the sequence.** A request whose every token is cached would have nothing to forward and so no logits to sample its first token from.
- **Eviction trims from the tail of a node rather than dropping it whole.** The end of a prefix is the least useful part of it, and what is left after trimming is still a prefix somebody can match. Evicting whole nodes made a preemption reclaim its own donation immediately, which took the hit rate under exactly the pressure the cache exists for to zero.
- **Preemption donates rather than discards** (§8.2, T7). A preempted sequence's history goes into the tree, where it stays evictable but matchable, so re-admission is usually a lookup instead of the second prefill recompute budgets for. This is the point at which T7's choice of recompute over swapping actually pays.

#### It costs bitwise determinism, and that is not free

A cache hit changes what the model computes *with*, not just how fast. A warm request forwards only its unmatched tail, so cuBLASLt may pick a different algorithm for the shorter GEMM, and bf16 reductions do not commute. Measured on a partial match: `max|diff|` of 0.223 over a logit range of 21.5, the same ~1% as chunked prefill's rearrangement.

Usually invisible. On a close call it is not: `ServerTest.IdenticalRequestsProduceIdenticalOutput` asks for three colours and gets "Green" cold, "Yellow" warm, stably and forever after. **So greedy output can depend on cache state, which depends on other traffic.** Every engine with prefix caching has this; it is worth stating plainly rather than discovering. The answer remains stable for a given cache state — that weaker property is what the test now asserts, and it is still enough to catch R8, whose symptom was alternation between consecutive requests. `SchedulerConfig::enable_prefix_cache = false` buys the stronger guarantee back at the cost of the throughput.

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

**Built at M9** as `model::MoeFfn` (`include/inferx/model/moe_ffn.h`), at TP=1. The shape follows the decision above: routing, grouping and the GEMM loop are separate calls, so TP-over-experts is this same code with a narrower `moe_intermediate` and one all-reduce appended, while EP replaces the grouping and combine without touching the router. Two properties were treated as requirements rather than nice-to-haves, and both cost implementation choices. **Determinism** (§6.3, R8): no kernel uses an atomic to place or accumulate a value, because a float sum whose order depends on thread scheduling breaks the identical-requests-identical-tokens contract — so the grouping is a stable block-per-expert compaction rather than an atomic cursor, and the combine sums each token's `k` contributions in slot order from one block. **Graph-capturability**: the dispatch writes its counts into the offsets buffer and scans them in place, so it needs no scratch and no synchronize. The GEMM loop then spoils that anyway — a GEMM's `m` is a host argument, so the offsets have to come back to the host once per layer per step. That round-trip is the strongest argument for a grouped GEMM, over and above the `E` launches.

**MLA (DeepSeek-V2/V3)** — the important one for the KV abstraction. MLA caches a *compressed latent* per token per layer (plus a small decoupled RoPE key), not per-head K and V. Two consequences that must be baked in now:

1. The KV pool layout `[k|v][heads][head_dim]` **does not apply**. The KV manager must be parameterized by a `KvLayout` describing bytes-per-token-per-layer, with the per-head split being one instantiation and the MLA latent another. Getting this wrong means rewriting the block allocator later.
2. **The MLA latent is replicated across TP ranks, not sharded.** KV memory therefore does *not* shrink with TP for MLA. Any capacity-planning code that assumes `kv_bytes / tp_size` is wrong for MLA. This must be a property queried from the model, never computed by the scheduler.

**Built at M9** as `model::MlaAttentionLayer` (`include/inferx/model/mla.h`), at TP=1. Both consequences landed as stated: the pool takes `MlaAttentionLayer::LayoutFor(config)` — `entries_per_token = 1`, `kv_heads = 1`, `head_dim = kv_lora_rank + qk_rope_head_dim` — and `KvBlockPool::ValueCache()` *fails* for that layout rather than returning half a latent, which a test pins; and the per-token cost is `ModelConfig::KvElementsPerTokenPerLayer()`, a model property, so nothing downstream is in a position to divide it by `tp_size`. For DeepSeek-V2's numbers that is 576 elements per token per layer against GQA's 8192.

What is built is the **unabsorbed** form: each step gathers the sequence's latents, runs one `kv_b` GEMM to reconstruct every cached token's K and V, then attends. It is correct and it is what the tests compare against a host reference, but it hands back most of MLA's decode advantage — the reconstruction is O(context) work every step, where the absorbed form folds `kv_b` into the query once and attends against the latent directly. Absorption changes no interface here and is the obvious next move.

One honesty note that matters more than it looks. **The RoPE convention is a choice, not a verified match.** MLA rotates the trailing `qk_rope_head_dim` of each Q head and the one shared key; this implements the half-split ("NeoX") form, the same rotation `RotaryEmbedding` applies, and the tests assert that the two agree with each other. HF's DeepSeek implementation reaches the same mathematics through an interleaving of its own, and with no DeepSeek checkpoint on this box there is no way to assert the two produce identical numbers. A loader will have to settle that against real weights before any MLA output can be called correct rather than self-consistent.

### 7.4 Making blind TP honest

TP cannot be perf-tested on this box. Three mitigations, all cheap now and expensive to retrofit:

1. **`Communicator` is an interface.** Backends: `NcclComm` (real), `SingleRankComm` (no-op, always exercised at TP=1), and `HostSimComm` — a CPU shared-memory backend that runs N rank threads on *one* GPU or even on CPU tensors. `HostSimComm` lets us correctness-test all sharding logic and SPMD control flow here, today, at TP=4.
2. **Sharded weight loading is validated by reconstruction.** A test loads a model at TP=4 into host buffers and asserts the concatenated shards equal the TP=1 weights bit-for-bit. This catches ~80% of TP bugs without a second GPU.
3. **Numerical differential test.** TP=1 vs TP=4-under-`HostSimComm` logits must match within FP tolerance for a fixed prompt.

Real NCCL perf work is explicitly deferred until multi-GPU hardware exists, and the doc should say so rather than pretend otherwise.

---

## 8. Scheduling policy

### 8.1 Continuous batching with chunked prefill

Every step builds a **mixed batch**: some decode slots (1 token each) plus prefill chunks capped by a global token budget, `SchedulerConfig::max_batch_tokens` (default 2048).

Chunking prefill is what keeps p99 TBT flat. Without it, a 32k-token prefill monopolizes a step and every decoding sequence stalls for hundreds of milliseconds. The cost is that a chunked prefill is slightly less GEMM-efficient than a monolithic one, and each chunk re-reads prior KV. We take that trade because G2 (stable TBT) is a stated goal and raw prefill throughput is not.

Priority within a step: **decode first, then prefill fills the remaining token budget.** This bounds TBT by construction.

**Delivered.** `PrepareStep` plans the whole batch before it writes any of it, because the two orders differ: the budget goes to decodes first wherever they sit in the running set, while the batch itself must be emitted in running order — the attention plan requires query rows grouped by sequence and in ascending sequence index. So it is two planning passes over `running` and one emission pass.

Two consequences worth stating, because both were latent bugs before rather than new behaviour:

- **A prompt longer than the budget is now servable at all.** It used to be admitted, then skipped on every step for not fitting, forever.
- **A prefill can no longer take the budget ahead of a waiting decode.** The old loop walked `running` in arrival order and gave each sequence all or nothing, so a prompt admitted before a decoding sequence pushed that sequence's next token into a later step — the precise TBT hole this section exists to close.

Only the chunk that reaches the end of a prompt asks for logits. The earlier ones compute keys and values and nothing else, so `ForwardBatch::logits_indices` is empty on those steps and `CommitStep` takes no tokens back. That is an ordinary step, not an idle one.

Measured in `bench/tbt_bench.cc` and tabulated in §14: on this workload chunking is worth **10.6× on p99 TBT** (183 → 17.3 ms) and costs **1.6× on TTFT** (183 → 294 ms). That is the trade this section commits to, now with numbers on both sides of it.

Chunking also forced sequence lengths to become **explicit** in the plan (`ForwardBatch::seq_lens`). The model used to recover each sequence's KV length from the largest query position it owned, which only works while every sequence contributes a query. Under chunking a sequence the budget skipped contributes none, and its history does not stop existing — so the batch states the lengths and the model reads them. Without that, any step containing a skipped sequence fell back to the reference attention kernel, which the measured prefill table below prices at 12.8× slower at 2k.

### 8.2 Preemption

When KV allocation fails for a running sequence:

- **Recompute** (default): drop the sequence's KV, return it to the waiting queue head. Its prefix is likely still in the radix tree, so re-admission is cheap. Wastes compute, frees memory instantly, no host bandwidth.
- **Swap to host**: D2H the blocks. Preserves work but costs PCIe bandwidth on a contended bus and adds a copy the overlap pipeline must hide.

Default to recompute; swap exists behind a flag for long-context workloads where recompute cost dominates. With prefix caching enabled, recompute is nearly always the right call — this is a change from vLLM's original defaults and is worth measuring rather than assuming.

**Delivered, recompute only.** Swap is not built and is not needed until there is a long-context workload to justify the D2H copy.

Three things the implementation had to settle that this section did not say:

- **A full context is not a shortage.** `Reserve` failing because the sequence hit `max_seq_len` and failing because the pool is empty were the same `ResourceExhausted`, and the combined error left `PrepareStep` — where the engine's only available answer is to fail every request in flight. So one sequence reaching its context limit took the whole server down with it. They are now distinct outcomes: a full context retires that sequence alone with `kContextLimit`, and only an empty pool preempts.
- **The victim is the most recently admitted sequence**, and it goes to the *head* of the waiting queue. Newest-first because it has the least invested; head-of-queue because it was admitted before everything still waiting and a client has already seen tokens from it, so sending it to the back is the one unfairness FCFS exists to prevent. When the sequence that needs the block *is* the newest, the next-newest gives way instead — self-preemption would free that sequence's blocks only to demand all of them back plus one.
- **Admission is deliberately not a preemption site.** A shortage there means there is no room for a request that has not started; taking KV from a sequence already producing tokens to admit one that is not would trade real progress for none. It still defers, as before.

The one case with no answer is a single running sequence that has drained the pool by itself: there is no victim, so it retires with `kOutOfMemory`. That reason now means specifically "nothing could be taken on its behalf" rather than "memory ran out".

Preemption requires *reserving for the whole batch before emitting any of it*, since removing a sequence renumbers every row after it and changes how the token budget divides. That is the same plan-then-emit split §8.1 already needed, which is why the two land together.

`Scheduler::preemptions()` counts them. It is the number to watch under load: preemption is pure wasted compute, and a rate that climbs means the pool is undersized for `max_running` rather than that anything is wrong.

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
| T4 | Radix-tree prefix cache | Block-hash (vLLM) | Token-granular longest-match, no per-block hashing. More complex; safe because single-writer. **Built (§6.3).** Eviction is a linear walk over leaves, which is fine while the tree has one node per distinct prefix rather than one per block | If tree walk shows in profiles |
| T5 | Chunked prefill, decode-priority | Prefill-priority / separate phases | Bounds p99 TBT by construction (G2). Costs some prefill throughput. | With 2+ GPUs, evaluate disaggregated P/D |
| T6 | Overlap depth 1 | Synchronous stepping | Hides all CPU work behind GPU. Costs < 1% wasted steps on finished seqs. | Depth 0 stays as the correctness reference |
| T7 | Recompute preemption | Swap to host | Prefix cache makes re-admission cheap; no PCIe contention. **Collectable since §6.3 landed** — a preempted sequence donates its history to the tree, so coming back is usually a match rather than a second prefill | Long-context workloads where recompute dominates |
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
| R3 | Torch-free quantized GEMM (FP8, W4A16, MoE grouped) is the largest single unknown | ~~High~~ **Retired for FP8; Medium for W4A16** | **FP8 e4m3 W8A8 lands at ~2× FP16**, measured with the SM clock locked at 2400 MHz (`nvidia-smi -lgc 2400`) and reproducible to two decimals across independent runs: 1.92–2.05× for every shape at m ≥ 128, and 183.9 vs 92.5 TFLOP/s at m=8192 — the 2× the FP8 tensor cores promise. Mean 2.34× over 28 shapes. Two exceptions matter more than the mean, both below. Torch-free quantized GEMM is demonstrated. **W4A16 de-risked at M8:** CUTLASS 4.6.1 has *no turnkey* int4×fp16 path for sm_89 (mixed-input `CollectiveBuilder` is Hopper/TMA-only; the SM80 `MmaMixedInputTensorOp` framework lacks the ratio-4 int4 warp-fragment shuffle and produces wrong results — see `mma_mixed_input_tensor_op.h`), so a true fused W4A16 needs a custom kernel (in-mainloop dequant, or a third-party kernel like Marlin). The foundation landed regardless: `QuantizeF16ToInt4`/`DequantizeInt4ToF16` (per-group symmetric int4, group 128, packed 2-per-byte) and a dequant+cuBLASLt path that is correct and is the **lower bound** a fused kernel has to beat. Measured on the decode shapes it is 0.22–0.75× of fp16 (the double-touch penalty: read int4, write fp16, read fp16 again); a fused kernel reading int4 once would approach 4× fp16, so the headroom is roughly an order of magnitude. |
| R3a | **Quantization can cross the L2 residency threshold, and that is worth more than the arithmetic.** `down_proj` at m=8 runs **6.2×**, not 2×, reproducibly across every run. Its f16 weights are 135.8 MB against a 64 MB L2; the FP8 copy is 67.9 MB and becomes nearly resident, measuring 957 GB/s — above this card's ~736 GB/s HBM peak — where f16 manages 307. | — | Not a risk but a design input: the decode-path win from quantization is a step function at each cache threshold, not a smooth 2×. It strengthens the case for W4A16 (§15 Q2), which would push more shapes across the same line, and it means KV-cache and weight sizing should be reasoned about against L2 as well as HBM. |
| R3b | **cuBLASLt's FP8 heuristic is not uniformly better.** `gate_up` at m=8 measured **0.88×** — FP8 *slower* than FP16 — in two of three locked runs. | Low | **Mitigated in M6.** `GetOrBuild` now asks the heuristic for up to 8 candidates and `Tune()` times them on zeroed FP8 scratch at plan-build time, keeping the fastest (FP8 path only, memory-bound shapes `m ≤ 64`; the heuristic stays untouched for f16/bf16 and for compute-bound prefill). The investigation refined the root cause: the heuristic's top-1 sits in a ~0.45 ms / half-bandwidth cluster even when ~0.21 ms / full-bandwidth algorithms are reachable, and the tuner selects the fast one whenever the device is in that state — a strict improvement, since in the state where no algorithm is fast they are all ~0.45 ms and the pick ties top-1. Tuning runs once per FP8 plan at warmup and costs **~2.25 s across the 12 decode shapes** (m ≤ 64; measured), one-time at engine startup; `CublasLtGemm::Create(., /*autotune=*/false)` skips it for differential testing or a faster cold start. The residual cross-run variance (the original "2 of 3 runs") is the **unlocked memory clock**, not algorithm selection: only `-lgc` (SM clock) is locked, `-lmc` (mem clock) floats, and a bandwidth-bound GEMM reads at half bandwidth when DVFS has not settled — `nvidia-smi` even reads 11501 MHz in both states, so the reading cannot be trusted to reflect per-launch effective bandwidth. The tuner's ramp reaches the fast state more often than a cold start does. Locking the mem clock would be the only complete fix, **but it is not available on this setup**: under WSL (where the bench runs) `nvidia-smi -lmc` hangs — the WSL driver does not support clock control at all, and the existing 2400 MHz SM lock is applied from an elevated Windows process and inherited read-only — and native Windows `nvidia-smi -lmc` reports "no permission" without elevation, with consumer Ada (RTX 4080 SUPER) further restricting mem-clock locking and `SUPPORTED_CLOCKS` exposing only a single mem-clock value. The variance is therefore inherent to the hardware/environment and must be tolerated — take the best of several cold runs, or validate on the serving path at M8 where sustained decode load holds the mem clock up. Re-measure there, where FP8 decode first runs. |
| R4 | 16 GB constrains what can be validated end-to-end | Medium | Target 7B-class at W4A16 for the main loop; MoE/MLA validated for correctness at toy sizes |
| R5 | FlashInfer pinned-commit drift | Medium | Pin, wrap behind our own interface, and write attention conformance tests against a naive reference kernel |
| R6 | Folly build weight slows iteration | Low | Narrow target build; moodycamel fallback |
| R7 | Tokenizer exactness vs HF | Medium | ~~FFI to the real Rust `tokenizers`~~ — **taken the other way at M4.** Reimplemented as byte-level BPE to keep Rust off the build, so exactness is asserted rather than guaranteed: 140-string corpus generated by HF's own tokenizer, exact id equality required. FFI remains the fallback if the corpus ever fails on input we cannot fix |
| R8 | ~~KV state leaks between sequences~~ | **Retired** | Found and fixed at M4. `PrepareStep` copied a sequence's block-table row *before* `Reserve` grew it, so on the one step where a sequence first reached into a newly allocated block, `slots` wrote the KV there while the block table still read 0 — sending attention to block 0. Silent whenever that zero happened to name the right block, which is why the symptom was an alternating answer rather than a steady one: a stack free list hands consecutive requests their blocks in opposite order. Guarded host-side by `SchedulerTest.EveryBatchSlotIsCoveredByTheBatchBlockTable`, which asserts every batch slot is reachable through that batch's own table |
| R9 | ~~CUDA graph capture corrupts the first sequence's KV~~ | **Retired** | Found and fixed. `CaptureDecodeGraph`'s probe runs the real decode body, so it really appends keys and values — and it wrote them to slot 0, which is physical block 0, which the descending free list hands out *first*. Capturing a graph therefore overwrote the first live sequence's position-0 KV in every layer. The probe now borrows a block from the pool and returns it. Replay was never at fault: the test that found this runs launch-by-launch *after* capturing a graph for an unrelated shape and was corrupted just the same. It hid since M3 because every graph test compared sampled tokens rather than logits, and an argmax survives a large error; `EngineTest.GraphedAndUngraphedLogitsAgreeForABatch` now compares logits and requires bit-equality, with controls that separate capture-corrupts-state from replay-is-wrong. `capture_graphs` is on by default again. The 1.05x previously attributed to CUDA graphs had been measured on a corrupted computation; **re-measured on correct arithmetic it is 1.06–1.07x across batch 1 to 32**, so the claim stands. |

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
| **M1** | ✅ **FP8 e4m3 W8A8 GEMM vs cuBLASLt FP16 baseline**, with `bench/` harness and CI. Delivered on cuBLASLt, *not* CUTLASS (§15 Q2) | R3 retired for FP8 — ~2× throughout, 6.2× where FP8 makes weights L2-resident (R3a), one shape slower (R3b) |
| **M2** | Safetensors loader + Llama forward, FP16, batch 1, no cache. Logits match HF reference. | Model layer correctness |
| **M3** | Paged KV + FlashInfer attention + naive scheduler, TP=1, synchronous stepping | The engine exists |
| **M4** | ✅ **HTTP server, tokenizer, SSE streaming, OpenAI API, temperature/top-p sampling.** The original cpp-httplib transport has migrated to Boost.Beast/Asio (§9/§5.1); tokenizer history remains documented in R7. | End-to-end serving — `inferx-serve` answers, honours `temperature`/`top_p`/`seed`, and building it found the paged-attention bug R8 and the graph-capture bug R9 that every single-request test had missed |
| **M5** | ✅ **Chunked prefill and decode-first mixed batching** (§8.1, 10.6× on p99 TBT), with sequence lengths made explicit in `ForwardBatch`; **recompute preemption** (§8.2); **radix prefix cache** (§6.3, 5.9× on warm TTFT behind a shared preamble), which costs bitwise determinism across cache states | Competitive throughput |
| **M6** | ✅ **CUDA graphs for decode** (delivered at M3/M4, 1.06–1.12×). **Overlap pipeline measured and not built**: with graphs on, the host is 1.6–2.1% of a step and the scheduler 0.2%, so there is nothing left to hide (§5.2a) | G4: CPU off the critical path — **met**, 1.9% against a 30% target |
| **M7** | **Tensor parallelism implemented; physical two-GPU validation pending.** `Communicator`, `SingleRankComm`, and deterministic `HostSimComm` validate reusable collectives and dense Qwen2 TP arithmetic. The optional `NcclComm` adds opaque bootstrap IDs, fixed rank/device ownership, stream-ordered in-place BF16 all-reduce, graph capability reporting, and abort without leaking NCCL types into model code. Qwen2 derives every allocation from its communicator device. The server's TP=2 runner owns two persistent rank threads/models, mirrors each `ForwardBatch`, returns rank-0 samples, captures/replays both rank graphs concurrently, and aborts both communicators after a rank error or 30-second worker timeout; CLI selection is `--tensor-parallel-size 2 --devices 0,1 --comm-backend nccl`. A hardware-gated test exercises two-GPU BF16 reduction on non-default streams. The existing real-checkpoint TP=1 serving regression and all HostSim TP comparisons pass. **Open:** run the NCCL test, injected-failure case, and end-to-end TP=2 checkpoint/serving suite on the rented host, then record topology and scaling benchmarks. | G5 implementation complete; hardware gate open |
| **M8** | W4A16 + FP8 KV quant in the serving path | Fits real models in 16 GB. **Serving paths in:** `--w4a16` quantizes Qwen2 projection weights to per-group symmetric int4 at load, releases their bf16 storage, and dispatches captured serving projections through the fused `W4A16Gemm`; the real 3B checkpoint shrinks from 6.17 GB to 2.05 GB resident weights and preserves the expected continuation. `--fp8-kv` selects an e4m3 `KvBlockPool`, freezes per-layer scales during warmup, quantizes on write, and dispatches graph-captured decode through `RunFp8`; its end-to-end serving continuation is also validated. The fp8 **prefill** path remains worked around because of FlashInfer's fa2 fp8 V-fragment bug (`prefill.cuh:988-1031`); serving dequantizes fp8 KV to bf16 for prefill and keeps the fast fp8 decode. **Open:** tune and benchmark the custom W4A16 kernel until it beats bf16 on serving decode shapes; fix upstream `PrefillFp8` V scaling before prefill can consume fp8 KV directly. |
| **M9** | ✅ **MoE and MLA layers, at TP=1, validated against host references.** **MoE:** a deterministic router (softmax + top-k, ties to the lower index), a stable atomic-free grouping of the `tokens·k` assignments by expert, gather/combine permutation kernels, and `MoeFfn` running one GEMM per expert plus Qwen2-MoE's shared expert. **MLA:** `MlaAttentionLayer` over a paged *latent* cache — `KvLayout{entries=1, heads=1, head_dim=kv_lora_rank+qk_rope_head_dim}`, the instantiation §7.3 required the pool to admit — with the decoupled RoPE key shared across heads, and `KvElementsPerTokenPerLayer()` as a model property so no scheduler ever divides it by `tp_size`. Config parsing and validation for both. **Open, and deliberately so:** (1) neither is wired into a `Model` and neither has a checkpoint on this box, so correctness is against the definition rather than against HF logits — in particular *the RoPE convention is unverified against DeepSeek's*; (2) the per-expert GEMM loop is `E` cuBLASLt calls where a CUTLASS grouped GEMM belongs, and its host round-trip for the offsets makes the MoE FFN uncapturable as a graph; (3) MLA is the **unabsorbed** form, reconstructing K/V for the whole context every step — absorption is what makes MLA decode cheap and is the next thing to build; (4) TP itself is M7 | G6 — layers exist and are tested; no throughput claim |
| **M10** | ✅ **Serving benchmark harness and the head-to-head vs vLLM 0.26** (`bench/serve_bench.py`, `scripts/bench_serve.sh`): TTFT/ITL/decode at batch 1, an output-throughput sweep over concurrency, and prefill tok/s, measured through one client so a row differs only by the engine under it. **bf16 lands within a few percent of vLLM on decode** (0.97x batch 1, 0.93x at concurrency 8), ahead on TTFT, 0.86x on prefill; **fp8 leads vLLM by 1.37x on decode and 1.17x on throughput**. Found and fixed two defects no microbenchmark could see: `TCP_NODELAY` off (a flat ~43 ms TTFT floor) and single-block FP8 activation quantization (prefill 3727 → 14691 tok/s). Also closed an API gap it needed — `stream_options.include_usage`. SGLang still unmeasured | G1 — **met for bf16 parity, exceeded for fp8** |

### Measured decode performance

Qwen2.5-3B-Instruct, batch 1, context 256, RTX 4080 SUPER with SM clocks locked
at 2400 MHz. Re-measured after R9 was fixed, because every earlier figure was
taken while graph capture was corrupting a sequence's KV.

| configuration | launch-by-launch | with CUDA graphs | weight-bandwidth floor | % of floor |
|---|---|---|---|---|
| bf16 | 11.58 ms | 10.90 ms | 8.39 ms (6.17 GB @ 736 GB/s) | 77% |
| FP8 e4m3 | 9.28 ms | 8.29 ms | 4.62 ms (3.40 GB @ 736 GB/s) | 56% |

Sustained batch-1 generation with on-device sampling: 89.7 tok/s bf16, 118.4
tok/s FP8. CUDA graphs are worth 1.06x on bf16 and 1.12x on FP8 -- the same
~0.4 ms of dispatch overhead against a shorter step.

The step is weight-bandwidth bound, which is why the floor is quoted: `gate_up`
and `down_proj` alone are 7.3 ms of the bf16 step at 642-682 GB/s, and no
amount of launch elimination touches them. That is the argument for FP8 and,
later, for W4A16 (M8).

#### Prefill, measured for the first time

Batch 1, bf16, one sequence, FlashInfer paged prefill. Minimum of 10 measured
runs after 3 warmups on the RTX 4080 SUPER.

| prompt tokens | prefill ms | tok/s | vs one decode step |
|---|---|---|---|
| 128 | 15.821 | 8091 | 1.5x |
| 256 | 26.169 | 9783 | 2.4x |
| 512 | 45.834 | 11171 | 4.2x |
| 1024 | 89.971 | 11382 | 8.3x |
| 2048 | 176.127 | 11628 | 16.2x |
| 4096 | 358.780 | 11417 | 32.9x |
| 8192 | 773.677 | 10588 | 71.0x |
| 16384 | 1776.047 | 9225 | 162.9x |

The integration removes the engine's worst bottleneck: 2k falls from 2.25 s to
176 ms (12.8x), 8k from 105 s to 774 ms (135.8x), and 16k now runs in 1.78 s
instead of exceeding the reference kernel's shared-memory ceiling. Throughput
holds above 9.2k tok/s across the matrix. Multi-sequence ragged prefill uses the
same path and is guarded by page-permutation, block-reuse, per-layer K/V, and
server-lifecycle repeatability tests.

#### Inter-token latency under a mixed workload

The first measurement of what §8.1 is actually about, and the first that drives
the scheduler rather than a hand-built batch — inter-token time is a property of
how steps are *composed*, so there is nothing to measure without the component
that composes them. `bench/tbt_bench.cc`, same box and clocks, CUDA graphs off.

Eight sequences decoding throughout; one 2048-token prompt arrives partway in.
TBT is the wall-clock gap between one sequence's consecutive tokens — what a
client experiences — so it is measured on the wall clock rather than with CUDA
events. TTFT is the arriving prompt's own latency, from submission to its first
token. With the decoders alone and nothing arriving, p50 TBT is 11.7 ms and p99
is 12.6 ms: that is the floor, and chunking cannot go below it.

| chunk | p50 TBT | p99 TBT | worst step | TTFT | chunks | prefill tok/s |
|---|---|---|---|---|---|---|
| 128 | 12.2 ms | **16.9 ms** | 16.9 ms | 291 ms | 18 | 7047 |
| 256 | 12.1 ms | 27.5 ms | 27.5 ms | 230 ms | 9 | 8921 |
| 512 | 12.0 ms | 47.6 ms | 47.6 ms | 200 ms | 5 | 10250 |
| 1024 | 12.1 ms | 91.4 ms | 91.4 ms | 193 ms | 3 | 10635 |
| 2048 | 12.0 ms | 167.1 ms | 167.1 ms | 179 ms | 2 | 11423 |
| 4096 | 12.0 ms | **180.9 ms** | 180.9 ms | 181 ms | 1 | 11320 |

`chunk` is `max_batch_tokens`. The last row exceeds the prompt and runs it whole
in one step, so it is what chunking replaced: **p99 TBT falls 10.6×, from 181 ms
to 16.9 ms, while TTFT rises 1.61×, from 181 ms to 291 ms.** Chunking does not
make the prefill cheaper and is not meant to — it decides who waits for it.
Reproducible to 1–2% across runs.

Three things worth reading off it beyond the headline:

- **p50 is flat at ~12 ms in every row.** The median never moves, because the
  median step is a decode either way. A mean would have hidden the entire
  effect, which is why the tail is what is reported.
- **The 4096 row's 11320 tok/s cross-checks the prefill table above**, which
  measures 11628 tok/s for a 2048 prompt in isolation. Interleaving eight
  decodes into those steps costs ~3%, and the two harnesses agreeing to that
  margin is the reason to believe either.
- **The 2048 row takes two chunks, not one.** Decodes are served first and spend
  8 tokens of the budget before the prompt sees any of it, so a prompt exactly
  the size of the budget never fits in one step. Not a defect — a consequence of
  the priority rule, and visible here rather than inferred.

The knob is genuinely two-sided, so `max_batch_tokens` stays at 2048 as a
default rather than being tuned to the best column: the right value depends on
whether a deployment is serving chat (TBT dominates) or batch summarization
(TTFT dominates), and this table is what makes that choice with evidence.

#### What the prefix cache is worth

`bench/prefix_cache_bench.cc`, same box and clocks, CUDA graphs off. Requests
run one at a time, because this measures the prefill path and interleaving
another sequence's decodes would put its step time in the denominator. Three
workloads, each run twice with `enable_prefix_cache` the only difference.
`cold` is the first request, which can never hit; `warm` is the mean of the
rest.

| workload | | cold TTFT | warm TTFT | prefill tokens | reused |
|---|---|---|---|---|---|
| shared system prompt, 1072 tok | off | 96.5 ms | 96.8 ms | 8576 | 0 |
| | **on** | 98.5 ms | **16.3 ms** | 1408 | 7168 |
| multi-turn chat, 6 turns to 736 tok | off | 26.4 ms | 51.8 ms | 2976 | 0 |
| | **on** | 26.4 ms | **15.4 ms** | 576 | 2400 |
| distinct prompts, 1024 tok | off | 89.5 ms | 89.5 ms | 8192 | 0 |
| | **on** | 88.7 ms | 89.5 ms | 8192 | 0 |

**A shared preamble costs 5.9× less to serve after the first request** — 96.8 ms
of time-to-first-token falls to 16.3 ms, because each warm request forwards 48
tokens instead of 1072. The match is 1024 rather than the full 1072: it rounds
down to a block boundary and stops one token short of the sequence, and at
block 16 that lands on 64 blocks exactly.

**Multi-turn is the case where recomputing is most obviously absurd**, and it is
the only row where the *uncached* number gets worse as the workload proceeds:
each turn resends the whole conversation, so without a cache turn 6 re-prefills
everything turns 1–5 already computed, and warm TTFT (51.8 ms) runs to twice
cold (26.4 ms). With the cache it is flat at ~15 ms regardless of how long the
conversation has grown, which is the property a chat deployment actually feels.

**The third row is the one that had to be checked rather than assumed.** A miss
costs a tree descent against the prompt's first tokens, and if that were
measurable against a 90 ms prefill it would be the finding. It is not: 1.00×,
within run-to-run noise.

Two cross-checks that the harness is measuring what it claims. Cold TTFT for the
256-token chat opening is 26.4 ms against `prefill_bench`'s 26.2 ms for 256
tokens measured standalone, and the 1072-token cold prompt lands at 96.5 ms
against 89.97 ms for 1024 — both within what the extra tokens and the
scheduler's per-step overhead account for. And the reuse arithmetic closes
exactly: 7168 reused over 7 warm requests is 1024 each, which is the block-
aligned match of a 1072-token prompt.

Reproducible to ~1% across runs.

#### And what it costs when the pool is too small to hold it

`bench/eviction_bench.cc`. The table above gives the cache room to breathe,
which is the easy half. The hard half is what happens when cached blocks and
running sequences want the same memory, and there are three ways that could go
badly: **churn** (blocks evicted before anyone reuses them, so the hit rate
collapses and the tree walk is pure overhead), **displacement** (the cache holds
memory a running sequence needed, so the engine preempts where it otherwise
would not), or a **soft landing** — eviction gives memory back on demand and the
cache degrades to roughly the uncached case rather than below it.

12 requests sharing a 1024-token preamble, 4 resident at once, 64 generated
each, submitted as one burst so the pool is genuinely contended. The pool is
swept downwards; `enable_prefix_cache` is the only thing that differs between
the two rows at each size.

| blocks | cache | wall ms | mean TTFT | hit % | preempt | evicted | |
|---|---|---|---|---|---|---|---|
| 320 | off | 4575 | 2497 | 0 | 0 | 0 | |
| | on | 3892 | 2220 | 64 | 0 | 0 | 1.18× |
| 192 | off | 5965 | 2628 | 0 | 0 | 0 | |
| | on | 3469 | 1431 | 80 | 0 | 0 | 1.72× |
| 128 | off | 10480 | 4897 | 0 | 0 | 0 | |
| | on | 3413 | 1546 | 88 | 0 | 11 | **3.07×** |
| 96 | off | 10662 | 4943 | 0 | 0 | 0 | |
| | on | 3478 | 1586 | 88 | 0 | 43 | 3.07× |
| 88 | off | 10526 | 4951 | 0 | 0 | 0 | |
| | on | 3352 | 1534 | 89 | 2 | 56 | 3.14× |
| 80 | off | 10547 | 4918 | 0 | 0 | 0 | |
| | on | 4595 | 1804 | 92 | 11 | 76 | 2.30× |

**The cache is worth more under pressure, not less** — 1.18× with room to spare,
3.07× at a quarter of the memory. The reason is that the two configurations do
not have the same working set. Uncached, each resident sequence needs its whole
1136 tokens: 71 blocks, times four is 284. Cached, all four point at the *same*
64 blocks of preamble and own only their own tails, so four of them fit in about
96. Prefix caching is a memory optimization as much as a compute one whenever
the sharing is between *concurrent* requests, and the uncached column is a step
function of how many sequences still fit — four, then two, then one.

**The hit rate rises as the pool shrinks**, 64% to 92%, which looks backwards
until you see what causes a miss here: concurrency. The first four requests are
admitted together, before any of them has finished and populated the tree, so
all four miss. Less memory means less concurrency means more requests arriving
after somebody else has already paid for the preamble.

**No churn.** Evictions climb from 0 to 76 while hits climb too. That is
tail-trimming doing its job — eviction takes the request-specific end of a node
and leaves the shared head, so what gets reclaimed is precisely the part nobody
was going to match.

The knee is at 80 blocks, where the preamble plus four tails no longer fit
together: preemptions jump to 11 and the win falls to 2.30×. Still well ahead of
uncached, but this is where the pool is genuinely too small for the workload.

**No displacement either**, on the workload built to expose it — every request a
different 1072-token prompt, so each cached block is dead weight from the moment
it is stored:

| blocks | cache | wall ms | hit % | preempt | evicted | |
|---|---|---|---|---|---|---|
| 512 | off | 4625 | 0 | 0 | 0 | |
| | on | 4568 | 0 | 0 | 332 | 1.01× |
| 256 | off | 5404 | 0 | 0 | 0 | |
| | on | 5372 | 0 | 0 | 587 | 1.01× |
| 128 | off | 10506 | 0 | 0 | 0 | |
| | on | 10506 | 0 | 0 | 713 | 1.00× |

713 blocks stored, evicted, and never once reused, for no measurable cost and no
extra preemption. Eviction on allocation failure is doing exactly what §6.3 asks
of it: the cache is as large as it can be right up until the memory is wanted,
and then it is not in the way.

Reproducible to ~2%. Writing this benchmark also found the hit-rate metric to be
wrong: `Acquire` counted every lookup, but admission can match and then fail for
want of room for the *rest* of the prompt, and the scheduler retries on a later
step. Every retry was charged, so under memory pressure — where retries are most
frequent and the number matters most — the reported hit rate fell towards zero
while the cache was working. It reads 88% now where it read 14%. Counting moved
to `RecordAdmission`, called once the admission commits.

#### The M3-to-M4 progression, re-run

Each commit that claimed a number was checked out, rebuilt and re-measured on
current hardware and clocks. The claimed figure is what its commit message
said; the two columns are what it measures today.

| commit | claim | launch-by-launch | with graphs |
|---|---|---|---|
| `35e42aa` FlashInfer wired in | 13.9 ms | 13.46 ms | 18.40 ms (0.73x) |
| `ba3f060` FlashInfer inside the graph | 11.5 ms | 12.03 ms | 10.93 ms |
| `10b4d48` fused QKV and gate/up GEMMs | 11.24 ms | 11.30 ms | 10.59 ms |
| `b6970d9` pinned host copies | 10.73 ms | 11.19 ms | 10.48 ms |
| `8a73665` FP8 available, bf16 path | 10.75 ms | 11.13 ms | 10.49 ms |
| `8a73665` FP8 weights | 8.16 ms | 9.03 ms | 7.89 ms |

The shape holds and the ordering is monotonic, but the claimed figures sit
*between* the two columns rather than on either, so the original measurements
are not exactly reproducible — today's graphed numbers run 2-4% faster than
what was recorded. Driver, thermal state and the R9 fix all changed underneath
them; the discrepancy is small and uniform enough not to disturb any
conclusion drawn from it.

One step does not reproduce. `b6970d9` claimed pinned host staging was worth
11.24 -> 10.73 ms, a 4.5% gain; measured against its own parent today it is
10.59 -> 10.48 ms, **1.0%**. The others hold: FlashInfer-in-graph claimed 17.3%
and measures 18.8%, the GEMM fusion claimed 2.7% and measures 3.1%, FP8 claimed
24.1% and measures 24.8%. The pinned-memory win is the one to treat as
unconfirmed.

Note that `35e42aa` is the commit where graphs made things *worse* — 0.73x,
because a captured graph there fell back to the reference attention kernel.
That is what `ba3f060` fixed, and it is why the graph column is not
monotonic.

#### M9 — what "correct" means without a checkpoint

Neither an MoE nor an MLA checkpoint is on this box, and downloading one would
not help much: Qwen2-MoE-A2.7B and DeepSeek-V2-Lite are both larger than the
16 GB envelope leaves after a KV pool, and the one MoE checkpoint that *is*
cached (gpt-oss-20b) is MXFP4, which is a quantization format this engine does
not implement. So M9 could not be validated the way M2 validated the dense
path — against HF's own logits — and pretending otherwise would be the
expensive kind of dishonesty.

What it is validated against instead is **the definition**, computed on the
host, exhaustively, at a shape small enough to make that affordable:

| property | what is asserted |
|---|---|
| MoE router | softmax + top-k matches a host softmax + top-k, weights to 1e-5; renormalized weights sum to 1; un-renormalized ones do not |
| MoE routing determinism | all-equal logits route to experts 0 and 1 — ties break to the lower index, so routing is a function of the logits and not of scan order |
| MoE dispatch | every assignment appears exactly once, inside its own expert's range; `rows` and `dest` are exact inverses; an expert nobody chose gets an empty range, not garbage; within an expert, rows come out in ascending assignment order |
| MoE layer | the routed mixture equals a host reference that applies **every** expert to **every** token and mixes by gate weight — 2% of the largest output, against bf16 through two GEMMs and a mixture |
| MoE determinism | four identical `Forward` calls produce bitwise identical output |
| MoE shared expert | turning it on changes **every** row, which is what "shared" means and what a shared expert wired into the top-k by mistake would not do |
| MLA layout | `entries_per_token == 1`, `ValueCache()` fails, and the per-token cost equals `KvElementsPerTokenPerLayer()` |
| MLA layer | the full layer matches a host reference of the MLA definition to 3% of the largest output |
| **MLA cache** | **decoding 9 tokens one at a time equals prefilling all 9 at once**, to 2% |
| MLA RoPE | only the trailing `rope_dim` moves; position 0 is the identity |
| MLA paging | append-then-gather round-trips exactly through a **deliberately out-of-order** block table |

The last MLA row is the one worth the most. A slot mapping, a block-table walk
or a causal bound that is off by one produces two paths that each look
plausible on their own and disagree with each other, and nothing but running
both catches it. It is the same property `qwen2_paged_test` asserts for the GQA
path, for the same reason.

Two things this deliberately does **not** claim. There is no throughput number
for either layer: MoE's per-expert GEMM loop and MLA's unabsorbed
reconstruction are both known-slow shapes, and benchmarking them would measure
a decision already documented as temporary. And neither layer is wired into a
`Model`, so "MoE works" means the FFN computes the mixture, not that
`inferx-serve` can load Qwen2-MoE — the loader, the expert-weight stacking from
a real checkpoint, and the dispatch from `Qwen2Model` are all still to do.

#### M10 — head to head against vLLM

`bench/serve_bench.py` driven by `scripts/bench_serve.sh`. The same client
measures every engine over HTTP; each engine is started, measured and stopped
before the next one starts, because both size their KV cache to fill the card
and two live servers would be measuring each other's memory pressure.

Qwen2.5-3B-Instruct, RTX 4080 SUPER, SM clocks locked at 2400 MHz, vLLM 0.26.0.
Both engines: bf16 weights (except the fp8 row), bf16 KV, chunked prefill and
prefix caching on, `max_num_seqs`/`--max-running` 8, 4096-token context. Greedy
everywhere, so both generate the same text and stop in the same place. Prompts
carry a unique random tag *in front*, so neither engine's prefix cache can
answer a request the other had to compute.

| batch 1, 128 tokens out | TTFT | ITL p50 | ITL p99 | decode tok/s |
|---|---|---|---|---|
| inferx bf16 | **16.6 ms** | 10.81 ms | 11.88 ms | 92.5 |
| **inferx fp8** | **16.5 ms** | **7.53 ms** | **8.57 ms** | **131.2** |
| vLLM bf16 | 17.8 ms | 10.44 ms | 11.58 ms | 95.5 |

Output tokens per second with N requests in flight:

| concurrency | inferx bf16 | inferx fp8 | vLLM bf16 |
|---|---|---|---|
| 1 | 93.8 | 128.9 | 95.7 |
| 2 | 177.0 | 249.5 | 181.5 |
| 4 | 345.1 | 465.0 | 365.4 |
| 8 | 670.1 | **842.6** | 719.8 |

Prefill, from a long prompt with `max_tokens=1`:

| prompt | inferx bf16 | inferx fp8 | vLLM bf16 |
|---|---|---|---|
| 515 tok | 8944 tok/s | 10574 tok/s | 10191 tok/s |
| 2054 tok | 10509 tok/s | **13973 tok/s** | 12254 tok/s |

**bf16 against bf16, the two engines are within a few percent of each other on
decode** — 0.97x at batch 1, 0.93x at concurrency 8 — and inferx is ahead on
TTFT (16.6 vs 17.8 ms) and behind on prefill throughput (0.86x at 2k). Decode
tok/s moves ~6% between runs on this box for *both* engines (vLLM measured 90.0
and 95.5 on two runs an hour apart; inferx bf16, 91.7 to 94.0), which is the
unlocked memory clock of R3b and is larger than the gap between them. So the
honest reading of the bf16 rows is a tie on decode, a small real loss on
prefill. That is the result G1 asked for: a torch-free C++ engine reaches a
mature Python one on the same hardware and the same model. It is not a win over
vLLM and is not claimed as one.

**The win is quantization, which is the thing this hardware needed.** FP8
weights are worth 1.42x on batch-1 decode and 1.26x on concurrency-8 throughput
against inferx's own bf16, and 1.37x / 1.17x against vLLM — and 1.33x on
prefill, where FP8 is compute-bound rather than bandwidth-bound. vLLM can serve
FP8 too, and this does not claim otherwise; the point is that the 16 GB
envelope §1 is built around is reachable without leaving C++.

Two defects this benchmark found, both invisible to every microbenchmark in
this repo because both live between components rather than inside one:

- **TCP_NODELAY was off**, so cpp-httplib handed each SSE frame to a kernel that
  held it until the peer's delayed ACK fired. A prompt-length sweep of 6 / 20 /
  62 / 231 / 455 tokens measured 47.3 / 43.9 / 43.8 / 43.8 / 48.1 ms — a floor
  that did not move with prefill work, which is what gave it away. With
  `set_tcp_nodelay(true)` the same sweep reads 13.0 / 13.1 / 16.2 / 28.4 /
  48.0 ms, and batch-1 TTFT went from 25 ms behind vLLM to 1 ms ahead of it.
  Nothing about the engine changed.
- **FP8 activation quantization ran in a single CUDA block.** One block is the
  right call for a decode activation — one launch instead of four, over a few
  thousand elements — and an order of magnitude wrong for a prefill one, where
  a whole step's activations went through one SM. Serving with `--fp8` alone —
  weights quantized, KV cache left bf16, so the fp8 KV path is not implicated —
  measured **3727 tok/s of prefill against bf16's 8925**: quantization made the
  compute-bound path 2.4x *slower*. `gemm_bench` could not see it: it times the
  matmul on operands somebody else quantized, and reports FP8 at 1.9–2.0x fp16
  for exactly these shapes. `QuantizeToF8E4M3Dynamic` now dispatches to the
  existing multi-block amax/quantize kernels above 128k elements, leaving the
  decode path byte-identical — prefill goes 3727 → 14691 tok/s (3.9x) and
  concurrency-8 throughput 818 → 871 tok/s.

Both are one-line-scale fixes for defects that had been shipping since M4 and
M8 respectively, and neither was reachable without an end-to-end serving
benchmark and a second engine to disagree with. That is the argument for M10
existing at all, independent of the numbers it produced.

Method notes, since a comparison is only as good as its fairness:

- Token counts come from each server's own `usage`, never from counting SSE
  chunks — a chunking difference would otherwise read as a throughput
  difference. Getting this required implementing `stream_options.include_usage`
  in inferx-serve, which had been an OpenAI-compatibility gap.
- ITL percentiles are measured from chunk *arrival* times, which is what a
  streaming client experiences, while token counts come from `usage`.
- Each engine gets a keep-alive connection per concurrent worker, so no
  measured request pays TCP setup.
- The prompt for the prefill scenario is sized by asking the server what it
  tokenized, not by a client-side estimate, so both engines are handed the same
  token count rather than the same string.
- vLLM needs two environment overrides on this box, both documented in
  `scripts/run_vllm.sh`: `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (its 0.26 model runner
  allocates UVA buffers, which need pinned memory, which vLLM disables under
  WSL2) and `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's JIT sampler does not
  build against CUDA 13's CCCL — `sampling.cuh` calls
  `cub::BlockAdjacentDifference::FlagHeads`, removed in CCCL 3.x). Neither
  touches the paths being measured at temperature 0.

SGLang is not measured. It is the same client and one more row in
`bench_serve.sh` whenever its install lands.

M1 before M2 is deliberate: the quantized GEMM story is the largest risk in a torch-free design, and discovering it is intractable *after* building the model layer would be expensive.

---

## 15. Open questions

1. **Which arch first — Llama or Qwen?** Qwen2.5-7B at int4 is a better fit for 16 GB and has broader current relevance; Llama has more reference material. Leaning Qwen2.5-7B-Instruct-AWQ as the M2 target.
2. ~~**Quantization format for v1** — AWQ, GPTQ, or FP8?~~ **Resolved: FP8 e4m3 first, AWQ second.** M1's job is to retire R3, not to ship the final quantization story, and FP8 gets there on the shortest path: sm_89 has native e4m3 tensor cores, CUTLASS's SM89 FP8 GEMMs are well-trodden against the SM80 collective builders, and `kFloat8E4M3FN` is already in `DType`. There is no packing and no dequant in the mainloop, so a wrong number is a kernel bug rather than a layout bug. The cost is real and accepted: 2× compression leaves a 7B with roughly 5–6 GB for KV where W4A16 would leave ~11 GB, so **FP8 is the de-risking vehicle, not the endpoint** — AWQ still has to land for §1's 16 GB envelope to work, and it follows in M8 against a harness that already exists.

   **Delivered, and it did not need CUTLASS.** cuBLASLt implements FP8 e4m3 matmul natively on sm_89, so M1 shipped against a dependency the project already had. Two things made that free rather than lucky: cuBLASLt's FP8 path is only implemented for the TN layout, which is exactly what the row-major `y = x·wᵀ` mapping already produces, so storing weights `[out, in]` as checkpoints do turned out to be what makes FP8 reachable at all; and the scales are bound as device pointers per call, so they never round-trip to the host. CUTLASS is consequently **not an M1 dependency** and moves to M8/M9, where mixed-input W4A16 and grouped MoE GEMM genuinely have no alternative. That is roughly a week of integration removed from the critical path, and one fewer 500 MB submodule until it earns its place.
3. **Structured output (JSON schema / grammar)** — a differentiator and a scheduler-visible feature (the logit mask must be computed per step per sequence, on CPU, in the overlap window). Not in the milestone list; needs a decision on whether it is v1 scope.
4. **Determinism guarantee** — do we promise bitwise-reproducible output for a fixed seed and batch composition? This constrains kernel selection (split-k reductions, atomics) and is much cheaper to promise now than to retrofit.
