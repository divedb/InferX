# Tensor-parallel model abstraction

## Status

Implementation in progress. The engine-facing mechanism is now generalized:

- `ParallelContext` carries rank, world size, placement, and the TP
  communicator into rank-model construction.
- `RankModel` type-erases the per-rank execution surface.
- `TensorParallelRunner` owns model-neutral rank workers, NCCL bootstrap,
  KV-size rendezvous, watchdog/abort handling, telemetry, and rank-zero result
  collection.
- Architecture builders inject a `RankModelFactory`; Qwen2 is an adapter in
  the architecture factory rather than a runner type. There is no
  `qwen2_runner.cc` or `Qwen2Runner` API.

The declared strategy layer has started landing: `TpLayout` and `TpDims`
describe and validate Qwen2's partition, and shard-aware `WeightLoader` verbs
stream individual and fused heterogeneous shards without materializing the
discarded ranks. Qwen2 weight loading now consumes that layout and no longer
contains an imperative axis-selection lambda. Parallel linear/MoE/MLA
primitives have not landed, so Qwen2 still places its direct collective calls
manually, while gpt-oss and DeepSeek-V2 still reject TP greater than one.

Relationship to existing documents:

- `nccl.md` designed the **transport and runtime** for TP=2 (the `Communicator`
  contract, NCCL backend, rank workers, failure propagation, CUDA graphs). That
  layer is implemented and is not redesigned here.
- `docs/ARCHITECTURE.md` §7 states the mathematical partition (Megatron-style
  column/row sharding). That partition is validated with `HostSimComm` and is
  not changed here.
- This document designs the **missing middle layer**: the abstractions between
  `Communicator` and the model implementations, so that TP stops being a
  hand-coded property of `Qwen2Model` and becomes a declared property of every
  model.

## Problem statement

TP support today is real but entirely Qwen2-shaped:

1. There is no common model interface. `Qwen2Model`, `GptOssModel`, and
   `DeepseekV2Model` are unrelated concrete classes; the engine dispatches with
   an `if/else` chain, and the two non-Qwen arms return
   `Unimplemented("tensor parallel serving is currently implemented for Qwen2
   only")`.
2. The sharding rules are imperative and local: a lambda inside
   `Qwen2Model::Load` picks an axis per tensor, and `Impl` methods like
   `LocalHeads()` divide global config dimensions by `comm->size()` at use
   sites. Nothing outside the model can see, validate, or test the layout.
3. Communication placement is manual: Qwen2 calls `comm->AllReduceSum(...)`
   inline at two hand-chosen points per layer. A new model must rediscover
   where collectives go and must reproduce the identical-order-on-every-rank
   discipline by hand.
4. The execution boundary, `QwenRunner`, is named for and typed to
   `Qwen2Model` (including `Qwen2Model::SampledLogprob` in its interface),
   even though its machinery — rank workers, dispatch with watchdog, KV-size
   rendezvous, communicator abort — is model-agnostic.
5. `GptOssModel`, `DeepseekV2Model`, and their layer objects (`MoeFfn`,
   `MlaAttentionLayer`) contain eleven literal `DeviceId::Cuda(0)` placements
   and accept no communicator at all.
6. `WeightLoader`'s pipelined verbs have no sharding support; TP loading falls
   back to the slow `Upload` escape hatch after a host-side `ShardHostTensor`.

The goal is an API where a model author writes the *strategy* (what shards on
which axis, where reductions happen, how the KV cache is distributed) and the
runtime supplies the *mechanism* (groups, ranks, devices, collectives,
sharded loading, failure handling).

## Goals

- Every model implements one common TP interface; the engine and runner are
  model-agnostic.
- Sharding rules are declared, not improvised: the runtime can validate
  divisibility before allocating GPU memory, estimate per-rank memory, verify
  reconstruction in tests, and report the layout in telemetry.
- Communication is encapsulated in reusable layer primitives; model forward
  code never calls `Communicator` directly.
- Model code stays free of backend types (NCCL/MSCCL++), device literals, and
  world-size literals. TP=1 remains the zero-overhead default.
- The abstraction supports any divisibility-valid TP size; the current
  runtime cap of two ranks is a runtime restriction, not an API restriction.
- The API leaves room for pipeline, expert, and sequence parallelism without
  reshaping the model-facing surface.

## Non-goals

- Changing the mathematical partition already validated for Qwen2.
- Implementing pipeline/expert/sequence parallelism now (only leaving seams).
- Multi-node or multi-process execution (per `nccl.md` non-goals).
- Automatic sharding inference from checkpoint shapes. Layouts are authored
  per model, explicitly.
- A PyTorch-style module system. InferX models call kernels directly with
  `TensorView`s; the primitives below follow that idiom.

## Architecture overview

Four layers, from hardware up to the model:

```text
+---------------------------------------------------------------+
| Model implementations (qwen2, gpt_oss, deepseek_v2)           |
|   - author a TpLayout (declared strategy)                     |
|   - compose parallel layer primitives in Forward              |
+---------------------------------------------------------------+
| Parallel layer primitives (model/parallel/)                   |
|   ColumnParallelLinear, RowParallelLinear, TpDims,            |
|   VocabParallelEmbedding*, ParallelMoeFfn, ParallelMla        |
|   - own sharded weight views                                  |
|   - issue their own collectives (the ONLY comm call sites)    |
+---------------------------------------------------------------+
| Parallel runtime (engine + comm)                              |
|   ParallelContext (rank/size/device/communicators),           |
|   ModelRunner (rank workers, dispatch, failure propagation),  |
|   shard-aware WeightLoader verbs                              |
+---------------------------------------------------------------+
| Transport (existing, unchanged in spirit)                     |
|   comm::Communicator: SingleRank | HostSim | NCCL | MSCCL++   |
+---------------------------------------------------------------+
```

The inversion relative to today: communication moves *down* out of model
forward code into the primitives, and layout knowledge moves *up* out of load
lambdas into a declared, queryable structure.

---

## 1. Parallel runtime interface

### 1.1 ParallelContext

The single object a model receives that describes "where am I and who are my
peers." It replaces passing a bare `Communicator` (Qwen2 today) or passing
nothing (the other models).

```cpp
// include/inferx/engine/parallel_context.h  (sketch)
class ParallelContext {
 public:
  // Trivial context: one rank, given device. Preserves all current TP=1
  // callers and is the default everywhere.
  static ParallelContext Local(DeviceId device = DeviceId::Cuda(0));

  // --- Tensor-parallel axis (populated today) ---
  int tp_rank() const;
  int tp_size() const;
  comm::Communicator& tp_comm() const;   // collectives over the TP group

  // --- Placement ---
  DeviceId device() const;               // this rank's device; the only
                                         // source of truth for placement

  // --- Future axes; fixed at size 1 until implemented ---
  int pp_rank() const;  int pp_size() const;   // pipeline
  int ep_rank() const;  int ep_size() const;   // expert
};
```

Rules:

- Models derive **all** device placement from `ctx.device()`. No
  `DeviceId::Cuda(0)` literal may appear in model or layer code (this extends
  the rule `nccl.md` already imposes on Qwen2 to every model).
- `ParallelContext` is mesh-shaped from day one: axes are named (`tp`, `pp`,
  `ep`), each axis has a rank, a size, and (when size > 1) a communicator.
  Adding pipeline or expert parallelism later means populating an axis, not
  changing the model-facing type. A model written against the TP axis is
  untouched.
- Backend selection, bootstrap, and unique-ID plumbing stay behind the
  context's construction (in the runner), exactly as `nccl.md` requires:
  model sources contain no NCCL/MSCCL++ types.

### 1.2 Communicator collective vocabulary

`Communicator` currently exposes only `AllReduceSum`. That is sufficient for
the Megatron dense pattern, but the API must *name* the collectives that
declared strategies will eventually need, so strategies can be expressed
without redesigning the transport contract:

```cpp
class Communicator {
 public:
  // Existing, unchanged semantics (in-place sum, stream-ordered, identical
  // order on every rank; see nccl.md "collective contract").
  virtual Status AllReduceSum(const TensorView&, void* stream = nullptr) = 0;

  // New vocabulary. Default implementations return Unimplemented; each
  // backend implements a collective when its first consumer lands.
  virtual Status AllGather(const TensorView& local, const TensorView& out,
                           void* stream = nullptr);        // vocab-parallel
                                                           // LM head, SP
  virtual Status ReduceScatterSum(const TensorView& in,
                                  const TensorView& local_out,
                                  void* stream = nullptr); // sequence
                                                           // parallelism
  virtual Status Broadcast(const TensorView&, int root,
                           void* stream = nullptr);        // bootstrap /
                                                           // debug paths
};
```

Design decision: **declare the vocabulary now, implement lazily.** This keeps
`nccl.md`'s minimalism (no dead NCCL code paths) while making the strategy
layer expressible. `CommCapabilities` grows matching feature bits so a
strategy that needs `AllGather` can be rejected cleanly at model-load time on
a backend that lacks it, instead of failing mid-forward.

The existing collective contract is unchanged and becomes the contract for
every collective: in-place or explicitly-shaped, stream-ordered, no hidden
synchronization, identical dtype/count/order/communicator on every rank,
tensor residence equal to `communicator.device()`.

### 1.3 ModelRunner (generalizing QwenRunner)

`QwenRunner`'s machinery is already the right runtime shape; only its typing
is wrong. It becomes `ModelRunner`, parameterized by a model factory instead
of hard-coding `Qwen2Model::LoadFromDirectory`:

```cpp
// include/inferx/engine/model_runner.h  (sketch)
class ModelRunner {
 public:
  struct Config {
    // model_dir, devices, backend, kv sizing, graph options... (as today)
  };

  // The factory runs once per rank, on that rank's worker thread, after
  // cudaSetDevice and communicator construction.
  using ModelFactory =
      absl::AnyInvocable<StatusOr<std::unique_ptr<Model>>(
          const ParallelContext&) const>;

  static StatusOr<std::unique_ptr<ModelRunner>> Create(const Config&,
                                                       ModelFactory);

  // Unchanged surface (today's QwenRunner):
  KvBlockPool* kv_pool();
  Status ReserveActivations(int64_t max_tokens);
  Status StepAsync(const model::ForwardBatch&);
  Status AwaitStep(std::vector<int32_t>* sampled);
  Status ReadSampledLogprobs(std::vector<model::SampledLogprob>*);  // shared
                                                                    // type, no
                                                                    // longer
                                                                    // Qwen2's
  Status CaptureDecodeGraph(int64_t num_seqs, int64_t max_blocks_per_seq);
  std::vector<RankTelemetry> telemetry() const;
};
```

Everything proven in the current implementation carries over verbatim as
model-agnostic mechanism: thread-per-rank workers, `DispatchIndexed` with the
30-second watchdog and abort-all-communicators on failure, the KV-size
rendezvous (min across ranks, poison value on failure), rank-0 result
selection, per-rank telemetry. None of it inspects the model beyond the
interface above, so generalizing it is a typing change, not a redesign.

The engine's `if/else` architecture dispatch collapses to choosing a
`ModelFactory`; the per-architecture TP rejection disappears — a model that
does not support a requested layout rejects it itself, with a specific reason,
via its layout validation (§2.2).

### 1.4 Common model interface

The runner needs a minimal virtual `Model` interface — the surface all three
models already expose by convention, plus the TP declarations:

```cpp
// include/inferx/model/model.h  (sketch)
class Model {
 public:
  virtual ~Model() = default;

  virtual const ModelConfig& config() const = 0;
  virtual const TpLayout& tp_layout() const = 0;          // §2

  virtual Status AttachKvCache(int64_t num_blocks, int64_t block_size) = 0;
  virtual int64_t KvBlockBytes(int64_t block_size) const = 0;
  virtual KvBlockPool* kv_pool() = 0;

  virtual Status ReserveActivations(int64_t max_tokens) = 0;
  virtual Status StepAsync(const ForwardBatch&) = 0;
  virtual Status AwaitStep(StepOutputs*) = 0;   // sync models implement
                                                // StepAsync+AwaitStep as
                                                // run-now + return
  virtual Status CaptureDecodeGraph(int64_t num_seqs,
                                    int64_t max_blocks_per_seq) = 0;
  virtual Status AbortCommunication() = 0;      // no-op at TP=1
};
```

This interface exists *because* TP needs a model-agnostic runner; it is
deliberately the intersection of what the engine actually calls, not a
speculative framework. Qwen2-only extras (device sampling, FP8/INT4
quantization) stay on the concrete class and are reached through the factory,
not through `Model`.

---

## 2. Tensor-parallel model abstraction

Each model declares its strategy in two forms that must agree: a **queryable
layout** (data — what the runtime can inspect and validate) and a **composed
forward** (code — the primitives it calls). The layout is the contract; the
primitives are the enforcement.

### 2.1 Partition vocabulary

```cpp
// include/inferx/model/parallel/partition.h  (sketch)
enum class Partition : uint8_t {
  kReplicated,  // full copy on every rank (norms, small routers, embeddings
                // for now)
  kRows,        // shard axis 0 = output features ("column-parallel" linear;
                // Q/K/V, gate/up)
  kCols,        // shard axis 1 = input features ("row-parallel" linear;
                // O, down); forward ends in an all-reduce
  kVocab,       // shard axis 0 over the vocabulary (future embedding/LM head)
};

struct ShardSpec {
  Partition partition = Partition::kReplicated;
  // The sharded dimension must split into tp_size contiguous pieces that are
  // each a multiple of `unit`. Encodes head_dim granularity for attention
  // projections, quant-block granularity for MXFP4/INT4 weights, and
  // interleaving units for fused gate_up layouts.
  int64_t unit = 1;
};
```

Weights are `[out, in]` row-major throughout InferX (`y = x · Wᵀ`), so
`kRows`/`kCols` map exactly onto the axis-0/axis-1 sharding Qwen2's load
lambda performs today. The names are chosen over Megatron's
"column/row-parallel" for the *storage* axis because that is what the loader
and the reconstruction tests actually manipulate; the primitives (§3) carry
the Megatron names for the *math*.

### 2.2 TpLayout: the declared strategy

```cpp
// include/inferx/model/parallel/tp_layout.h  (sketch)
enum class KvSharding : uint8_t {
  kHeads,       // KV pool holds num_kv_heads / tp_size heads per rank (GQA)
  kReplicated,  // KV pool identical on every rank (MLA latent)
};

class TpLayout {
 public:
  // Partition rule for a checkpoint tensor name. Names not listed are
  // kReplicated; listing is still required for every sharded tensor.
  ShardSpec SpecFor(std::string_view checkpoint_name) const;

  // Fails fast, before any GPU allocation, with a message naming the
  // offending dimension: heads % tp, kv_heads % tp, intermediate % tp,
  // quant-block alignment, etc.
  Status Validate(const ModelConfig&, int tp_size) const;

  // How the KV cache scales with tp_size. Drives KvBlockBytes, the KV-size
  // rendezvous, and --gpu-memory-utilization auto-sizing.
  KvSharding kv_sharding() const;

  // Deterministic collective schedule for one decoder layer, e.g.
  // {kAllReduceAfterAttnOut, kAllReduceAfterFfnDown}. Used for validation
  // against the primitives actually invoked, telemetry labeling, and graph
  // cache keys — not for driving execution.
  Span<const CollectivePoint> collective_schedule() const;

  // Per-rank weight-memory estimate at a given tp_size, for admission checks
  // before load ("every selected GPU must have enough free memory for its
  // shard" — nccl.md).
  StatusOr<int64_t> ShardWeightBytes(const ModelConfig&, int tp_size) const;
};
```

What this buys, concretely:

- **Which layers support TP** is now an inspectable fact (`SpecFor` +
  `collective_schedule`), not folklore.
- **Validation before allocation**: today Qwen2 checks divisibility inside
  `Load`; DeepSeek and gpt-oss check nothing. `Validate` makes the check
  uniform, early, and specific.
- **Honest-TP testing** (ARCHITECTURE.md §7.4) becomes generic: a single test
  can load any model at TP=1 and TP=N under `HostSimComm`, reconstruct each
  sharded tensor via `ReconstructHostTensor` using `SpecFor`, and diff
  logits — no per-model test authoring for the layout itself.
- **KV auto-sizing** stops guessing: `ModelConfig::KvElementsPerTokenPerLayer`
  currently documents MLA's replication in a comment; `kv_sharding()` makes it
  an input the runner's rendezvous can trust for any model.

### 2.3 TpDims: derived local dimensions

Replaces the ad-hoc `Impl::LocalHeads()`-style methods with one validated
helper, computed once at load:

```cpp
struct TpDims {
  // Validates divisibility (delegating to TpLayout::Validate) before
  // returning; a model never divides a config dimension by tp_size anywhere
  // else.
  static StatusOr<TpDims> For(const ModelConfig&, const TpLayout&, int tp_size);

  int64_t local_heads;
  int64_t local_kv_heads;
  int64_t local_q_dim;          // local_heads * head_dim
  int64_t local_kv_dim;         // local_kv_heads * head_dim
  int64_t local_intermediate;   // dense FFN
  int64_t local_moe_intermediate;
};
```

`ModelConfig` itself stays global and TP-unaware — it describes the
checkpoint, not the placement. This preserves the existing rule that all
ranks parse identical config and derive locality, which is what makes the
"identical collective order on every rank" discipline checkable.

---

## 3. Layer-level parallel primitives

Models compose these instead of writing communication logic. They follow the
established InferX layer idiom (`MoeFfn`, `MlaAttentionLayer`): plain structs
or pimpl classes holding `TensorView`s, taking `CublasLtGemm*` and
`cudaStream_t` per call, returning `Status`. They are *not* modules with
their own allocators or streams.

**These primitives are the only call sites of `Communicator` collectives in
the entire model layer.** That single rule is what makes collective ordering
deterministic by construction: the schedule falls out of the (fixed) order of
primitive calls in a model's forward, identically on every rank.

### 3.1 ColumnParallelLinear

Output features sharded (`kRows` storage). No communication; output is
sharded on the feature axis and stays local.

```cpp
struct ColumnParallelLinear {
  TensorView weight;                 // [local_out, in]
  std::optional<TensorView> bias;    // [local_out]
  Status Forward(ops::CublasLtGemm& gemm, const TensorView& x,
                 const TensorView& y_local, cudaStream_t stream) const;
};
```

Used for: Q/K/V projections (unit = head_dim), gate/up projections, MLA
q_b/kv_b projections, MoE expert gate_up (per expert).

### 3.2 RowParallelLinear

Input features sharded (`kCols` storage). Consumes a feature-sharded
activation, produces the full-width output via **all-reduce, issued inside
the primitive**:

```cpp
struct RowParallelLinear {
  TensorView weight;                 // [out, local_in]
  std::optional<TensorView> bias;    // [out]; added after the reduction so
                                     // it is applied exactly once
  Status Forward(ops::CublasLtGemm& gemm, comm::Communicator& comm,
                 const TensorView& x_local, const TensorView& y,
                 cudaStream_t stream) const;
};
```

Used for: attention O projection, FFN down projection, MoE combine output,
MLA output projection. The bias-after-reduce rule matters for gpt-oss, whose
projections carry biases; Qwen2's O/down have none, so its numerics are
untouched.

At TP=1 the communicator is `SingleRankComm` and the collective is the
existing zero-cost no-op — the primitive has no single-GPU overhead and no
`if (tp)` branches in model code.

### 3.3 Attention head ownership

Attention itself needs no new primitive: with Q/K/V column-parallel at
head_dim granularity, each rank runs the existing attention kernels
(FlashInfer, paged, MLA) over `TpDims::local_heads` / `local_kv_heads`, and
per-head state shards with the heads automatically. This includes gpt-oss
attention sinks (one scalar per head → sharded with `kRows`, unit 1, on the
head axis) and each rank's KV pool (already parameterized by "KV heads on
this rank" in `core/kv_cache.h`).

### 3.4 Embedding and LM head

Decision: **replicated for now, vocab-parallel later.**

Today both are replicated on every rank, and sampling runs identically on
every rank from all-reduced logits with rank 0 read back. That is correct and
simple, and for the current model sizes the embedding/LM-head memory cost is
acceptable. Vocab-parallel embedding (masked lookup + all-reduce) and LM head
(local logits + all-gather, or distributed top-k) are specified as `kVocab`
in the partition vocabulary and depend on the `AllGather` collective; they
are the first consumers that will force its NCCL implementation. The
device-sampling path complicates distributed argmax, so vocab parallelism is
scheduled after the abstraction lands, as a memory optimization with its own
differential test — not bundled into the migration.

### 3.5 ParallelMoeFfn

Follows the plan already written into `moe_ffn.h`: **TP over expert
weights** — every rank holds all experts with `moe_intermediate / tp_size`
per expert; routing is replicated (router weights `kReplicated`); dispatch,
grouped GEMMs, and combine run on the narrowed dimension; one all-reduce
after combine, plus the shared-expert contribution folded in before the
reduction. `MoeFfn::Config` gains the local intermediate width from `TpDims`;
the collective is issued by the layer, mirroring `RowParallelLinear`.

Quantized expert weights (gpt-oss MXFP4) constrain the shard boundary to the
quantization block size — expressed as `ShardSpec::unit`, validated by
`TpLayout::Validate`, enforced by the loader.

Expert parallelism (experts partitioned across ranks, all-to-all dispatch) is
explicitly a *different* strategy on a *different* axis (`ep`), not a mode of
this primitive. The seam is that `ParallelMoeFfn` reads its expert set and
widths from the context and layout; an EP layout would list experts under a
future `kExperts` partition and use the `ep` axis communicator.

### 3.6 ParallelMla

Follows the constraint documented in `mla.h`: the **latent KV is replicated**
(`KvSharding::kReplicated` — KV memory does not fall with tp_size; this is a
property of MLA, not a limitation of the implementation). Sharding is over
attention heads on the non-latent parameters:

- `q_a_proj`, `kv_a_proj` (into the latent): replicated — their outputs are
  the shared latent every rank needs.
- `q_b_proj`, `kv_b_proj` (out of the latent, per-head): column-parallel,
  unit = the per-head output width.
- `o_proj`: row-parallel → all-reduce, via `RowParallelLinear`.

Each rank runs the absorbed/unabsorbed MLA paths over its local heads against
the full replicated latent cache. Communication per layer is unchanged from
the dense pattern: one all-reduce after attention output, one after the FFN
(dense or MoE).

---

## 4. Model-specific strategies

Each model ships a `TpLayout` capturing exactly this table; forward passes
compose §3 primitives. Per-layer communication is uniformly two all-reduces
(after attention output, after FFN down/combine) across all three
architectures — a useful invariant for telemetry and graph keys.

### Qwen2 (already validated; refactor is a numerical no-op)

| Tensors | Partition | Unit |
|---|---|---|
| q/k/v proj + biases | kRows | head_dim |
| o_proj | kCols | head_dim |
| gate/up proj | kRows | 1 |
| down_proj | kCols | 1 |
| embed_tokens, lm_head, all norms | kReplicated | — |

KV: `kHeads`. Collectives: all-reduce ×2 per layer. The refactor's acceptance
gate is bit-identical HostSim TP behavior versus the current inline
implementation.

### GPT-OSS

| Tensors | Partition | Unit |
|---|---|---|
| q/k/v proj + biases, sinks | kRows | head_dim (sinks: 1, head axis) |
| o_proj + bias | kCols | head_dim |
| router | kReplicated | — |
| expert gate_up (MXFP4 blocks + scales) | kRows (within each expert) | fused-pair × quant-block |
| expert down (MXFP4 blocks + scales) | kCols (within each expert) | quant-block |
| embed, lm_head, norms | kReplicated | — |

KV: `kHeads` (sliding + full attention layers both shard by head).
Collectives: all-reduce ×2 per layer (o_proj; MoE combine, shared expert
included pre-reduce).

### DeepSeek-V2 (MLA + MoE)

| Tensors | Partition | Unit |
|---|---|---|
| q_a_proj, kv_a_proj(+mqa), a-layernorms | kReplicated | — |
| q_b_proj, kv_b_proj | kRows | per-head width |
| o_proj | kCols | per-head v width |
| dense mlp gate/up, expert gate_up, shared-expert gate/up | kRows | 1 |
| dense mlp down, expert down, shared-expert down | kCols | 1 |
| router | kReplicated | — |
| embed, lm_head, norms | kReplicated | — |

KV: `kReplicated` (latent). Collectives: all-reduce ×2 per layer. The layout
declaration is also where the "KV does not scale with TP" fact becomes
machine-readable for the KV rendezvous instead of a comment on
`KvElementsPerTokenPerLayer`.

---

## 5. Weight loading

`WeightLoader` verbs gain an optional `ShardSpec` + TP coordinates so sharded
loading uses the fast pipelined path instead of the `Upload` escape hatch:

```cpp
// Sharded overloads; shape parameters are the GLOBAL checkpoint shapes, and
// the returned view has the local shape. Validation reuses
// TpLayout::SpecFor via the model's load code.
StatusOr<TensorView> Load(std::string_view name, const Shape& global,
                          const ShardSpec&, int tp_rank, int tp_size);
StatusOr<TensorView> LoadStacked(Span<const std::string> names,
                                 const Shape& part_global,
                                 const Shape& out_global,
                                 const ShardSpec&, int tp_rank, int tp_size);
```

Design points:

- `kRows` shards are contiguous row ranges of the mmap'd source — the
  existing extent-streaming path applies directly with a narrowed extent, and
  no staging copy of the discarded rows is ever made.
- `kCols` shards are strided; the pack step (worker threads packing pinned
  slots) performs the strided gather it already performs for other layouts.
- `ShardHostTensor`/`ReconstructHostTensor` remain the host-side reference
  implementation and the test oracle: a loader test shards via the fast path
  and via `ShardHostTensor` and requires identical bytes.
- `Upload` remains the escape hatch for genuinely transformed weights
  (quantization repacking), unchanged.
- Per-rank loaders run on their rank threads as today; sharing a read-only
  checkpoint mapping across ranks is already the design in `nccl.md`.

---

## 6. Synchronization and failure model

Unchanged from `nccl.md`, restated as obligations on the new layers:

- **Ordering**: collectives are issued only by primitives, in the model's
  fixed forward order, on the model stream. Every rank executes the same
  primitive sequence because every rank runs the same model code over the
  same `ForwardBatch`. `TpLayout::collective_schedule()` lets the runner
  cross-check the declared schedule against communicator metrics in debug
  builds.
- **Step atomicity**: the runner commits scheduler state only when all ranks
  succeed; a failing rank aborts every communicator (existing
  `DispatchIndexed` behavior, now model-agnostic).
- **Graph capture**: gated on `capabilities().cuda_graph_capture` per rank;
  graph keys include tp_size and backend (already implemented for Qwen2;
  becomes the `ModelRunner` default). New collectives added to the vocabulary
  must document their capture-safety before any primitive uses them inside a
  captured region.
- **Sampling**: all ranks sample identically from all-reduced logits; rank 0
  is read back. This is the implemented behavior (vs. ARCHITECTURE.md §5.3's
  rank-0-only sampling) and this document makes it the specified behavior —
  it costs a replicated sampling kernel but keeps every rank's step identical,
  which is what the ordering discipline and graph capture want.

## 7. Design decisions

| # | Decision | Alternatives rejected | Why |
|---|---|---|---|
| 1 | Strategy = declared `TpLayout` (data) + composed primitives (code), required to agree | (a) Pure imperative (status quo): unverifiable, non-reusable. (b) Full declarative graph/planner: alien to InferX's direct-kernel style, large speculative machinery | Data gives validation/telemetry/tests; code keeps the hot path explicit and idiomatic |
| 2 | Collectives issued only inside layer primitives | Models call `Communicator` directly (status quo) | Deterministic cross-rank ordering by construction; new models cannot misplace a reduction |
| 3 | `ParallelContext` with named axes (tp/pp/ep), TP populated first | Bare `Communicator` parameter; global singletons | Placement + topology in one injectable object; PP/EP become new axes, not new model APIs |
| 4 | Extend `Communicator` vocabulary now (`AllGather`, `ReduceScatterSum`, `Broadcast`), implement per-backend only when consumed | Only-what's-used (blocks expressing strategies); implement-everything (dead untested transport code) | Strategies are writable today; transport stays minimal; capability bits give clean early rejection |
| 5 | Common `Model` interface + `ModelRunner`; kept to the intersection the engine actually uses | Per-model runners (status quo); rich framework base class | The runner machinery is already model-agnostic; only typing blocks reuse |
| 6 | `ModelConfig` stays global; locality only via `TpDims` | Rank-local config rewriting | Identical config on all ranks keeps rank symmetry checkable; one validated division site |
| 7 | KV sharding is a declared model property (`kHeads` vs `kReplicated`) | Infer from head counts; special-case MLA in the runner | The runner must size KV correctly for models it knows nothing about |
| 8 | Embedding/LM head replicated now; `kVocab` specified for later | Vocab-parallel immediately | Needs `AllGather` + distributed sampling rework; memory win doesn't justify coupling it to the migration |
| 9 | MoE TP = shard every expert's intermediate; EP is a future separate axis | EP first | Matches `moe_ffn.h`'s documented plan; no all-to-all, reuses the two-all-reduce pattern; EP seam preserved via `ep` axis |
| 10 | Replicated sampling with rank-0 readback is the spec | Rank-0-only sampling (ARCHITECTURE.md §5.3) | Keeps every rank's step stream identical — simpler ordering and capture; codifies implemented behavior |
| 11 | API supports any divisibility-valid tp_size; runtime keeps its TP≤2 cap for now | Bake 2 into the API | The cap is a validation-hardware constraint (`CLI::Range(1,2)`, NCCL runner's two-device check), not a design one; lifting it later must not touch model code |

## 8. Adoption sequence (design-level, not a work plan)

1. **Extract without behavior change — partial**: introduce `ParallelContext`, `TpDims`,
   `ColumnParallelLinear`/`RowParallelLinear`, shard-aware loader verbs, and
   Qwen2's `TpLayout`; rewrite Qwen2's load/forward onto them. Gate: HostSim
   TP differential vs. current Qwen2 is exact; TP=1 perf unchanged.
2. **Generalize the runner — complete**: rank-model interface + tensor-parallel runner; engine
   dispatch becomes factory selection. Gate: Qwen2 serving behavior unchanged
   at TP=1 and TP=2.
3. **DeepSeek-V2 TP**: device-literal removal (5 in `deepseek_v2.cc`, 4 in
   `mla.cc`), `ParallelMla`, MoE sharding, `kReplicated` KV declaration.
4. **GPT-OSS TP**: device-literal removal (6 in `gpt_oss.cc`, 2 in
   `moe_ffn.cc`), sinks/bias handling, MXFP4 shard-unit validation.
5. **Restore the TP test suite**: `communicator_test.cc` and
   `qwen2_tp_test.cc` were deleted in the async-HTTP commit (`99e5ed3`) and
   never restored, while `scripts/validate_tp.sh` still invokes
   `communicator_test`. The generic layout-reconstruction and HostSim
   differential tests from §2.2 replace and supersede the per-model versions.
   This is a prerequisite for calling any of the above validated.
6. **Later, independently**: vocab-parallel embedding/LM head (first
   `AllGather` consumer), physical multi-GPU validation per
   `docs/TP_VALIDATION.md`, MSCCL++ evaluation per `nccl.md`, then PP/EP on
   their own context axes.

## 9. Open questions

- **Distributed sampling for vocab-parallel LM head**: all-gather full logits
  (simple, bandwidth-heavy) vs. local top-k + gather (cheap, complicates
  penalties/logprobs which need full-vocab views). Decide when `kVocab` is
  scheduled; the partition vocabulary does not depend on the answer.
- **Sequence parallelism for norms/residuals**: the `ReduceScatterSum` +
  `AllGather` pair replaces all-reduce when activations dominate. Named in
  the vocabulary; no primitive is designed for it yet.
- **Per-rank checkpoint mmap sharing**: one shared read-only `Checkpoint`
  across rank threads vs. per-rank instances — a load-time memory/perf trade
  the loader work should measure, invisible to the model API either way.
- **`--tensor-parallel-size` vs `devices.size()`**: today world size is
  effectively `devices.size()` and the flag is only cross-checked downstream.
  The `ModelRunner` config should make one of them authoritative and validate
  the other at parse time.
