# Unified cache architecture: contracts and scenario validation

Status: revised target architecture, reviewed 2026-10-09. This is a design and
validation specification, **not an implemented cache manager or a claim of
production qualification**. The numbered cases below are executable-test
specifications and invariant arguments; GPU integration tests remain release
gates. The existing `include/inferx/cache/cache_manager.h` explicitly requires
execution to finish before release and has no asynchronous leases. Nothing in
this document silently changes that implementation contract.

This revision replaces the previous proposal, including its unsafe partial
acceptance, hybrid-prefix, sparse-index, and asynchronous reuse examples.
The scope inventory is the previous document's attention families and operations,
`include/inferx/models/requirements.h`, `docs/speculative_decode.md`, and the
explicit scenarios requested for this review. Upstream features beyond that
inventory are compared where relevant; this is not a claim to implement every
backend or configuration in either upstream project.

## 1. Evidence and terminology

**V** means a mechanism was inspected in vLLM source; **S** means inspected in
SGLang source; **D** means upstream documentation only; **P** means an InferX
proposal. V/S evidence establishes the named mechanism, not the proposed C++ API,
not compatibility of all feature combinations, and not measured performance.
All InferX contracts and case proofs below are P unless explicitly labeled.
Links to `main`/`latest` are mutable, inspected on the review date, not pinned
release promises. Before implementation qualification, archive exact upstream
commits and rerun compatibility checks for the selected model/backend.

| ID | Primary reference and narrowly verified observation |
|---|---|
| V1 | [vLLM cache specifications](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py): distinct full, MLA, sliding, chunked-local, recurrent, encoder-only, cross and R-SWA specifications; unequal K/V geometry and quantization metadata. |
| V2 | [vLLM per-type managers](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py): specialized allocation/hit/skip behavior including local attention and cross attention. |
| V3 | [vLLM hybrid design](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/), D: grouping, heterogeneous memory requirements and intersecting usable prefix hits. The page itself describes an older commit; current source takes precedence. |
| V4 | [vLLM prefix design](https://docs.vllm.ai/en/latest/design/prefix_caching/), D: block hash chaining, full-block publication, contextual hash inputs and eviction. |
| S1 | [SGLang radix cache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py): page-aligned matching, namespace/salt keys, node locks, insert and eviction. |
| S2 | [SGLang memory pools](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/memory_pool.py): request-to-token mappings, MHA/MLA/DSA pools, recurrent parts and intermediate speculative state. |
| S3 | [SGLang MLA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_mla_backend.py): separate extend/decode paths and MHA chunk-cache handling. |
| S4 | [SGLang DSA backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/dsa_backend.py): query selection metadata and logical/physical index transformations. |
| S5 | [SGLang EAGLE worker](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/eagle_worker_v2.py): draft, verify and draft-extension phases, topology metadata, previous-bonus anchor, and result lifetimes. |
| S6 | [SGLang HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design), D: GPU/host/storage tiers, transfer policies and asynchronous I/O. |

The cases reference these IDs locally. Repeated references do not mean additional
upstream claims; the demonstrations are original deductions about this design.
No upstream timing or throughput number is adopted as an InferX guarantee.

## 2. Gaps found and final decisions

| Previous assertion / architectural gap | Required revision | Revalidation |
|---|---|---|
| A recurrent scan's final state also represents any accepted prefix | False: retain an exact intermediate checkpoint or restore the base and replay the accepted inputs | Cases 5, 6, 12 |
| The sampled correction/bonus already has a computed KV row | Separate sampled output length from computed-input length; commit anchor plus accepted proposals only | Case 5 |
| A hybrid can skip 11,200 tokens using only full-attention hits and zeroed recurrent state | Resume only from a complete, compatible cross-group checkpoint; otherwise recompute from an earlier complete boundary | Case 13 |
| Persistent DSA top-k rows are sufficient; watermark alone invalidates them | Store indexer keys/scales persistently; selections are derived views with content, mapping and visibility versions | Case 10 |
| MLA mode is fixed for the engine lifetime | Canonical latent storage plus bounded expanded scratch supports prefill/decode switching; persistent expanded mode is a separate choice | Case 9 |
| Chunk-local always retains historical boundary blocks | Strict chunk-local drops old chunks; boundary retention is a separately declared model mask | Case 8 |
| R-SWA prefix length belongs to the immutable spec | Prompt boundary belongs to each sequence and its reuse identity | Case 7 |
| Flattening segments preserves positions automatically | Carry logical block positions and explicit write slots; validate kernel support for discontiguous ranges | Cases 7, 8, 10 |
| Geometry-equivalent specs can share a pool's layer bundle without further accounting | Intern geometry separately; physical allocation class includes layer-lane count and layout. Different groups allocate distinct units | Case 14 |
| Recording a compute event at free time covers every use | Lease every plan/transfer and join every consuming stream; recycle after last use, not metadata removal | Case 16 |
| Fixed-address staging can be overwritten on the next CPU step | Use completion-gated staging slots, including pinned H2D source buffers | Case 16 |
| Offload frees memory while retaining the same physical BlockId | Separate stable logical BlockRef from device FrameId; detach frames only after transfer completion | Case 17 |
| State byte size is always fp32; snapshots cannot exhaust | Sum aligned per-part dtype bytes; snapshot acquisition is fallible and budgeted | Cases 6, 14 |
| Token IDs alone identify reusable state | Include weights, adapters, positions, masks, source/multimodal identity, format and security namespace | Case 3 |
| Identical requests imply deterministic TP allocation | Rank 0 assigns logical units; reuse requires all-rank completion acknowledgments | Case 18 |
| No D2H or host work per token, zero fragmentation, universal unchanged kernels | Account for sampled token/acceptance feedback, hashing, internal slack, layout adapters and capability checks | Cases 1, 5, 14, 16 |

The final design keeps two storage families, a single scheduler metadata writer,
immutable model plans and generation-checked handles. It introduces explicit
execution leases, prepared mutations, complete prefix checkpoints and recoverable
state transactions. These are necessary semantics, not optional optimizations.

## 3. Model plan, storage and ownership

### 3.1 Static plan

`CacheSpec` describes semantics; `PoolSpec` describes storage. Neither a diagnostic
name nor a model layer number is a semantic interning key. A `GroupId` binds a
semantic spec to an **ordered list of producer layers**. Layers share a block
table only when token mapping, scope, retention, write schedule and commit
boundary match. Sharing tables does not share layer values. Actual cross-layer
KV reuse requires an explicit producer/consumer alias, verified by the model
adapter (Case 22).

Storage planes generalize symmetric K/V: K and V may have unequal dimensions or
dtypes; latent, RoPE, indexer keys and quantization scales can occupy separate
planes. Each pool owns one preallocated device buffer, with checked, aligned
plane offsets. A dense plane is `[layer_lane, frame, token, head, width]`.
State planes are `[layer_lane, slot, rows, cols]`. Packed low-bit layouts describe
bytes and packing explicitly; kernels must support them. No `std::mdspan` or
`std::expected` is assumed: those are not C++20 facilities.

A paged allocation unit bundles `L` layer lanes. Equal geometry with different
lane counts does not imply the same allocation class. A shared allocator may
serve equal-sized bundles of distinct groups: a frame is owned by one bundle at
a time; group-local lane 0 maps to that group's first producer layer. It must
not silently bundle *all layers from all groups* into every allocation, nor let
independent groups write the same unit. Separate pools are the safe default.

Default page size is 16 tokens. It is a validated `PoolSpec` field, not a global
semantic constant: backends that require another size get another pool and graph
signature. Every numerical case uses 16 unless stated. All multiply/add/align
operations use checked size arithmetic; reject impossible shapes before allocation.

### 3.2 Concrete C++20 contract

This declaration block is the proposed public vocabulary, not definitions of
implemented methods. `Status`/`StatusOr` are the existing InferX aliases for Abseil.
`Tensor`, `Stream`, `DeviceEvent` and executor submission are repository adapter
boundaries. The block deliberately uses only standard C++20 declarations plus
`inferx/core/status.h`; it can be syntax-checked independently.

```cpp
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include "inferx/core/status.h"

namespace inferx::cache::design {
using Token = std::int32_t;
using Position = std::int64_t;
using GroupId = std::uint32_t;
using PoolId = std::uint32_t;
using FrameId = std::int32_t; // rank-local device index; -1 is never dereferenced
struct Handle { std::uint64_t id, generation; };
struct SourceHandle { std::uint64_t id, generation; };
struct BlockRef { PoolId pool; std::uint64_t id, generation; };
struct SlotRef { PoolId pool; std::uint64_t id, generation; };
struct Fence { std::uint64_t id; }; // registry-owned event set, not a CUDA pointer
struct TxId { std::uint64_t id; };
struct StepId { std::uint64_t id; };
struct TransferId { std::uint64_t id; };
enum class DType { BF16, FP16, FP32, I32, I64, FP8, Packed4 };
enum class DataKind { KV, Latent, IndexKey, HiddenFeature, Recurrent };
enum class Scope { Sequence, Source, Pack };
enum class WriteKind { Append, WholePack, Once, MutableState };
enum class MaskKind { Causal, Sliding, ChunkLocal, ChunkBoundary,
                      PrefixWindow, Bidirectional, Cross, Recurrent };
enum class Tier { Device, Host, Storage, InTransit };
enum class Recovery { None, Checkpoints, Replay };
struct PlaneSpec {
  DType dtype;
  std::vector<std::int64_t> shape_per_token_or_slot;
  std::uint32_t alignment, packing_version;
  bool operator==(const PlaneSpec&) const = default;
};
struct PoolSpec {
  bool state;
  std::uint32_t page_tokens, layer_lanes, layout_version;
  std::vector<PlaneSpec> planes;
  bool operator==(const PoolSpec&) const = default;
};
struct CacheSpec {
  PoolId pool;
  DataKind data;
  Scope scope;
  WriteKind write;
  MaskKind mask;
  Position window, chunk, boundary_tokens;
  bool prefix_cacheable;
  bool operator==(const CacheSpec&) const = default;
};
struct LayerBinding {
  std::uint32_t layer, lane;
  GroupId group;
  bool read_only_alias; // model-declared, not inferred from shape
};
struct Identity {
  // Full canonical bytes retained/compared; digest is an index, not equality.
  std::string model_weights, adapter, position_scheme, mask_config;
  std::string source_and_multimodal, format_and_topology, security_namespace;
};
struct Range { Position begin, end; }; // half-open, checked, nonnegative
struct Mapping { Position logical_block; BlockRef block; };
struct SourceAttachment { GroupId group; SourceHandle source; };
struct SequenceInit {
  Identity identity;
  Position prompt_boundary;
  std::span<const SourceAttachment> sources;
};
struct AppendInput {
  std::span<const Token> tokens; // copied into manager-owned journal
  std::span<const Position> positions;
};
struct StateBoundary {
  GroupId group;
  Position computed_end;
  SlotRef slot;
  Fence ready;
};
struct Completion {
  StepId step;
  Handle sequence;
  TxId transaction;
  std::uint64_t revision;
  Position computed_end;
  Fence ready; // includes all writers, table uploads and dependent copies
  std::vector<StateBoundary> states;
};
struct SourceCompletion {
  SourceHandle source;
  std::uint64_t build_revision;
  Position computed_end;
  Fence ready;
};
struct CommitChoice {
  Position computed_end; // absolute INPUT boundary, never sampled output count
};
struct SparseStamp {
  Handle sequence;
  std::uint64_t branch, content, mapping, visibility;
  Position query;
  std::uint64_t indexer_version;
};
struct DeviceSpan { std::uintptr_t address; std::size_t bytes; };
struct PagedView {
  GroupId group;
  std::vector<DeviceSpan> planes;
  DeviceSpan frame_ids, logical_blocks, row_offsets, write_slots;
  DeviceSpan query_positions, committed_ends, computed_ends, pack_ids;
  DeviceSpan prefix_ends, window_starts, valid_rows;
};
struct StateView { GroupId group; DeviceSpan slot_ids; std::vector<DeviceSpan> parts; };
struct SourceView { GroupId group; PagedView pages; DeviceSpan source_of_sequence; };
struct WorkId { std::uint64_t id; }; // immutable commands owned by manager registry
struct Prepared {
  Handle sequence;
  std::uint64_t revision;
  WorkId work; // copies/resets/table uploads; executor resolves commands
};
struct StepLease {
  StepId step;
  std::uint64_t staging_slot;
  std::vector<PagedView> paged;
  std::vector<StateView> states;
  std::vector<SourceView> sources;
  StepLease(StepLease&&) noexcept;
  StepLease(const StepLease&) = delete;
  ~StepLease(); // queues cancellation if not submitted; never frees live GPU data
};
struct CheckpointId { std::uint64_t id; };
struct PrefixMatch { Position boundary; CheckpointId checkpoint; };
struct Budget { std::size_t device, pinned_host, scratch, metadata; };
struct ReclaimRequest { PoolId pool; std::size_t desired_free_units; };
struct Stats { std::size_t free, owned, leased, retired, transferred, bytes; };
class Transaction {
 public:
  Transaction(Transaction&&) noexcept;
  Transaction(const Transaction&) = delete;
  ~Transaction(); // noexcept cancellation request, scheduler performs recovery
  TxId id() const;
};
class CacheManager {
 public:
  StatusOr<Handle> Create(std::uint64_t request, const SequenceInit&);
  StatusOr<Prepared> Reserve(Handle, Range next_queries);
  StatusOr<Prepared> Stage(Handle, const AppendInput&);
  StatusOr<StepLease> BuildStepPlan(std::span<const Handle>);
  Status OnSubmitted(StepId, Fence last_use);
  Status Commit(Handle, const Completion&, CommitChoice);
  Status Discard(Handle);
  StatusOr<Transaction> BeginTx(Handle, Recovery);
  StatusOr<Prepared> RecoverTo(TxId, CommitChoice);
  Status CommitTx(TxId, const Completion&, CommitChoice);
  StatusOr<Prepared> RollbackTx(TxId);
  Status FinishRollback(TxId, const Completion&);
  StatusOr<Handle> Fork(Handle parent, std::uint64_t child_request);
  Status MarkIdle(Handle);
  Status Resume(Handle, const Identity&);
  Status Release(Handle);
  StatusOr<PrefixMatch> MatchPrefix(const Identity&, std::span<const Token>);
  StatusOr<Prepared> Graft(Handle empty_sequence, PrefixMatch);
  StatusOr<CheckpointId> PublishCheckpoint(Handle);
  StatusOr<SourceHandle> CreateSource(std::uint64_t id, GroupId,
                                     Position length, const Identity&);
  StatusOr<Prepared> PrepareSource(SourceHandle);
  Status SealSource(SourceHandle, const SourceCompletion&);
  Status AttachSource(Handle, GroupId, SourceHandle);
  Status DetachSource(Handle, GroupId);
  Status RetainSource(SourceHandle);
  Status ReleaseSource(SourceHandle);
  StatusOr<Prepared> StagePack(Handle, Range, std::uint64_t pack_version);
  Status SealPack(Handle, const Completion&);
  StatusOr<TransferId> Offload(std::span<const BlockRef>);
  StatusOr<TransferId> Restore(std::span<const BlockRef>);
  StatusOr<TransferId> Offload(std::span<const SlotRef>);
  StatusOr<TransferId> Restore(std::span<const SlotRef>);
  Status CompleteTransfer(TransferId, Fence, Status io_status);
  Status InvalidateDerived(Handle);
  Status Reclaim(const ReclaimRequest&);
  Status Drain(); // shutdown only, allowed to wait
  Stats GetStats() const;
};
} // namespace inferx::cache::design
```

`Reserve` reserves the read/write working set of **next queries**, not every token
from zero to an absolute capacity. A token journal retains accepted IDs and
positions for hashing/replay even when GPU rows have been evicted. Its host memory
is O(history); an external journal is necessary if host history must be bounded.
`Stage` allocates any COW replacement atomically and returns commands; it does
not pretend that integer metadata mutation performs a device copy. The executor
runs `Prepared.work`, waits on prerequisite fences, then launches model kernels.
Source builders and sequence handles inhabit distinct namespaces.

`StepLease` and `Transaction` are move-only registry tokens. A manager control
block outlives these tokens; shutdown rejects new work and drains registered
leases/cancellation messages before destroying pools. Destructors never launch
fallible restores or throw. Explicit recovery reports errors. An executor reports
submission through a scheduler mailbox; it never mutates allocation metadata.

### 3.3 Runtime records and invariants

Each sequence stores handle generation, identity, `revision`, `branch`, journal,
`committed C`, `staged S`, per-group block mappings or exclusive slots, attachments,
`prompt_boundary`, pending commands and an optional transaction. State slots have
an exact `computed_end`. A block record stores logical owners, execution/transfer
pins, generation, residency, optional device frame and last-writer fence.
A source stores all its layer lanes, length, identity, seal fence and references.
A checkpoint owns required paged mappings and immutable state snapshots at one
boundary, not just an integer length.

1. **I1 — progress:** `0 <= C <= S`; rows `[C,S)` are private. Capacity is a set
   of mappings, not a scalar implying earlier rows are resident. Sampled outputs
   may extend past C; they are inputs only in a later step.
2. **I2 — ownership:** every table/checkpoint/source owner contributes a reference.
   Every submitted use contributes a lease. Reuse requires owners=0, leases=0,
   transfer pins=0 and all last-use fences complete. Allocation generation changes
   before reuse. A copied handle contributes no ownership.
3. **I3 — writes:** a writer has an exclusive target; shared boundary pages are
   copied before writes. Copy covers all producer lanes and associated scales.
   Committed bytes cannot be changed by another sequence or an in-flight step.
4. **I4 — reads:** a kernel may address only mapped, resident rows admitted by
   the per-query mask and computed dependency order. Within a prefill/verify step,
   earlier staged rows are visible to later queries; outsiders see only C.
5. **I5 — atomic boundary:** publication advances all sequence groups together.
   Every recurrent slot must represent that exact boundary; sources remain sealed.
   `Completion` must match handle, revision, transaction, step and write coverage.
6. **I6 — readiness:** host publication can precede physical completion only with
   a fence on every consumer. No kernel sees half a table or half a checkpoint.
   Async device failure poisons affected objects; it does not certify their data.
7. **I7 — retention:** keep the union needed by active queries, next committed
   continuation, rollback base, reusable checkpoints and in-flight consumers.
   Visibility can shrink before physical storage is reclaimed, never conversely.
8. **I8 — reuse identity:** equal token prefixes are necessary but insufficient.
   Canonical context and required group state must match. Hash collisions are
   resolved by equality checks; digest equality alone never authorizes graft.
9. **I9 — derived data:** a selection using logical positions requires valid
   content/visibility identity; physical gathers additionally require mapping
   identity. Cached logits/features have their own provenance and ready fence.
10. **I10 — admission:** reserve all blocks, slots, snapshots, table space and
    scratch before device work. Failure before submission has no visible effect;
    after submission resources retire through leases and the request recovers
    or fails explicitly. Never return submitted memory directly to a free list.

State machine: `Ready(C) -> Reserved -> Staged(C,S) -> Submitted -> Ready(C')`.
A transaction adds `Recovering -> Ready(C')` before publication. `Idle` retains
owners; `Release` invalidates the handle immediately but retires storage later.
Only one mutation/transaction per sequence is active. Independent sequences may
have concurrent steps. Fork, idle, resume and source changes require a stable
boundary; fork during a transaction is rejected. A transaction retains its base
mappings and defers retention trimming, so rollback does not resurrect freed rows.

### 3.4 Prepared mutations and exact commit/rollback rules

The manager keeps an immutable operation record behind every WorkId. It contains
source/destination generations, copy/reset ranges, prerequisite fences and resource
pins. An executor resolves that record into Tensor views and stream operations.
`Reserve` may have no device commands; `Stage` chains any earlier reset/COW commands.
`BuildStepPlan` includes all pending commands for its handles, including fork and
transaction snapshot copies. Discarding a returned `Prepared` value does not discard
registered work or its ownership. Reserve/Stage failures roll back their own
provisional mutation; a later Stage failure does not revoke a successful Reserve.

| Call | Preconditions | Success transition | Failure / ownership rule |
|---|---|---|---|
| `Reserve(h,[C,E))` | Stable or transaction base; E within configured limits | Atomically install enough working-set mappings for proposed queries and recovery | Unsubmitted provisional units return to their pools; existing mappings unchanged |
| `Stage(h,input)` | No previous staged mutation; input starts at C, reservation sufficient | Copy journal input; COW shared write targets; S=E; revision increments | Allocate all replacements before enqueueing any copy; leave C unchanged |
| `Commit(h,done,{E})` | No open tx; completion matches registered lease and every producer; E=S | C=S=E; store producer ready fence; publish eligible immutable pages | Ordinary state updates are all-or-fail; partial commit requires a transaction |
| `Discard(h)` | Append-only update, no destructive state to undo | S=C; invalidate rejected derived data; trim unused pages | Submitted writers retain leases; next overwrite waits for their completion |
| `BeginTx(h,mode)` | Stable boundary; no tx; recovery strategy supported | Pin base mappings; snapshot all mutable state before subsequent writes | Snapshot acquisition is atomic; old state remains intact |
| `RecoverTo(tx,{E})` | Cbase<=E<=S; accepted tokens form a prefix of staged inputs | Produce commands selecting an exact checkpoint or restoring/replaying; sequence becomes Recovering | No published boundary changes; failure preserves base owners |
| `CommitTx(tx,done,{E})` | Matching verify/recovery completion; every state boundary equals E | Atomically C=S=E; journal truncates at E; rejected pages/derived data retire; release base pins after final use | Full acceptance can use verify completion directly; partial mutable-state acceptance must use recovery completion |
| `RollbackTx(tx)` | Open tx, including submitted work | Queue wait-for-writers then state restore; retain base mappings; enter Recovering | Keep request pinned; no new decode may use old metadata with new state |
| `FinishRollback(tx,done)` | Restore completion matches base and all state parts | C=S=Cbase; reinstall base mappings; invalidate derived suffixes; close tx | Failed recovery poisons sequence; never report successful rollback |
| `SealSource(src,done)` | Exact builder generation/revision and all layer/plane writes cover declared length | Immutable source with producer fence; builder reference becomes server retention | A fence alone is not write-coverage evidence; unknown/incomplete completion is rejected |

Completion coverage is established by the executor's registered producer DAG and
successful submissions, not caller-supplied integers alone. A completion object is
a receipt into that registry. The cache manager validates it; a forged or stale
`computed_end` cannot authorize publication. CUDA errors discovered later poison
all dependent objects. A same-sequence next step waits for earlier writes even
if the CPU has already committed their ready-fenced result.

Partial append-only acceptance uses `CommitTx` without state recovery. It closes
the staged range (`S=Cnew`), so rejected rows cannot later be committed by accident.
Prefix-index publication includes **accepted token IDs only**. Rejected proposal
IDs are removed from the journal; the sampled correction remains with the scheduler
until staged. Pages kept for reuse after truncation are private and next writes
must overwrite every row they expose. A full rollback may retain a private COW
copy instead of the old base page only if its committed bytes are identical and
ownership bookkeeping is adjusted; reinstalling base mappings is the default.

All transitions are serialized on the scheduler. In-flight GPU reads are accounted
for independently. Atomic host metadata is therefore sufficient only in conjunction
with the producer/consumer fences in I6; single-threaded metadata by itself does
not establish GPU correctness.

### 3.5 Masks, tables and checkpoints

For a query at absolute position q (window W includes the query):

| Mask | Allowed logical positions at that layer |
|---|---|
| Causal | `[0,q+1)` |
| Sliding | `[max(0,q-W+1),q+1)` |
| ChunkLocal, size H | `[floor(q/H)*H,q+1)` |
| ChunkBoundary | Current causal chunk plus explicitly specified prior boundary tokens |
| PrefixWindow, prompt boundary P | During prompt: model-declared prompt mask; afterward `[0,P) union [max(P,q-W+1),q+1)` |
| Bidirectional pack | All positions in the same fully constructed pack, optionally declared immutable context |
| Cross | Entire sealed source, source padding excluded |

These are token masks, not page masks. Round storage outward to whole pages.
`logical_blocks` preserves positions when packing disjoint segments. A legacy
kernel accepting only a zero-based dense table gets a validated adapter (base
position/segments), a gather into bounded scratch, or `Unimplemented` at startup.
Never reinterpret packed index 1 as logical block 1 after a gap. `write_slots`
separately maps each staged input to frame/offset; read tables alone are insufficient.

A prefix match at C is usable only if every group can resume from C: full history
for full attention, the next-query window for sliding, current chunk for local,
prompt plus tail for R-SWA, and exact state snapshots for recurrent layers.
Checkpoint creation is optional and budgeted. Missing groups shorten the match
to an earlier **complete** checkpoint; no complete hit means C=0. Same-sequence
idle continuation can retain its state without creating a global checkpoint.

### 3.6 Memory and cost model

For a paged bundle, `page_bytes = sum(aligned plane bytes for B tokens) * L`,
with actual alignment applied to offsets rather than counted twice. For state,
`slot_bytes = sum(aligned part regions) * L`, using each part's dtype. Account for
live slots, base snapshots, accepted-boundary checkpoints, prefix snapshots,
retired units, quantization metadata, pinned host mirrors, table rings and scratch.

Admission uses integer, per-pool demand vectors. Size fixed state needs first,
then full-history, bounded-window and source demand under configured concurrency;
round page counts **down** to fit the budget and check remaining bytes. Equal token
capacity across all pools is neither necessary nor efficient for hybrid models.
Unused capacity in one geometry cannot satisfy exhaustion in another. A fixed-size
allocator avoids external fragmentation but still has page tail slack, alignment
padding, lane padding and cross-pool stranded capacity.

CPU work is O(active groups + allocated/evicted units + copied table entries +
new token hashing). A full table rebuild is O(total resident table entries), not
O(active sequences). Differential uploads are an optimization. Async acceptance
and token feedback still transfer small GPU results to the CPU; event polling can
avoid a blocking CUDA synchronization but cannot remove data dependencies.

## 4. Numbered validation cases

All flows use the API in §3. `run(work/lease)` below means executor submission of
prepared commands and model kernels, reporting `OnSubmitted` and a matching
`Completion`; it is explanatory shorthand, not another manager method. Every
case includes a deterministic oracle. Expected outcomes are specifications, not
claims that these tests already pass on GPU. Unless a case changes them, I1–I10,
page size 16 and single-writer metadata apply.

### Case 1 — MHA, GQA and MQA prefill followed by continuous decode

**Scenario.** Serve ragged prompts and decode requests in one batch. V1/S2 verify
paged attention storage mechanisms. Query-head count differs from KV-head count:
MHA uses Hkv=Hq, GQA uses a divisor, MQA uses Hkv=1 per supported rank mapping.

**Execution Flow.** (1) `Create(A)` at C=S=0. (2) `Reserve(A,[0,18))` allocates
blocks 0 and 1; `Stage` journals 18 inputs and sets S=18. (3) Build a lease,
upload write slots for offsets 0..17 and query positions; append each layer's
K/V before that layer's attention reads them. Query 5 sees 0..5, not 6..17.
(4) `Commit(A,done,{18})` publishes C=S=18 after ordering dependencies.
(5) Stage input 18, then commit 19. Independently remove completed B and admit C
without changing A's physical mapping.

**Data Structures.** Causal/KV/Append spec; two page mappings; per-layer K/V
planes, row offsets, absolute query positions, valid lengths and write slots.
For BF16 Hkv=8, dk=dv=128, one layer's page is
`16*8*(128+128)*2 = 65,536` bytes; MQA uses 8,192 bytes.

**Correctness Proof.** I3 grants private writes, I4 provides the causal intra-step
view, I5 requires all layers' coverage through 18. Other requests cannot see A's
staged rows. No fork/rollback is needed for successful ordinary execution;
`Discard` before publication makes staged KV invisible, and shared prefix owners
remain unaffected. Window eviction is forbidden for this full-history spec.
Oracle: compare each query against dense causal attention over exactly its prefix
for all three head configurations; fill padding with NaNs to detect reads.

**Performance Analysis.** Two initial allocations; no new allocation at 18->19;
one tail has 13 unused rows after that step. Table upload and kernel launch cost
scale with batch/table size. KV bytes scale with Hkv, not Hq. No host KV copies;
sampling feedback and events still incur scheduling latency.

**Edge Cases.** Empty prompt uses the model's explicit BOS convention. Reject
negative/overflowing positions and unsupported TP head divisibility. Allocation
failure is atomic; cancellation after submission retires the two pages only after
completion. An asynchronous kernel error fails the request rather than committing.

### Case 2 — Chunked prefill and mixed prefill/decode scheduling

**Scenario.** Prefill 48 tokens in chunks of 17, 15 and 16 while another request
decodes. This is an inference scheduling workflow, independent of local-attention
masks (V3 describes hybrid allocation; the precise schedule here is P).

**Execution Flow.** (1) Reserve/stage `[0,17)`, execute and commit 17.
(2) Reserve/stage `[17,32)`; its first query reads 0..17, including the previous
chunk. Commit 32 and optionally publish a checkpoint. (3) Process `[32,48)` and
commit 48. (4) Build ragged plans containing this chunk and a one-row decode from
another sequence; `row_offsets` distinguish the 16-row and one-row requests.

**Data Structures.** Same full-attention spec as Case 1; journal of 48 tokens,
three block mappings, per-query positions and independent sequence revisions.
A recurrent group, if present, carries its exact state between chunks (Case 12).

**Correctness Proof.** Induct on chunk boundaries: previous committed rows equal
unchunked prefill; causal queries in the next chunk depend on those rows and earlier
rows of that chunk only. Therefore chunking preserves results within the numerical
contract. A failed last chunk leaves C=32 for KV; stateful rollback requires a
transaction. Fork only at committed boundaries. Oracle: compare 17/15/16 with
48-token prefill, including attention at positions 16,17,31,32.

**Performance Analysis.** Bounded activation scratch trades additional launches
and table fills for admission fairness. Full KV remains O(48), unlike a windowed
spec. State snapshots per chunk are optional if failure terminates the request.

**Edge Cases.** A checkpoint at 17 must own/copy a partial boundary page; otherwise
publish only at 16 or 32. A full-prompt cache hit does not automatically supply
last-token logits: retain compatible logits/features or recompute from a supported
earlier checkpoint. Zero-length chunks are no-ops; never overwrite a leased table
when alternating prefill/decode graph buckets.

### Case 3 — Prefix matching, namespace isolation and LRU eviction

**Scenario.** Reuse a 32-token system prompt across requests with different suffixes.
V4/S1 establish prefix indexing and namespace mechanisms; complete cross-group
checkpointing is P.

**Execution Flow.** (1) A commits 35 tokens; publish the full pages at 16 and 32
with their full prefix/context keys. The partial 32..35 page stays private.
(2) `Create(B)`, `MatchPrefix(identity,B_tokens)` returns boundary 32;
`Graft(B,match)` pins and references both pages atomically. (3) B stages its
suffix into a new page. (4) Release A; prefix entries and B still own the first
two pages. (5) LRU removes an idle checkpoint, then release B; only pages with
no other owners or leases retire and eventually become free.

**Data Structures.** A hash chain or radix index, canonical identity, token-journal
slices, checkpoint payloads, per-page refcounts and writer fences. Identity covers
model/weight epoch, LoRA, RoPE/position inputs, attention mask, source versions,
multimodal payload digest, quantization and tenant namespace.

**Correctness Proof.** Equality of the complete deterministic computation identity
and token prefix establishes equal cached values. Collision resolution checks
canonical identity and tokens. Atomic lookup-and-pin prevents eviction between
match and graft; an expired `CheckpointId` fails cleanly. I2 preserves live users
when LRU drops its own ownership. Oracle: same tokens/different adapter, image,
position offset, source version or salt must miss; injected digest collision
must not graft. Compare suffix outputs to a cold run.

**Performance Analysis.** Hashing is O(new tokens); matching/refcounting is
O(matched blocks). Prefix entries consume host memory and retain GPU pages; their
budget must be capped. Recompute saved depends on usable hit length, not token
string overlap alone. Eviction need not synchronize the GPU.

**Edge Cases.** Duplicate publication chooses an existing equal entry, releasing
the losing ownership after its writer completes. Index node removal must not
orphan descendant payload references. Tenant cleanup removes namespace entries
and retires their storage. Sources/pack contexts that are non-causal require
whole-object equality, not token-prefix matching.

### Case 4 — Beam fork and copy-on-write at a partial page

**Scenario.** Branch a 19-token sequence for beam search or parallel sampling;
prefix-index publication normally shares full pages, but fork shares a partial
page too. P COW contract, related to upstream paged/radix sharing (V4/S1).

**Execution Flow.** (1) Parent maps `[0,16)->b0`, `[16,32)->b1`, C=19.
(2) `Fork(parent,child)` copies mappings and increments both references from 1
to 2. Child state slots, if any, are physical copies. (3) Child stages token 19:
allocate b2, copy valid rows 16..18 for every layer/plane, replace b1 with b2,
drop child's b1 reference, append only after copy completion. (4) Parent appends
to b1 after its fork-copy reader is complete; if no other owner, no COW needed.
(5) Commit each branch separately and release the losing beam.

**Data Structures.** Two handles/revisions/branch IDs, distinct tables, shared b0,
private b1/b2, ready fences and a child state-slot bundle. COW copies scales with
payload; shared quantization metadata must not be updated in place.

**Correctness Proof.** Up to 19 the bytes are equal; after 19 each branch has an
exclusive writable page. Copy leases protect b1 even if its logical refcount
becomes 1. Neither branch can overwrite memory still read by an earlier lease.
Rollback drops only the branch's staged suffix. Oracle: alternate parent/child
launch order; their token-19 sentinels and recurrent slots must diverge while
positions 0..18 remain equal.

**Performance Analysis.** Fork costs O(resident mappings), one boundary copy per
writing branch, plus full recurrent slot copies. Full pages incur no GPU copy.
Metadata table copies are not constant-time for long contexts.

**Edge Cases.** Fork at length 16 needs no partial-page copy. Reject fork during
an open transaction or in-flight destructive state mutation. Preflight all child
slots/COW resources; failure must undo refs. Release with pending copy leaves
leases until completion. A stale parent handle cannot fork a reused request ID.

### Case 5 — Speculative chain with partial acceptance and pending bonus

**Scenario.** Target verification with K=4 proposals, anchor convention aligned
with `docs/speculative_decode.md` and S5. The exact commit semantics are P.

**Execution Flow.** (1) C=20 means inputs x0..x19 are computed; x20 is a sampled
but uncomputed anchor. `BeginTx(h,Replay)` pins the base. (2) Reserve `[20,25)`;
stage `[x20,d1,d2,d3,d4]`. Verify writes rows 20..24. (3) Sampling accepts d1,d2
and rejects d3; new correction z is sampled from the appropriate target row.
(4) `RecoverTo(tx,{23})` obtains recurrent state after `[x20,d1,d2]` if needed;
`CommitTx(tx,completion,{23})` sets C=S=23. Rows 23,24 are discarded/invisible.
(5) z is the next uncomputed anchor at position 23; it is **not** the cached d3
row. Next step overwrites position 23 after requisite ordering/COW.

**Data Structures.** Base mappings/state snapshot, five verify write slots,
accepted count a=2, staged token journal, completion revision, optional exact
state-boundary records, separate scheduler sampled-output buffer.

**Correctness Proof.** Causal KV for `[x20,d1,d2]` is identical to ordinary target
execution. The committed increment is `a+1=3` because of the **old anchor**,
not because the newly sampled correction has KV. Rejected rows are excluded by
C, table lengths and invalidated derived selections. All accepted groups publish
at 23 only after recovery. Oracle: a=0 gives C=21; a=4 gives C=25 with a new
uncomputed bonus; K=0 computes only the anchor. Compare cache rows to sequential
target execution, and verify the next token never reads rejected d3's row.

**Performance Analysis.** Verify scratch and reserved KV scale with K+1; rejected
work consumes bandwidth. Accepted count and token IDs require asynchronous D2H
feedback or a GPU-resident scheduler, not a fictional zero-feedback CPU commit.
Partial state replay adds work (Case 6). Whole-page surplus can be trimmed after
leases; a retained partial page has bounded tail slack.

**Edge Cases.** Stop/EOS clips committed *computed* rows independently of emitted
tokens. Reserve/snapshot failure falls back to ordinary decode before submission.
Stale completion or acceptance outside [0,K] is rejected. Abort uses `RollbackTx`
to C=20, not C=21; rollback preserves the anchor as pending scheduler input.
Cancellation waits for verify leases before freeing scratch and pages.

### Case 6 — Recurrent partial acceptance, restore and replay

**Scenario.** GDN or Mamba verification destructively updates conv, matrix and
scalar state through five inputs, but only three are kept. S2 contains explicit
speculative recurrent state storage; this recovery policy is P.

**Execution Flow.** (1) At C=20 reserve a base snapshot R20 of **all** state parts;
copy it before verify writes. (2) Verify produces R25 in a private work slot or
exclusive live slot. (3a) With Checkpoints, choose a produced R23 record with a
ready fence and copy/install it. (3b) With Replay, restore R20 and replay exactly
inputs 20..22 to obtain R23. Replay may rerun the full model using private output
pages, or use retained exact per-layer inputs; merely rerunning a state kernel
without its correct inputs is insufficient. (4) Complete recovery and commit 23;
retire base/intermediate snapshots after all restores finish. For full rollback,
restore R20, report `FinishRollback(tx,done)`, and reset S=C=20.

**Data Structures.** State pool planes with per-part dtype, live/work slot,
transaction base snapshot, optional intermediate slots, `StateBoundary` and
recovery completion. The snapshot includes conv cursor and any time-step scalar.

**Correctness Proof.** For recurrence `R(t+1)=F(R(t),input(t))`, R25 generally
cannot equal R23. A snapshot at 23 or induction over replay from R20 supplies
exactly R23. All layer states and paged rows must agree before I5 publication.
Oracle: scalar recurrence `R'=2R+token`, conv ring and counter sentinels distinguish
every prefix; check accept lengths 0..5 and full rollback. For floating kernels
use declared tolerances or the same execution path if bitwise equality is required.

**Performance Analysis.** A `[32,128,128]` FP32 state is 2 MiB per layer, excluding
conv/scalars. Across 24 such layers a snapshot is 48 MiB; snapshot+restore copies
96 MiB of payload, approximately 192 MiB of read/write traffic. Bandwidth and
launch overhead must be measured. Per-token snapshots cost O(K*state_bytes);
base+replay costs O(state_bytes) storage plus accepted-prefix compute.

**Edge Cases.** `Recovery::None` rejects partial state commit. Snapshot exhaustion
is `ResourceExhausted`, not an invalid slot. A destructor queues recovery and pins
the request until it completes. Failed restore poisons the request; it must not
resume with mixed parts. No state checkpoint is reused until all readers finish.

### Case 7 — Sliding windows, R-SWA and attention sinks

**Scenario.** Bound decode KV while preserving a reference prompt where required.
V1's R-SWA spec preserves the prompt; V2 has local retention managers. P masks
are defined precisely in §3.5, including the current token in W.

**Execution Flow.** (1) Sliding W=20 at C=48 stages q=48, whose mask is [29,49).
Keep physical pages [16,32),[32,48),[48,64); mask out 16..28. (2) Commit 49;
next q=49 needs [30,50). At next q=51 the interval is [32,52), so page [16,32)
can retire if no transaction/checkpoint/lease needs it. (3) For R-SWA P=18,
W=16 at q=80, mask is [0,18) union [65,81). Keep logical pages 0,1,4,5;
drop pages 2,3 only after their last uses. (4) Fork preserves those mappings;
rollback retains the base window until transaction closure.

**Data Structures.** Sliding or PrefixWindow spec; per-request P; segment mappings
and packed frame table paired with logical block numbers `[0,1,4,5]`; per-query
bounds. A learned softmax sink is model/kernel metadata, not necessarily a retained
token. Fixed token sinks use an explicit PrefixWindow mask and identity.

**Correctness Proof.** Every allowed token is mapped; outward page rounding may
retain forbidden tokens but the mask excludes them. P is never inferred from the
current C. Evicting from the final staged endpoint before acceptance is forbidden:
rollback could need the earlier window. Oracle: compare exact allowed-position
sets to a dense mask at q=48..52 and q=80, including positions 17,18,64,65.

**Performance Analysis.** Sliding retention is O(W+B+chunk_size+speculation) per
sequence while work is pending. R-SWA is O(P+W) plus page/lease slack, not O(W)
when prompts are long. Two-segment table adapters avoid history-sized uploads;
a fallback dense hole-filled table has O(context/B) metadata cost.

**Edge Cases.** W=1, W<B, unaligned P, overlapping prefix/window, P=0 and differing
prompt lengths require explicit mask tests. Prefix reuse needs retained continuation
windows, not only prompt pages. Non-causal prompt computation uses Case 11.
Pressure never overrides pins; rejected speculation cannot evict the rollback base.

### Case 8 — Strict chunk-local and explicit boundary-retaining attention

**Scenario.** A hybrid uses causal chunk-local attention, as represented in V1/V2.
Historical boundary retention from the original proposal is a separate P model
extension; it is not the default meaning of upstream chunk-local attention.

**Execution Flow.** (1) Chunk H=32: q=31 reads [0,32); q=32 reads [32,33).
Reserve a prefill chunk `[30,35)` using the union [0,35) because its early queries
still require the old chunk. (2) Execute, commit 35, then retire pages 0 and 1
once no lease/checkpoint needs them. (3) For an explicitly configured boundary
variant retaining the last 2 tokens of each past chunk, at q=70 read
{30,31,62,63} union [64,71); retain their containing pages and mask extra rows.

**Data Structures.** Distinct ChunkLocal/ChunkBoundary specs, H and boundary count,
query positions, segment table, exact boundary mask metadata and checkpoint payload.
A non-page-aligned chunk H=24 needs partial-page masks rather than integer-page
truncation. Shared old/new rows in one page cannot be physically half-freed.

**Correctness Proof.** Union-of-query demand prevents early prefill queries from
losing their context. At a strict chunk boundary, no old tokens are dependencies;
in the boundary variant the declared boundary tokens remain. Oracle: q=23,24,31,32
for H=24 and H=32 against dense masks; ensure the strict variant never reads 31
at q=32. Fork/rollback retains the parent's/base's required chunk independently.

**Performance Analysis.** Strict local live storage is O(H+batch_chunk+B).
Keeping boundaries from every past chunk grows O(history/H) pages; it is not
constant memory. Sparse boundary gather may require a new kernel or scratch.

**Edge Cases.** Reject H=0 and undefined boundary rules at plan build. Speculation
crossing a chunk boundary pins old chunks until accept/reject resolution. An
unsupported mask is rejected before admission. Delayed table consumers keep their
pages leased even after logical trimming.

### Case 9 — MLA absorbed decode and non-absorbed prefill/fallback

**Scenario.** Use compact latent history during decode and expanded attention for
prefill. S2/S3 verify distinct MLA storage and execution paths; P makes their
representation relationship explicit.

**Execution Flow.** (1) Choose canonical latent planes c (rank 512) and positional
key (64). Prefill inputs 0..31, append both once. (2) Expand required latent rows
through model projections into bounded scratch K/V for a non-absorbed prefill
kernel, retaining exact RoPE convention and logical positions. (3) Commit 32 and
release scratch after the prefill lease. (4) Decode q=32 using absorbed query and
value projections over the same latent history. (5) A deployment requiring a
persistent expanded fallback instead selects separate K and V planes and commits
them atomically; changing an existing cache representation requires conversion
and a new format/version, never reinterpreting bytes.

**Data Structures.** Latent/Append/Full spec, separate positional plane, projection
version in identity; temporary expansion arena. With 8 heads, dk=192,dv=128,
BF16 latent bytes/token=1,152 versus expanded=5,120. Unequal K/V widths are legal.

**Correctness Proof.** Both paths evaluate the same model's linear projections and
position encoding; absorption moves the linear operations, not their semantics.
Numerical equivalence requires backend tolerance checks. Canonical committed
latents survive path changes without lossy reverse projection. Oracle: prefill
non-absorbed then decode absorbed vs all-expanded reference, including prefix
hits, fork and partial rollback; compare logits and latent rows separately.

**Performance Analysis.** Compact history saves `(expanded-latent)*tokens*layers`
bytes; expansion costs scratch and projection bandwidth. Chunk expansion must
cover the attention algorithm's required history or use a correct chunked softmax
merge; expanding only the new rows cannot serve arbitrary full-context attention.

**Edge Cases.** Reject missing projection adapter, wrong RoPE position/version,
and unsupported mixed dtype. Speculative scratch remains leased until verification
finishes. Allocation failure selects a prevalidated smaller chunk/fallback. A
sliding MLA group applies Case 7 retention to latent planes identically.

### Case 10 — DSA indexer cache, sparse selections and invalidation

**Scenario.** Sparse attention keeps compressed values and persistent indexer keys;
query-dependent top-k selections are derived. S2/S4 verify those distinct roles.
Persisting selected positions is optional P optimization, not the canonical DSA
cache representation.

**Execution Flow.** (1) At C=32 store latent rows plus indexer-key/scale rows for
0..31 in two groups. (2) Stage q=32; indexer produces logical selections
{2,17,31,32} within q's mask. Resolve them through this lease's mappings, then
sparse attention gathers. (3) Fork: a branch changes future inputs, so its later
query has a new branch/content stamp. (4) COW or restore changes frame mappings:
`InvalidateDerived` increments mapping version and rebuilds physical gathers.
(5) Rollback rejects q=32, increments revision/content/visibility as appropriate
and drops its selections. Recompute a new q=32; never reuse a row just because
its numeric position equals the old one.

**Data Structures.** Latent and IndexKey append groups, quantization side planes,
`SparseStamp`, query-scoped selection buffer, optional memoized logical selections,
physical gather buffer owned by StepLease. Stored indices are int64 logical
positions if maximum context exceeds int32; kernel narrowing must be checked.

**Correctness Proof.** Selection validity requires unchanged query, candidate
content, mask and indexer. Translation validity additionally requires unchanged
mapping. Thus watermark alone is insufficient. Every selected position must be
resident and visible; an absent selected row triggers restore/recompute or error,
never substitution of an unrelated frame. Oracle: choose logical 17, move its
frame, reuse the old frame for a sentinel, and verify gathering still obtains
17; rollback/rewrite at q=32 must change the selection stamp.

**Performance Analysis.** Persistent index keys cost O(history*key_width), while
per-step selections cost O(query_rows*top_k). Memoizing top-k for every historical
query costs O(history*top_k) and is often wasteful. GPU translation avoids D2H of
all selections; generation checks and miss handling need a validated kernel path.

**Edge Cases.** top_k=0, top_k>visible length, ties, invalid/sentinel indices and
mixed branch batches need tests. Full DSA history cannot be window-evicted unless
the model's selection domain also excludes it; retaining only previously selected
rows changes the model. In-flight selections hold leases on gathered frames.
Indexer failure aborts the entire cross-group commit.

### Case 11 — Bidirectional packs, encoder-only work and pack sealing

**Scenario.** Embedding/encoder work uses whole-pack bidirectional attention;
some decoder models consume immutable bidirectional context. V1 distinguishes
encoder-only and cross storage; sealed pack caching is P, not a promise that all
encoder-only kernels use persistent paged KV.

**Execution Flow.** (1) Create a Pack/WholePack group and `StagePack([0,18),v1)`.
(2) For each layer construct all 18 K/V rows before its non-causal attention;
use pack IDs so another packed sequence cannot be attended. (3) `SealPack` after
all layers complete publishes the whole pack atomically. (4) A causal continuation
may attach the sealed context under a separately declared mask. (5) Changing token
17 creates pack v2 and recomputes the pack; it never appends in place to v1.

**Data Structures.** Pack ID/version, whole-pack identity, two pages, per-query
pack boundaries, source/context attachment if retained. Encoder-only outputs may
use ephemeral scratch with no persistent KV allocation at all.

**Correctness Proof.** Later-layer KV generally depends on the whole pack, so
prefix-only reuse is invalid. Seal prevents external readers seeing partial
layer results. Fork shares sealed pages only; abort before seal discards the
entire pack, not a token prefix. Oracle: perturb the final token and compare all
pack outputs with full recomputation; two packs in one batch must be isolated.

**Performance Analysis.** Pack computation needs whole-pack visibility or an exact
tiled attention algorithm; it cannot be converted into causal append by chunking.
Ephemeral encoder scratch avoids needless long-lived cache memory. Whole-pack
reuse can save compute when identity is exactly equal.

**Edge Cases.** Reject partial pack commit, append to a seal, and state recovery
inside a pack without a defined model protocol. Pad rows are masked. Failed
encoder work releases scratch only after its lease; replacing a source version
keeps old readers on the old immutable version.

### Case 12 — Ordinary recurrent decode, idle reuse and independent fork

**Scenario.** Linear attention (including GDN) and Mamba/SSM maintain bounded
state rather than growing KV. S2 verifies recurrent pools and separate state parts.

**Execution Flow.** (1) `Create`, reserve a live slot and enqueue reset of all
parts/cursors. (2) Stage a 16-token prefill; recurrence advances R0->R16; commit 16.
(3) Decode one input, R16->R17, commit 17. (4) `MarkIdle` retains the slot;
`Resume` validates identity and continues from 17. (5) Fork at 17 allocates a
second slot and copies R17; each branch updates independently. (6) Cancellation
of an ordinary destructive update either uses a preexisting transaction to restore
or fails/releases the sequence; `Discard` alone cannot undo state.

**Data Structures.** MutableState/Recurrent spec, exclusive SlotRef, conv ring,
matrix, model-defined scalars and computed-end tag. A pure linear recurrence with
no conv uses only its declared parts. Slots are not prefix-shared mutable storage.

**Correctness Proof.** Reset supplies the model's initial condition; induction on
ordered input processing establishes Rn at boundary n. Copy-before-child-use
establishes independent equal R17 states. Oracle: compare chunked scan and
single-token recurrence; fork, feed distinct suffixes, then restore all parts
including wrap-around cursor using Case 6.

**Performance Analysis.** State memory is O(live sequences*slot_bytes), independent
of context. Fork/snapshot is a full-slot copy; scattered slot copies are gathers,
not necessarily contiguous memcpy despite part-major layout. Host journals still
grow with input history unless spilled or bounded by policy.

**Edge Cases.** Zero-state is not universal: model adapter supplies nonzero initial
state if needed. Only one writer per slot; concurrent sequences get separate slots.
Idle eviction destroys the resume ability unless a checkpoint/offload exists.
Handle generations prevent stale updates to a reassigned slot.

### Case 13 — Hybrid prefix checkpoint and atomic multi-group recovery

**Scenario.** Full GQA + W=16 sliding attention + MLA + DSA index keys + recurrent
state + cross attachment. V3 establishes mixed-group needs; this complete
checkpoint/transaction protocol is P.

**Execution Flow.** (1) A reaches C=64. `PublishCheckpoint` retains full-group
pages 0..3, required sliding tail page(s), latent/indexer mappings, immutable R64
snapshot and source identity. (2) B matches the 64-token prefix, reserves all
required resources, grafts paged refs and copies R64 into B's exclusive slot.
(3) B processes only suffix 64..71 and commits 72. (4) If R64 or the tail window
was evicted, choose an earlier complete checkpoint, e.g. 32, and replay 32..71.
If none exists, recompute from zero. (5) Speculate to 77, accept through 75;
recover recurrent R75, invalidate derived sparse gathers, then publish all groups
at 75 in one scheduler transition.

**Data Structures.** Dynamic group vector (no arbitrary eight-group ceiling),
checkpoint payload per group, source versions, cross-group demand vector and
state-boundary records. Group count is validated against actual executor capacity.

**Correctness Proof.** At resume boundary every future dependency is present and
has the same provenance as a cold execution. Zeroing R64 would violate that
premise, even if full-attention rows hit. Induction over suffix computation then
establishes equivalence. Preflight failure leaves B empty; no partial graft is
published. Oracle: remove one checkpoint component at a time; match must shorten
or fail, never report 64. Inject failure in each group during commit preparation.

**Performance Analysis.** State prefix checkpoints consume additional slots;
checkpoint intervals trade hit granularity for memory. Window storage remains
bounded only if old checkpoint tail owners are evicted too. Atomic commit is
O(groups); replay cost depends on distance from the last complete checkpoint.

**Edge Cases.** Different group page sizes use a common supported checkpoint
boundary or explicitly copied partial pages. Recurrent checkpoint existence does
not imply cached logits. Source changes invalidate downstream decoder reuse.
A failure after submitted work retires every provisional group allocation and
preserves the old complete boundary or fails the whole request.

### Case 14 — Heterogeneous pool sizing, alignment and atomic exhaustion

**Scenario.** Fit full and local layers plus state into a fixed budget without
confusing geometry sharing with content sharing. V3 motivates heterogeneous sizing;
allocation classes and demand-vector admission here are P.

**Execution Flow.** (1) Build pool A: two BF16 GQA layers, 8 heads, dk=dv=128:
page bundle=128 KiB. Pool B: one local layer with the same per-layer geometry:
page=64 KiB. They cannot share an allocation class merely by TokenLayout equality.
(2) State parts: 10 BF16 values=20 bytes and 9 FP32 values=36 bytes; with 16-byte
part alignment, offsets are 0 and 32, slot stride 80 bytes. (3) Subtract live,
snapshot, checkpoint, table and scratch budgets. Choose integer page counts fitting
remaining bytes. (4) Reserve needs A:+1,B:+1,state:+1; inject B exhaustion after
provisional A allocation. Return A immediately only if never submitted; leave
sequence mappings and all free counts unchanged.

**Data Structures.** Interned plane geometry, pool class including lane count,
checked offsets/strides, separate free and retired lists, per-request demand vector.
Stats distinguish owned, leased and retired units rather than adding overlapping
counters as if they were disjoint physical categories.

**Correctness Proof.** Non-overlapping checked byte intervals prevent aliasing;
all-or-nothing metadata installation preserves I10. State size uses dtype bytes,
not sum(elements)*4. Oracle: enumerate every plane/frame address and assert bounds
and non-overlap; budget 1 byte below required must fail; every failure point
restores the initial ownership ledger.

**Performance Analysis.** Fixed allocation has no steady-state device malloc,
but the example state slot has 24 bytes of alignment slack. Tail pages waste up
to B-1 tokens per independently retained interval. Free capacity is not fungible
across pool classes; demand-aware sizing reduces stranded memory.

**Edge Cases.** Integer overflow, zero capacities, odd packed4 widths and excessive
alignment fail startup validation. Snapshot reserve must include actual configured
speculation depth/checkpoint strategy, not a fixed 50% heuristic. Reclaim may find
only retired pages; return backpressure without blocking for the GPU.

### Case 15 — FP8/low-bit caches and scale ownership

**Scenario.** Quantized cache storage reduces memory. V1/S2 expose quantized layouts;
this design must make scale lifetime part of allocation and rollback.

**Execution Flow.** (1) Select a supported FP8 payload plane and explicit scale
planes/granularity at plan build. (2) Stage tokens and write payload plus scales;
commit only with coverage of both. (3) Fork a partial page, copying payload and
all corresponding metadata. (4) If a block-wide scale must change when appending,
COW/requantize the entire private page with a supported conversion, or choose
fixed/per-token scales. Never change a scale shared by previously committed bytes
without a coherent payload rewrite. (5) Offload/restore serializes the same planes.

**Data Structures.** Quantization format/version in PoolSpec and Identity, aligned
scale planes, per-block or per-token scale ownership, conversion scratch leases.

**Correctness Proof.** Each read decodes payload with the scale that produced it.
Rollback excludes rejected payload and metadata; I3 preserves shared blocks.
Oracle: compare dequantized rows before and after fork/append/restore; verify that
modifying child scale metadata cannot change parent values. Approximate numerical
accuracy is tested against model tolerances, not advertised as bitwise BF16 equality.

**Performance Analysis.** Include scale and padding bytes in compression ratio.
Packing/quantization costs, conversion scratch and bandwidth can offset capacity
wins for short prompts. Dynamic page requantization is expensive and may alter
numerics; fixed scales are simpler where model accuracy permits.

**Edge Cases.** Reject unsupported scale granularity/layout at startup. NaN/Inf
and saturation behavior are kernel contracts. Quantization version mismatch must
miss prefix/remote reuse. Outstanding readers prevent in-place requantization.

### Case 16 — Overlapped execution, CUDA graphs, cancellation and ABA safety

**Scenario.** CPU scheduling overlaps compute and copy streams; graph addresses
must remain stable. This is a P lifecycle contract, not a claim that existing
InferX release methods already support overlap.

**Execution Flow.** (1) Build step 100 in staging slot A; pin tables, pinned H2D
source bytes, cache frames and scratch. (2) Submit table copy on copy stream;
compute waits for upload and producer fences, then replays graph A. (3) Scheduler
releases a request before compute completes: handle becomes invalid, owners drop,
but step lease pins remain. (4) Build step 101 in staging slot B, not A.
(5) Join every reader/writer stream into the last-use fence. Poll it; after all
uses complete, release pins, recycle A and retired pages. If the ring is full,
return backpressure rather than overwrite A.

**Data Structures.** Step registry, per-slot captured graph or equivalent stable
binding, immutable table images, upload and last-use events, generation-stamped
frames/handles, completion mailbox. A graph captured with A's addresses cannot
silently replay against B; each ring slot needs compatible graph bindings.

**Correctness Proof.** The event dependency graph orders upload -> kernel ->
last-use -> reuse. No CPU metadata deletion breaks these edges. Completion with
old handle generation or revision cannot publish into a reused request. Oracle:
delay a second-stream reader while compute finishes; free count must stay unchanged
until the delayed reader completes. Poison recycled frames and table slots to
detect early reuse under compute-sanitizer.

**Performance Analysis.** Ring size trades memory for overlap. Event batching
reduces per-unit queries; avoid one event allocation per page. No blocking device
sync is needed in normal admission, but consumer waits and resource dependencies
still affect latency. Graph padding consumes metadata/scratch and must be budgeted.

**Edge Cases.** Unsubmitted lease destruction cancels its plan without a GPU wait.
Submitted cancellation cannot undo launched kernels. All padded graph rows use
valid dummy/sink storage or a kernel-guaranteed no-access mask; `-1` must never
be dereferenced. Shutdown `Drain` waits before freeing pools. Fatal CUDA failure
quarantines the device context rather than recycling uncertain frames.

### Case 17 — Host offload, restore and memory-pressure preemption

**Scenario.** Retain idle context outside GPU memory, analogous to HiCache's tiers
(S6, documentation). P supplies a frame-remapping protocol absent from the old
stub interface. This is an optional extension; unavailable providers return
`Unimplemented` before mutation.

**Execution Flow.** (1) Select an immutable idle BlockRef with no active writer;
pin source frame f7 and host destination, wait for producer, D2H copy.
(2) `CompleteTransfer` validates success and checksum/format; install Host backing,
detach f7 and retire it after transfer/read leases. Logical owners remain.
(3) Another request may now use f7 with a new frame generation.
(4) Restore the original BlockRef into fresh f12; wait for H2D, install mapping,
increment mapping version and invalidate physical gathers. (5) Resume only after
all required blocks/state slots are resident and ready. Without offload, scheduler
preempts by `Release` and later recomputes from a complete checkpoint or zero.

**Data Structures.** Stable logical BlockRef, optional FrameId, tier state machine
`Device -> InTransitOut -> Host -> InTransitIn -> Device`, transfer ticket,
checksum/format and host-buffer lease. Whole recurrent slots follow the same
protocol but are frozen at their exact computed boundary.

**Correctness Proof.** Device memory is freed only after a valid other backing
exists; incomplete transfers never count as usable hits. Kernels receive only
resolved resident frames. I9 repairs derived mapping users after restore. Oracle:
offload, reuse f7 with a sentinel, restore to f12 and compare bytes/outputs; fail
D2H and H2D halfway and confirm no incomplete version is published.

**Performance Analysis.** At least payload/link_bandwidth transfer time plus
staging and events; host capacity and pinned-memory budget can exhaust too.
Synchronous restore on the critical decode path may cost more than recomputation.
Concurrent restores share a transfer if identities match, without duplicating pins.

**Edge Cases.** Cancel tickets idempotently; their buffers live until I/O completion.
Do not offload actively mutated state. Shared immutable blocks may be offloaded
only when all prospective consumers are suspended/restore-aware; refcount alone
is not a scheduling policy. On final release during transfer, retire every backing
once the transfer is cancelled/completed. Eviction must make actual free-frame
progress; dropping metadata for an actively owned block does not reclaim it.

### Case 18 — Tensor-parallel ranks and delayed reclamation

**Scenario.** Rank-local KV geometry with one logical allocation timeline. P extends
the repository's rank-0 block-delta convention; identical request order alone does
not make event-completion timing deterministic.

**Execution Flow.** (1) All ranks report capacities for every allocation class;
rank 0 admits against the per-class minimum and broadcasts plan epoch plus logical
allocation IDs. (2) Each rank binds the IDs to local frames and applies the same
revision-ordered mutation. (3) Local kernels complete at different times.
(4) Rank 0 cannot recycle a logical unit until every rank acknowledges its last
use; alternatively independent physical frame mappings require an explicit
rank-local translation protocol. (5) Broadcast commit/rollback choices, including
accepted boundary and state recovery mode, consistently across ranks.

**Data Structures.** Rank-local plane geometry (including replicated MQA heads),
logical allocation generation, command sequence number, acknowledgment bitmap,
plan/topology fingerprint and transaction revision.

**Correctness Proof.** All ranks compute the same logical token boundary and never
reuse a unit while any shard still consumes its old generation. Local frame IDs
need not be equal only when translations are explicitly represented. Oracle: delay
rank 1's free acknowledgment, request another allocation on rank 0, and ensure
no early reuse. Replay duplicate/reordered commands; reject or idempotently apply
them without changing refcounts twice.

**Performance Analysis.** Reuse may be constrained by the slowest rank; batch
acknowledgments with scheduler traffic. Replicated KV heads and different state
shards require actual rank-local byte accounting, not blind division by TP size.

**Edge Cases.** Worker failure aborts the distributed request/group; partial rank
commit cannot be exposed. Collective/control failures require restart/rebuild,
not optimistic continuation. Context/sequence parallel token sharding needs an
additional distributed-attention adapter and is capability-gated, not established
by this TP proof.

### Case 19 — Cross-attention sharing and source replacement

**Scenario.** Several decoders/beams attend one encoded speech/document source.
V1/V2 verify cross-attention storage; source-handle ownership here is P.

**Execution Flow.** (1) `CreateSource(id,g,18,identity)` allocates two pages across
cross-layer lanes and holds a builder reference. `PrepareSource` yields write
commands/views. (2) Encoder/projection work writes every required row;
`SealSource(handle,completion)` checks coverage and freezes it. (3) Attach A and B:
refs = builder/server retention + two attachments. Their source table row can be
shared; self-attention tables are independent. (4) Fork B attaches the same source.
(5) Detach/release decoders; release the server reference. Retire source pages
only after refs and leases reach zero. Source v2 gets a new generation/identity.

**Data Structures.** SourceHandle distinct from Handle, source-scoped Once spec,
source lengths, `source_of_sequence`, immutable source table and seal fence.
For multiple cross groups, attachment identity includes all required source
versions; downstream decoder prefix identity includes them too.

**Correctness Proof.** All source rows are defined before any dependent cross
read; attached readers wait on the seal fence. Sharing immutable K/V cannot change
self-attention progress. Decoder rollback does not roll back source contents.
Oracle: two branches produce the same cross-attention result for equal queries;
replace source v1 with v2 and ensure old attachments continue using v1 and old
decoder prefixes cannot be reused with v2.

**Performance Analysis.** Memory is O(unique sources) instead of O(decoder beams)
for cross KV. Per-batch source-row deduplication reduces table uploads. Encoder
computation, source projections and self-attention remain separate costs.

**Edge Cases.** Attach before seal returns FailedPrecondition; after logical seal
it waits on readiness. Source-build failure discards all unpublished rows. Reject
silent source changes on a decoder with committed state; create/recompute a new
sequence. Final detach during an in-flight kernel retires, never immediately frees.

### Case 20 — Tree speculation, standalone draft and hidden-state proposers

**Scenario.** Support branching proposals and separate draft/target lifecycles.
S5 verifies topology and phase separation. P baseline uses ordinary forks to
avoid pretending a tree is a linear append.

**Execution Flow.** (1) Target C=16; fork branches for proposals [a,b] and [c,d].
(2) Each child stages its own causal path; batch their queries with separate
rows/tables, or use a capability-validated tree mask and path write slots.
(3) Select path [c,d], adopt its child handle as the continuing target sequence;
release losers. If a fused tree buffer was used, gather accepted path rows into
private canonical pages before commit. (4) Recurrent branches use separate states
or restore/replay the accepted path. (5) Standalone draft has a distinct manager
identity/plan; truncate/recover it to the corresponding accepted inputs, then
process correction/bonus during catch-up. Hidden-feature proposers retain only
their declared features/KV/state with leases; do not assume EAGLE means zero KV.

**Data Structures.** Parent indices, branch IDs, ancestor masks, path-to-row mapping,
per-model handle namespaces, accepted path metadata and optional HiddenFeature
planes whose provenance matches the target boundary.

**Correctness Proof.** A query reads ancestors only, never a sibling at the same
numeric depth. Adopting an ordinary child supplies a valid linear history; fused
compaction establishes the same mapping. Target/draft progress is reconciled
explicitly rather than aliasing their weights' KV. Oracle: siblings contain
distinct sentinels; accepted-path cache and state must equal sequential execution
of just that path, including rejection at its first proposal.

**Performance Analysis.** Fork baseline costs O(branches*mappings) host work and
state copies; fused tree verification saves compute but needs custom masks and
compaction bandwidth. Feature caches and draft KV consume separate budget.

**Edge Cases.** Depth zero uses Case 5's anchor-only step. Reject malformed/cyclic
topology and incompatible vocab/position contracts. Cancelled losers remain leased
until verification completes. A proposer failure can fall back to target-only
execution after restoring a consistent target boundary.

### Case 21 — Disaggregated prefill and remote cache import

**Scenario.** Move a completed prefill to another worker, extending local tiering
motivated by S6. Remote transport and cross-topology conversion are proposed
extensions, not established by the local pool API or current InferX implementation.

**Execution Flow.** (1) Export a complete checkpoint at C=64: canonical identity,
token boundary, group manifest, layout/quantization/topology versions, source
versions, recurrent snapshot and payload checksums. Pin all exports through I/O.
(2) Destination validates manifest and preflights all pools; allocate private
frames/slots. (3) Receive bytes, validate lengths/checksums, convert only through
a registered compatible layout/sharding adapter. (4) After all groups are ready,
install the checkpoint and graft; acknowledge success. (5) Sender releases export
pins after acknowledged completion or safe cancellation. Invalid/incomplete imports
are discarded; normal prefill is the fallback.

**Data Structures.** A provider-owned transfer manifest and ticket around
`Restore`/checkpoint installation; provider APIs must be added before this feature
is enabled. Never serialize raw FrameId or device pointers as portable identity.

**Correctness Proof.** Destination publication has the same complete-boundary
precondition as Case 13. Checksums detect corruption; identity and a trusted/authenticated
transport establish provenance. Checksums alone are not authentication. Oracle:
truncate one group, alter a scale byte, mismatch adapter or TP topology; every
case must fail import without creating a prefix hit.

**Performance Analysis.** Transfer time scales with checkpoint bytes and network
bandwidth; sparse/window/state checkpoints can reduce payload. Conversion, host
staging and duplicate source retention must fit independent budgets.

**Edge Cases.** Retry IDs are idempotent; duplicate acknowledgments cannot double
free. Sender/destination crash aborts incomplete tickets and reclaim uses last-use
fences. Until the provider ABI and its tests exist, startup rejects requested
remote mode; this row records a designed seam, not completed production support.

### Case 22 — Cross-layer KV aliases and hidden-feature caches

**Scenario.** Some model layers reuse another layer's KV; feature-conditioned
proposers retain target hidden outputs. V3 discusses KV sharing and V1 has a
hidden-state cache marker. The ownership/alias rules here are P.

**Execution Flow.** (1) Model adapter declares layer 5 as KV producer and layer 6
as read-only consumer of the exact same representation; validate shape, positions
and mask needs. (2) Allocate one producer lane and one group table; layer 5 writes,
layer 6 waits on its dependency and reads the same plane. (3) Store selected hidden
features in a HiddenFeature group or bounded leased step scratch. (4) Fork shares
immutable producer pages; COW copies each physical producer once, not once per
alias. (5) Rollback invalidates features beyond the accepted computed boundary.

**Data Structures.** Explicit `LayerBinding::read_only_alias`, producer dependency
DAG, distinct producer lanes, feature provenance/version and lease. Pool ownership
counts logical mappings, not the number of kernel consumers; consumers add leases.

**Correctness Proof.** Aliasing is justified by the model computation, never by
equal CacheSpec geometry. Single producer avoids duplicate writes; all consumers
extend last-use lifetime. Oracle: compare alias-enabled execution to an explicit
copy reference, then delay the last consumer and check no early retirement.

**Performance Analysis.** True aliases save payload memory and writes; sharing
only block tables saves metadata. Feature retention increases scratch/bandwidth
and can constrain overlap. Dependency barriers may reduce parallelism.

**Edge Cases.** Reject cyclic producer dependencies, multiple writers and consumers
requiring evicted rows. Different masks can alias only if producer retention is
the union of consumers' requirements. Remote checkpoint manifests include any
persistent features needed to resume the proposer, or force proposer warm-up.

### Case 23 — Failure injection, transactional cleanup and bounded metadata

**Scenario.** Validate the architecture as a service under cancellation, OOM,
invalid handles and faults, rather than only successful inference. Entirely P.

**Execution Flow.** (1) Inject failure at each allocation in Create, Reserve,
Stage-COW, Fork, BeginTx, Graft and source creation. (2) For submitted work, cancel
at upload, copy, compute, recovery and transfer boundaries. (3) Drain cancellation
messages, complete delayed fences, reclaim and release all remaining owners.
(4) Reuse request IDs with new generations and deliver old completions. (5) Fill
prefix/token-journal/table metadata budgets and verify admission/backpressure
rather than unbounded allocation.

**Data Structures.** Ownership ledger with owner IDs, generation registry,
transaction state, cancellation queue, fence DAG, bounded host allocators and
stats. Checked refcount increment must reject overflow; generation wrap requires
retiring the ID namespace or restarting, never silently creating ABA reuse.

**Correctness Proof.** Before submission I10 rolls back provisional allocations;
after submission I2/I6 defer cleanup. Idempotent cancellation consumes each owner
once. Old messages fail generation/revision validation. Oracle: after all owners
are released and fences complete, every pool has its startup free count and no
pending leases/tickets; no operation can make refcounts negative. Device-fatal
faults are checked for quarantine instead of the normal free-count oracle.

**Performance Analysis.** Budget the ledger/events/queues; coalesce work and event
retirement by epoch without losing per-stream dependencies. Detailed debug ledgers
can be disabled in release builds, but counters and invariants remain. Pressure
metrics should separate useful payload, slack, idle retained bytes and retired bytes.

**Edge Cases.** Double release is idempotent only for the same generation's tombstone;
a stale handle never releases its replacement. Explicit rollback failure is surfaced;
destructors report cancellation through a live control block. `Drain` rejects new
work, joins executors/transfers, destroys metadata, then frees storage in that order.

## 5. Coverage matrix

Every requirement below has at least one concrete case. V/S/D evidence refers only
to the upstream mechanism identified in §1. **All proposed InferX support remains
P**; optional extensions are explicitly marked. A matrix entry is a design trace,
not a passed implementation test.

| Attention / representation requirement | Cases | Evidence / disposition |
|---|---|---|
| MHA | 1, 2 | V1, S2; P core |
| GQA | 1, 13, 14 | V1, S2; P core |
| MQA, replicated TP KV heads | 1, 18 | V1, S2; P core, backend validation |
| Sliding window, including MLA window | 7, 9, 13 | V1, V2; P core |
| Strict chunk-local | 8 | V1, V2; P core |
| Historical boundary-retaining local mask | 8 | P explicit model extension |
| R-SWA/reference prompt + recent generated tokens | 7 | V1; P core |
| Token sinks vs learned softmax sink | 7 | V1; P explicit model/kernel distinction |
| MLA absorbed | 9 | S2, S3; P core |
| MLA non-absorbed, unequal K/V, mode transition | 9 | V1, S3; P core, adapter required |
| Sparse/DSA latent and persistent indexer keys | 10 | S2, S4; P core |
| Sparse query selection and physical index invalidation | 10 | S4 mechanism; P version protocol |
| Linear attention, GDN | 6, 12 | S2; P core |
| Mamba/SSM conv, matrix, scalar state | 6, 12 | V1, S2; P core |
| Cross attention | 19 | V1, V2; P source lifecycle |
| Bidirectional packs and encoder-only ephemeral work | 11 | V1 encoder distinction; P sealed packs |
| Hybrid combinations and complete reuse boundary | 13, 14 | V3 D; P checkpoint protocol |
| Quantized and packed caches | 15 | V1, S2; P format capability gate |
| Actual cross-layer KV alias vs shared block table | 22 | V3 D; P explicit producer DAG |
| Proposer hidden features | 20, 22 | V1, S5; P lifecycle |

| Operation / required capability | Cases | Evidence / disposition |
|---|---|---|
| Spec/geometry interning and model-plan composition | 13, 14, 22 | V1/V3 comparison; P rules |
| Create, allocate, Reserve, exhaustion atomicity | 1, 14, 23 | P core |
| Stage, write slots, update, intra-step visibility | 1, 2, 5, 11 | P core |
| Commit and cross-group atomic publication | 5, 6, 13 | P core |
| Discard staged append-only data | 1, 2, 5 | P core |
| BeginTx, partial CommitTx, full RollbackTx | 5, 6, 13 | S2/S5 mechanisms; P protocol |
| State reset, snapshot, checkpoint selection and replay | 6, 12, 13 | S2 mechanism; P recovery choices |
| Handle copying vs Fork, beam branching | 4, 12, 20 | P ownership |
| Prefix publish, match, graft, collision resolution | 3, 13 | V4 D, S1; P complete checkpoint |
| Prefix sharing and partial-page COW | 3, 4, 15 | Upstream sharing; P exact COW contract |
| Source create, prepare, seal, attach/detach, retain/release | 19 | V1/V2 representation; P ownership |
| Pack stage, seal, replacement and abort | 11 | P |
| MarkIdle, Resume, multi-turn reuse | 3, 12 | P |
| Sliding/interior eviction and mask visibility | 7, 8 | V1/V2; P segmented views |
| Prefix/source LRU and pressure reclamation | 3, 17, 19 | S1/S6; P policy/leases |
| Release, generation validation, resource cleanup | 4, 16, 23 | P |
| Fixed pools, slack, aligned mixed dtype state | 14, 15 | S2 layouts; P budget |
| Chunked prefill and continuous batching | 1, 2, 7 | P demonstrated schedules |
| Speculative partial/zero/full acceptance, bonus lag | 5, 6 | S5 anchor mechanism; P boundaries |
| Standalone draft, EAGLE-like features, target catch-up | 20, 22 | S5; P adapters |
| Tree speculation and accepted-path compaction | 20 | S5 topology; P fork baseline/fused extension |
| Sparse invalidation after rollback, remap, eviction | 10, 17 | S4 index transforms; P stamp protocol |
| Async compute/copy, last-use joins, graph staging | 16 | P required production contract |
| CPU/GPU feedback and nonblocking backpressure | 5, 14, 16 | P, no zero-sync-cost claim |
| TP deterministic logical IDs / all-rank reuse | 18 | P distributed protocol |
| Host offload, whole-state swap, restore | 17 | S6 D; P optional provider |
| Recompute preemption | 13, 17 | P fallback |
| Remote/disaggregated import/export | 21 | S6 tier motivation; P optional provider |
| Context/sequence-parallel token sharding | 18 | Explicitly gated extension, not proven TP support |
| Failure, stale completion, cancellation, teardown | 16, 17, 18, 23 | P |
| Memory/performance accounting and production gates | 1–23, §6 | Analytical only until measured |

## 6. Implementation verification and release gates

### 6.1 Concrete test oracles and execution plan

The preceding cases are test specifications. Implement a small CPU reference model
using vectors of token/layer sentinels, exact recurrent scalar/ring transitions,
a deterministic fake fence scheduler, ownership ledger and dense attention masks.
Exercise production metadata code against that model; a second independent oracle
must compute masks and accepted histories directly, not repeat manager logic.

| Suite | Required assertions | Cases |
|---|---|---|
| Geometry/property | byte offsets non-overlapping/in-bounds; all integer overflow rejected; budget never exceeded | 14, 15 |
| Lifecycle/property | random Create/Stage/Commit/Fork/Release/Tx sequences obey I1–I10; allocation failures leave ledger unchanged | 1–6, 12–14, 23 |
| Mask differential | every allowed query/key pair matches dense reference under chunk/window/pack boundaries and staged queries | 1, 2, 7, 8, 11 |
| Numerical differential | cache hit vs cold, absorbed vs expanded, replay vs checkpoint, quantized vs stated tolerances | 3, 6, 9, 10, 13, 15, 19, 22 |
| Scheduler integration | ragged batches, sampled/computed lag, EOS clipping, draft catch-up and accepted tree path | 1, 2, 5, 20 |
| Adversarial async | random stream completion order, delayed H2D sources, graph-slot reuse, cancellation and shutdown | 16, 17, 23 |
| Distributed/provider | delayed rank ACK, retries, partial transfer, incompatible topology and poisoned import | 18, 21 |

Minimum boundary grid: C in {0,1,15,16,17,31,32,33}; W/H/P in
{1,15,16,17,31,32,33} plus P=0; K in {0,1,4,17}; every acceptance a=0..K;
refcounts with 1/2/many owners; each allocation failure point; every relevant
stream completion ordering. Include tiny pools so a one-block leak forces failure.
Full rollback, partial acceptance and source/pack abort are different tests.

Example acceptance-boundary oracle (C++20, independently runnable):

```cpp
#include <cassert>
#include <cstdint>
#include <vector>
int main() {
  for (int k : {0, 1, 4, 17}) {
    std::vector<std::int64_t> state{7};
    for (int i = 0; i < k + 1; ++i)
      state.push_back(2 * state.back() + i + 1);
    for (int accepted = 0; accepted <= k; ++accepted) {
      const int kept_inputs = 1 + accepted; // OLD anchor + accepted proposals
      const int committed = 20 + kept_inputs;
      assert(committed <= 20 + k + 1);
      std::int64_t replay = state[0];
      for (int i = 0; i < kept_inputs; ++i) replay = 2 * replay + i + 1;
      assert(replay == state[kept_inputs]);
      if (accepted < k) assert(replay != state.back());
      // Newly sampled correction/bonus occupies `committed`, still uncomputed.
    }
  }
}
```

This oracle catches the previous document's false recurrent-prefix claim; it is
not a test of a CacheManager implementation. Syntax-checking the declarations is
likewise not evidence that their methods exist or that GPU lifetimes are correct.

Checks performed for this document revision: extracted both C++ blocks;
syntax-checked the public declarations against repository headers with
`c++ -std=c++20 -Wall -Wextra -Werror -fsyntax-only`; compiled and ran the
acceptance-boundary oracle; checked consecutive Case 1–23 numbering and all six
required case fields; independently checked the explicit sliding/R-SWA page sets
and GQA/MLA byte examples. These checks passed. No new cache implementation,
GPU kernel test, distributed test or throughput benchmark was run by this review.

### 6.2 Repository migration

1. Add spec/plan/plane geometry and checked budgeting near `include/inferx/cache/`;
   extend `models::StateRequirements` with semantic masks, producer bindings,
   indexer/latent/quantization/source declarations. Existing geometry-only vectors
   cannot infer these semantics safely.
2. Replace the v0 manager's reservation-only progress with the journal and
   staged/computed boundary protocol; preserve its synchronous execution constraint
   until StepLease and event-aware retirement are implemented and tested.
3. Introduce table/write-slot adapters in `include/inferx/engine/types.h` and
   kernel dispatch. Full dense attention can keep its existing layout; disjoint
   windows, packs, sparse selection and unequal planes need explicit capability
   checks. Do not promise unchanged kernels for every new spec.
4. Add prefix checkpoints, sources, state transactions and recovery. Integrate
   `docs/speculative_decode.md` using input-boundary semantics from Case 5.
5. Enable asynchronous overlap only after graph/table/slot lifetime tests pass;
   extend rank command ordering and ACKs before distributed reuse.
6. Add offload and remote providers behind negotiated feature flags. Missing
   provider/adapter returns an actionable error before admission; never a silent
   fallback that changes attention semantics.

### 6.3 Production qualification criteria

Record peak GPU/pinned/host metadata memory, live versus retired bytes, allocation
failure/retry counts, prefix hit *usable* length, COW bytes, checkpoint/replay bytes,
H2D table bytes, D2H feedback latency, transfer latency and tokens committed per
verify. Benchmark throughput and p50/p95/p99 latency across context lengths,
concurrency, prefix-hit rates, speculative depths and heterogeneous mixes; compare
to a fixed baseline with the same model/backend/numerical settings. Analytical
bandwidth floors are not latency predictions.

Before a capability is called production-supported: its matrix cases must pass
metadata tests, numerical differential tests and applicable GPU sanitizer/soak
checks; no steady-state device allocation, unbounded host growth, premature reuse
or leaked owners may occur under configured limits. Publish the backend/version
capability matrix and measured limits. CPU-only proof cannot establish CUDA graph,
NCCL, transfer or numerical correctness. This revision supplies coherent contracts
and explicit counterexample-driven revalidation; those implementation and hardware
checks remain necessary evidence before shipping.
