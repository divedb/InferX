# KV cache subsystem: vLLM reference analysis and C++ design

Research and code inspection: **2026-10-16**. This is a design proposal, not an
implemented subsystem. Everything marked *exists* describes code in the tree
today; everything else is proposed. §6 gives the detailed class interfaces and
§7 the implementation plan; the backlog and per-mechanism verification cases
live in `TODO.md` (items C/F/A/P, matrix T04). This document is the
architecture behind them and does not repeat their acceptance criteria.

## 1. Decision

**Adopt vLLM V1's decomposition — cache specification → groups → per-group
managers → coordinator → one shared block pool — re-expressed over InferX's
existing, tested components, evolving them rather than rewriting them.**

Concretely:

1. Keep `KvBlockPool`, `CacheManager`, `RecurrentStatePool`, and the
   scheduler's driving loop as they are. Their semantics survive; the pool
   gains vLLM's block ledger (ref counts, cached state, LRU queue) so that
   prefix sharing, eviction, and future fork/copy-on-write run through one
   ownership counter. The old `PrefixCache`/`BlockHashChain` layer is deleted:
   its index is absorbed into the pool (§6.2) and its chain into `ChainHash`
   (§4.5).
2. Add the missing layers: a `CacheSpec` hierarchy describing what each layer
   needs, a grouping step that turns per-layer specs into cache groups, one
   `CacheGroupManager` per group holding that group's lifecycle policy
   (full attention, sliding window, recurrent state), and a `CacheCoordinator`
   that reconciles cache hits across groups for hybrid models.
3. Scope for this design: one GPU rank's local view (tensor-parallel ranks
   already share logical block ids through `ModelExecutor::KvPool`), CPU pools
   so all bookkeeping is unit-testable, no KV connectors or offload, no
   per-group page sizes, no dynamic pool growth. All are future extension
   points (§9), none are silently implied.

The strongest reasons for this route:

- InferX's current cache is already the vLLM V1 shape in miniature: one
  allocation, int32 block ids, chained SHA-256 block hashes over full blocks,
  LRU eviction under pressure, recompute-style preemption. What is missing is
  exactly what vLLM V1 adds over a flat block table: per-group lifecycle
  (sliding window retention, recurrent state) and multi-group reconciliation
  for hybrid models.
- The T04 parity matrix shows the gaps are concentrated: full attention is S;
  sliding window, hybrid, MLA, and recurrent are R today because storage
  exists but no lifecycle policy does. A manager-per-group architecture is
  the minimal structure that closes them without coupling the scheduler to
  attention mechanics.
- The alternative — generalizing `CacheManager` itself with window and
  recurrence flags — is the "monolithic manager" the task rules out and
  reproduces the inheritance of complexity vLLM shed in 2025.

## 2. Reference architecture: vLLM

### 2.1 Sources

- **Primary: the local checkout** `/home/gc/vllm`, HEAD
  [`b8b302cde4`](https://github.com/vllm-project/vllm/tree/b8b302cde434df8c9289a2b465406b47ebab1c2d)
  (2025-10-10), inspected directly. All file paths and class names below were
  verified against this tree. `TODO.md`'s pinned anchor `5d5c3d7…` is not an
  ancestor of this HEAD; where the two could differ, this document follows the
  checkout.
- **Secondary: upstream `main`**, synced 2026-10-15 (verified via web up to PR
  #58585), used only to confirm the design does not bake in APIs that upstream
  has since restructured. Differences that matter to this design are listed in
  §2.9.

### 2.2 File map and responsibilities

| vLLM file (`vllm/v1/…`) | Responsibility |
| --- | --- |
| `kv_cache_interface.py` | Spec dataclasses (`KVCacheSpec`, `AttentionSpec`, `FullAttentionSpec`, `MLAAttentionSpec`, `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`, `MambaSpec`, `CrossAttentionSpec`, `EncoderOnlyAttentionSpec`, `UniformTypeKVCacheSpecs`), `KVCacheTensor`, `KVCacheGroupSpec`, `KVCacheConfig` |
| `core/kv_cache_utils.py` | `KVCacheBlock` (id, `ref_cnt`, hash, intrusive free-list links), `FreeKVCacheBlockQueue`, block-hash chaining, `resolve_kv_cache_block_sizes` |
| `core/block_pool.py` | `BlockPool`: one id space, free queue, `BlockHashToBlockMap` (hash → blocks, duplicates coexist), get/cache/free/evict, reset |
| `core/kv_cache_manager.py` | `KVCacheManager`: request-facing facade — computed blocks, slot allocation, free, preemption support, stats |
| `core/kv_cache_coordinator.py` | `KVCacheCoordinator` ABC; `UnitaryKVCacheCoordinator` (single group) and `HybridKVCacheCoordinator` (fixed-point hit agreement across groups); factory `get_kv_cache_coordinator()` |
| `core/single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager` ABC plus `FullAttentionManager`, `SlidingWindowManager`, `ChunkedLocalAttentionManager`, `MambaManager`, `CrossAttentionManager` — one manager per group, each owning its retention/hit/free policy |
| worker attention modules | `get_kv_cache_spec()` walks layers; each `Attention` module reports its own spec; `get_kv_cache_groups()` groups mergeable layers |

### 2.3 Specifications and groups

A spec is *data*, not behavior: geometry (heads, head dim, dtype), page/block
size, and retention flags (window size, chunk size, latent width, state shapes).
Specs are computed per layer, then merged: layers with identical specs share a
group; different specs yield different groups. All groups draw block ids from
one `BlockPool`; the physical buffer aliases per-group regions via per-layer
`KVCacheTensor` strides, and "a block id is owned by one group at a time"
keeps overlaying sound. `MambaSpec` breaks the token-wise assumption entirely
— its page holds one recurrent state of `prod(shape)` elements, not K/V rows —
which is why grouping, not the block size, is the load-bearing concept.

### 2.4 Block pool, free queue, and prefix caching

`KVCacheBlock` carries `ref_cnt` and its hash; the free queue is an intrusive
doubly-linked list threaded through the blocks themselves (LRU at the front).
Hash chaining: `hash_i = H(parent_hash_i-1 ‖ tokens_i ‖ extra_keys)`, first
parent a fixed none-hash; only full blocks are hashed; the last partial block
is never published. Extra keys include LoRA id, multimodal hashes, and
`cache_salt` (first block only). Multiple blocks with the same hash coexist;
adoption takes the first. Eviction is lazy — on allocation pressure, LRU over
unpinned entries; freed uncached blocks return LIFO for L2 locality while
cached ones are evicted FIFO. `reset_prefix_cache()` flips discoverability
without breaking ownership.

### 2.5 Per-group managers and the coordinator

Each manager owns the policy for its group: how many blocks a sequence needs
(`get_num_blocks_to_allocate`), which cached blocks count as a hit
(`find_longest_cache_hit`), when blocks become skippable
(`get_num_skipped_tokens`), and what freeing means. The two policies this
design imports verbatim:

- **Sliding window**: a window of `W` tokens needs `cdiv(W−1, B) + 1` blocks
  (the +1 covers a window that starts mid-block); a hit requires
  `cdiv(W−1, B)` *contiguous* blocks found right-to-left; skipped tokens =
  `max(0, num_computed − W + 1)`; blocks fully below the window are freed and
  their table entries become the reserved null block, which kernels skip.
- **Mamba/recurrent**: state is addressed per state-slot, never per token; a
  cache hit can only resume from a state boundary; preemption restores the
  state at the last boundary; prefix hashing never applies to state pages.

The coordinator reconciles hits across groups: unitary models (one group) use
a trivial fast path; hybrid models iterate `find_longest_cache_hit` over
groups to a fixed point, taking the minimum, full-attention groups first, and
clamping recurrent groups to a boundary.

### 2.6 Memory planning

Available bytes come from a profiling run around a forward pass; the planner
computes `bytes_per_block` from the specs, divides, and workers equalize to
the minimum block count across ranks. Per-layer `KVCacheTensor`s map each
group's region inside one backing allocation.

### 2.7 Kernel contract

Kernels receive batch metadata, not cache internals: query offsets
(`query_start_loc`), sequence lengths, a flat block-table tensor, and a slot
mapping (flat `(block, slot)` positions) for scatter-writes. The cache manager
never touches kernels and vice versa.

### 2.8 Upstream deltas worth knowing (2025-10 → 2026-10)

On current `main`: planning moved to `kv_cache_planning.py`; specs gained
sinks (`SinkFullAttentionSpec`), R-SWA, HiSparse, partial-block prefix
publishing, and cross-page-size packing (`UniformTypeKVCacheSpecs`); the
`SharedBlockManager` was replaced by cross-layer aliasing
(`kv_sharing_target_layer_name`); `fork()`/`update_after_alloc()` were
removed in favor of request-side hash resolution; KV events, the
`kv_offload` tiering framework, and the `KVConnector` protocol grew
substantially. None of these change the core decomposition this design
copies; §9 records which of them are deliberate non-goals for now.

### 2.9 Adopt, adapt, skip

| vLLM concept | Verdict | Reason |
| --- | --- | --- |
| Spec → group → manager → coordinator decomposition | **Adopt** | Proven answer to heterogeneous layers without a monolithic manager |
| Single block pool, one id space, per-group regions | **Adopt** | Already InferX's shape (`KvBlockPool` per-layer regions) |
| Chained full-block hashing, salt in first block, LRU eviction | **Adopt** | Implemented and tested in the pool's absorbed index with `ChainHash`; one binding per key — republishing a key rebinds it and displaces the old block |
| Block-level `ref_cnt` + intrusive free queue in the pool | **Adopt** | One ownership ledger covering prefix pins, eviction, and future fork |
| Sliding-window retention/hit/skip math + null block | **Adopt** | Closes T04 P-02/P-05; kernels already take `window_left` |
| Mamba boundary-resume semantics | **Adapt** | State slots exist (`RecurrentStatePool`); wire lifecycle only |
| Per-group manager classes | **Adapt** | C++ sealed hierarchy, batched calls, no per-token virtual dispatch |
| Python-side spec objects produced by attention modules | **Adapt** | InferX computes specs from family configs into `ModelSpec::state` |
| Kernel-facing batch metadata contract | **Adopt** | Already InferX's flattened CSR (`kv_indices`/`kv_indptr`) |
| Per-group page sizes, cross-page packing | **Skip (now)** | One scheduler block size in phase 1; §8 explains the cost bound |
| KVConnector / offload / KV events / semantic cache | **Skip (now)** | No consumer in InferX yet; seams preserved (§9) |
| Watermark-based admission | **Skip (now)** | InferX preempts newest-first under pressure; equivalent effect for FCFS |
| Cascade attention, EAGLE-specific hit trimming | **Skip** | No cascade kernels, no spec decode on this path (A13 tracks it) |

## 3. InferX today: assets and gaps

### 3.1 Assets that survive unchanged in role

| Component | Path | Role today |
| --- | --- | --- |
| `CacheSpec` / `CacheComponent` | `include/inferx/cache/cache_spec.h` (working tree) | Byte-based per-layer storage + `Retention` variant, `NoCache`/`SharedCache` layer kinds, planning — replaces `KvLayout` and `models/requirements.h`, both removed in the working tree (dependents migrate in phase A, §7.2) |
| `KvBlockPool` | `src/cache/kv_block_pool.cc` | One device allocation, int32 block ids, per-layer regions, LIFO free list, `in_use` bitmap, `KeyCache`/`ValueCache` contiguous views, `Device::Cpu()` pools for tests |
| `BlockTable` | same header | Per-sequence block list + `Locate(pos → block, slot)` |
| `ChainHash` | `include/inferx/cache/block_hash.h` | SHA-256 chained digests, salt in the first link, full blocks only (was `BlockHashChain`; the tip-holding class folded into the free function) |
| prefix index (absorbed) | `src/cache/kv_block_pool.cc` | Hash index with ref-counted sharing, LRU eviction, `Clear()`-style tombstones — `PrefixCache` deleted, absorbed into the pool ledger (§6.2) |
| `CacheManager` | `src/cache/cache_manager.cc` | Per-request logical allocations; `AdoptPrefix` / `PublishComputed` / `Reserve` / `Release` / `Fork` / `ResetPrefixCache` |
| `RecurrentStatePool` | `src/cache/recurrent_state_pool.cc` | Fixed-slot per-layer state regions (delta-rule + conv), stream-ordered `ResetSlot` — **referenced by nothing** (T04 P-07: R) |
| Scheduler driving | `src/engine/scheduler.cc:58,95,112,117` | Reserve per step, adopt at admission, publish on chunk completion, release on finish/preempt/abort |
| Runner flattening | `src/models/model_runner.cc` (`RunStep`) | Per-step flat CSR (`kv_indices`, `kv_indptr`, `last_page_len`) + per-layer cache views via `ForwardContext` |
| Attention kernels | `src/ops/attention.cc`, `flash_attention.cc`, `cuda/*` | FlashInfer fast path (incl. `window_left`), generic fallback (any GQA ratio, sinks, any head dim), `AssembleMlaCaches`, full GDN op set — MLA/GDN unwired |

### 3.2 Gaps this design closes

1. **One implicit group.** Per-layer layouts can differ, but all layers share
   one logical block sequence and one hash namespace; hybrid models
   (SW+full, attention+recurrent) cannot express per-group retention.
2. **No group lifecycle.** Nothing owns "blocks below the window are
   skippable" or "state resumes at boundaries"; checkpoints with
   `use_sliding_window = true` are rejected at config load
   (`src/models/qwen3/qwen3_config.cc:115`,
   `src/models/llama/llama_config.cc:55`).
3. **Two-ledger risk.** Ref counting lives in `PrefixCache` entries while the
   pool tracks only an `in_use` bitmap; fork/beam sharing has no counter to
   use when prefix caching is off.
4. **Recurrent and MLA plumbing dangling.** Storage and ops exist; no manager,
   no scheduler path, no family (T04 P-06/P-07: R).
5. **Build debt.** The kvc provider teardown left dangling references:
   `CMakeLists.txt:41` (`src/kvc`), `:45` (`plugins/kvcache/cuda`), `:60`
   (`tests/kvc`); the tree does not configure as-is.

Not gaps, deliberately: chunked prefill enforcement (C08, scheduler concern),
CUDA-graph block-table buffers (A12), quantized KV write path (P-10).

## 4. Design

### 4.1 Principles

1. **Token-wise vs state-wise is the primary divide.** Paged K/V rows (full,
   sliding-window, MLA latents — same rows, different widths) share one block
   id space and one pool. Fixed-size recurrent state is slots, not blocks, and
   gets its own manager over `RecurrentStatePool`. Nothing forces recurrent
   state into K/V tensors.
2. **Specs are data; managers are behavior.** `CacheSpec` carries no virtuals.
   A factory maps a group's spec to one `CacheGroupManager` subclass.
   Polymorphism exists only in the manager hierarchy, whose methods run
   per-request per-step — never per token — so virtual dispatch is off the
   hot path.
3. **One pool, one ledger.** A block's ref count, cached-ness, hash, and free
   queue position live in exactly one place (`KvBlockPool`). Prefix pins,
   eviction, and fork sharing all move that one counter.
4. **The scheduler keeps a stable facade.** `CacheManager`'s public methods
   survive; new capability (per-group tables, fork) is additive. Attention
   mechanics stay invisible to the scheduler.
5. **Capacity ≠ committed.** Reservation is storage; `num_computed_tokens` is
   the watermark of valid KV (already the contract in
   `cache_manager.h:19` and `Request`).
6. **Single-threaded by ownership, not locks.** All cache mutation happens on
   the scheduler thread; pools and managers document this and carry no locks.
   Device work must complete before release (the existing no-async-lease
   contract) until stream-ordered eviction is designed.
7. **CPU-testable bookkeeping.** Everything except kernels runs on
   `Device::Cpu()` pools (existing pattern, `kv_block_pool.h:83`).
8. **Errors are `Status`, never exceptions.** `ResourceExhausted` from
   allocation is a scheduling input (preempt), not a crash
   (`kv_block_pool.h:102`).

### 4.2 Component overview

```text
ModelConfig ──family config──> CacheRequirements (per-layer NoCache / CacheSpec / SharedCache)
                                      │
                     PlanCacheGroups(state, block_size)     [planning, host-only]
                                      │  merges same-spec layers into groups
                              CachePlan (groups, block_size)
                                      │
                 ┌────────────────────┴─────────────────────┐
        KvBlockPool (token-wise: per-group regions,       RecurrentStatePool
                    block ledger, free queue, hash index)    (state slots)
                 │                                            │
        FullAttentionManager / SlidingWindowManager     RecurrentStateManager
                 └────────────────────┬─────────────────────┘
                                CacheCoordinator        (unitary fast path / hybrid fixed point)
                                        │
                                   CacheManager            (facade; scheduler-facing)
                                        │
        Scheduler ──Reserve/Adopt/Publish/Release──> CacheManager
        ModelRunner ──per-group CSR + LayerCache views──> kernels
```

Ownership: `CacheManager` owns the coordinator, which owns the managers.
Managers borrow the pools (`KvBlockPool*`, `RecurrentStatePool*`), which the
`ModelRunner`/`ModelExecutor` owns today — unchanged. All pointers
non-owning; lifetimes bound to the scheduler's single thread.

### 4.3 Cache specification and grouping

A `CacheSpec` is frozen data describing one layer: byte-denominated storage
`CacheComponent`s (per-token or fixed-slot — geometry stays with the family
that produced it, so quantized payloads with scales, sparse indexers, and
MLA latents are all just bytes) plus a `Retention` variant naming the policy
that governs how long contents stay resident. The variant — not an enum with
optionally-valid fields — makes illegal combinations unrepresentable and
manager dispatch exhaustive. `PlanCacheGroups` merges layers with equal
specs into one group regardless of their position in the model — hybrid
models interleave groups (gpt-oss: three sliding layers then one full) —
and puts all recurrent layers into one trailing group. `NoCache` layers
drop out; `SharedCache` layers (YOCO/Gemma-3n) become `LayerAlias` entries
resolved by `CachePlan::GroupOfLayer`, so cross-layer sharing is planned,
never inferred from compatible layouts. Full interfaces: §6.1.

`CacheRequirements` — `StateRequirements` renamed and moved into the cache
layer — is a per-layer variant, and family configs — which already parse
`use_sliding_window`/`sliding_window` before rejecting them — fill it
instead of erroring (§6.11). **One scheduler
block size applies across all token-wise groups in phase 1**; the planner
rejects a model whose groups would demand different page sizes (none exists
in the supported set), keeping `BlockTable::Locate` valid everywhere, and
gates the not-yet-managed retentions (`ChunkRetention`, `EncoderRetention`)
behind startup errors so the spec vocabulary never outruns the managers.
Per-group page sizes are §9.

### 4.4 Storage: `KvBlockPool` gains a ledger

The pool keeps its allocation strategy — one buffer, one region per
(layer, component), bytes-only — and gains the ledger. Dropping geometry
from the pool is deliberate: components generalize past K/V rows, and
kernel-shaped views become the model's job (`Tensor::Reshape`/`Bitcast` on
the byte views the pool returns). Changes:

1. **Per-group regions.** Layer → group mapping; each group owns a
   contiguous region holding its (layer, component) pairs; every group has
   the same block count, so a block id remains meaningful across groups
   ("owned by one group at a time"), matching vLLM's overlay rule.
   `LayerCache(layer)` returns the layer's byte view; the K/V-shaped views
   kernels consume today are constructed by `models::Attention::Forward`
   (§4.10). A GQA layer lists K and V as separate components, so each
   entry's contiguity across blocks survives — the old entry-outermost
   decision, generalized.
2. **Block ledger** replaces the `in_use` bitmap and absorbs `PrefixCache`:
   per-block ref count, hash, cached/discoverable flags, and intrusive
   free-queue/LRU links, all owned by the pool. Free order follows vLLM's
   split: never-cached blocks free LIFO (L2 locality), cached blocks leave
   the LRU tail under eviction pressure. Today's `PrefixCache` public
   semantics (pinned never evicted, duplicates coexist, `Clear()` tombstones
   + epoch guard) become pool methods unchanged — §6.2 has the migration
   table, §7.1 the test-port mapping.
3. **Null block.** Block id 0 is reserved (`kNullBlockId`); it is never
   allocated and never written. Sliding-window tables point rolled-off
   entries at it; kernels skip id 0 rows.
4. **RecurrentStatePool stays a separate pool.** Slots are not blocks; they
   are indexed by slot id, sized per component from the spec's fixed-slot
   components, and never hashed.

### 4.5 Prefix hashing

The facade owns one chain per request, stored as a plain digest vector and
extended through `ChainHash` (`block_hash.h`, implementation in
`src/cache/cache_manager.cc`):

```text
digest_i   = SHA256("inferx.kv.chainhash.v1" ‖ digest_{i-1}
                    ‖ u64le(token_count) ‖ tokens as u64le ‖ u64le(salt_len) ‖ salt)
                    // salt rides in the first link only, so it namespaces the whole chain
```

The same tokens in two groups are two different KV realities, so their index
entries must never collide — but the isolation lives in the pool's index key
(`BlockKey` = digest + group index), not in the digest itself, which keeps
digests group-agnostic and the length unchanged (vLLM packs the group id into
the stored hash key; same effect, different layer).

One binding per key: republishing a digest rebinds it and displaces the old
block (held blocks keep their owner's reference; idle ones return to the free
queue). Full blocks only; the partial tail stays private; the n−1 logits cap
stays at the `CacheManager` boundary. SHA-256 stays (deterministic across
processes and builds; faster digests are a recorded non-goal, T04 P-11).
Interface: §6.3.

### 4.6 Group managers

One `CacheGroupManager` subclass per retention policy owns that group's
retention, hit-search, reservation, release, and fork policy over a
`RequestGroupState` (the per-request, per-group slice of bookkeeping). Three
concrete managers in phase 1 — full interfaces and behavioral contracts in
§6.4–§6.7:

- **`FullAttentionManager`** — today's `CacheManager` logic, verbatim:
  block-aligned growth, adopt/publish at block granularity, LRU eviction
  under pressure. Covers MHA/MQA/GQA (any head counts, any head dim, causal
  or not) and **MLA**, whose latent is one
  `MlaLatentComponent(kv_lora_rank + rope_dim)` per layer; decompressed
  assembly uses the existing `ops::AssembleMlaCaches`, absorbed decode lands
  with F12.
  Full-attention layers with sinks need no special cache treatment — full
  attention retains everything by definition; sinks are a kernel concern
  (`AttentionParams.sinks`).
- **`SlidingWindowManager`** — vLLM's math imported exactly (§2.5): needs
  `cdiv(W−1, B)+1` blocks; hits require `cdiv(W−1, B)` contiguous blocks
  searched right-to-left; `skipped = max(0, num_computed − W + 1)`; rolled-off
  blocks are freed and their table entries become `kNullBlockId`. **Sinks are
  a retention floor here**: when the model sinks its first `sink_tokens`
  tokens (gpt-oss), the manager never releases the leading
  `cdiv(sink_tokens, B)` blocks, and `skipped` never advances past
  `sink_tokens`. Retention math is what makes the existing
  `window_left`/window-floor kernels sufficient.
- **`RecurrentStateManager`** — slot lifecycle over `RecurrentStatePool`:
  one slot per request at admission (reset is stream-ordered and issued by
  the runner before the step's kernels, §6.7), `Fork` deep-copies state (no
  CoW — states are written every step), release returns the slot. State is
  never hashed, never in the prefix index, never in a block table. Phase E
  has **no cross-request state sharing**: a hybrid model's recurrent group
  reports zero cache hit, which drives the coordinator's minimum to zero, so
  prefix adoption effectively disables itself for hybrid requests until
  boundary-state sharing lands (§9); preemption recomputes from scratch, as
  it does today.

Chunked-local, encoder-side, and sparse managers are deliberately absent:
no supported family needs them — their spec vocabulary exists
(`ChunkRetention`, `EncoderRetention`) but the planner rejects it at startup
(T04 P-04/P-08) — and adding a manager later touches exactly two places: a
retention alternative and a manager subclass.

### 4.7 Coordinator

The coordinator owns the group list and turns per-group decisions into one
per-request answer. Unitary models (≤1 token-wise group) take a fast path
that skips the iteration; hybrid models run the fixed point: every group
reports its longest hit over the same token stream, the agreed hit is the
minimum, token-wise groups first, and the recurrent group's answer is zero in
phase E. Two passes suffice — hits only shrink. `CacheHit` carries per-group
adopted block lists plus the agreed `cached_tokens` (the same role
`PrefixAdoption.cached_tokens` plays today). Interface: §6.8.

### 4.8 `CacheManager` facade and scheduler contract

The facade owns per-request state (`std::vector<RequestGroupState>` in group
order plus per-group hash chains and the epoch — today's `PrefixState`
generalized) and keeps its public API. Detailed interface: §6.9.

| Method | Change |
| --- | --- |
| `AdoptPrefix(request, prompt, salt, max_tokens)` | Same signature; delegates to the coordinator; `PrefixAdoption.block_ids` is the group-0 view (hybrid callers use `group_block_ids`) |
| `Reserve(request, target_tokens)` | Same contract (all-or-nothing, never shrinks); internally asks every manager and allocates only after all groups pass — window groups reserve fewer blocks, the recurrent group checks slot availability |
| `PublishComputed(request, prompt, output)` | Same; per-group chains, extended by the facade |
| `Blocks(request)` | Same for unitary; `GroupBlocks(request, group)` added for hybrid |
| `Release(request)` / `ResetPrefixCache()` / counters | Same semantics; the pool's reset fails while any block is held |
| `Fork(request, children…)` | **New**: coordinator fans out per-group `Fork`; children register as new request ids |
| `TakeSlotsNeedingReset(slots*)` | **New**: the runner drains newly assigned recurrent slots and resets them stream-ordered before its kernels |

`SchedulerOutput`'s `block_ids`/`new_block_ids` stay as the group-0 view; a
hybrid model additionally fills per-group tables (§6.11) that the runner
flattens per group. When the model is unitary the wire format is
byte-identical to today's.

### 4.9 Request lifecycle

```text
admission:   AdoptPrefix (pin hits, record chains)
             Reserve(first chunk)   → blocks granted / ResourceExhausted → preempt-newest (existing path)
step:        Reserve(target_tokens) → may grant nothing (decode with capacity)
             runner writes KV at (table[pos/B], pos%B) slots, reads via per-group CSR
             PublishComputed over prompt+accepted output (full blocks only)
preemption:  Release → cached blocks stay in index (evictable); recurrent slot freed; recompute on resume
finish/abort:Release → unpinned cached blocks become LRU candidates, slots freed
fork (beam): Fork → children share token-wise blocks (ref-counted), copy state;
             first write to a shared partial block duplicates it (CoW)
clear:       ResetPrefixCache → discoverability off; fails while any block is held (drain first)
```

Invariants, each directly testable (§7.9):

1. A block is on the free queue **xor** `ref_cnt > 0`; double-free and
   free-of-cached-pinned are errors, not silent corruption.
2. `ref_cnt` equals the number of holders (request ownership + prefix pin +
   fork sharers); it reaches 0 only via `Release`/unpin paths.
3. Pinned blocks are never eviction victims; evicted blocks leave the LRU
   queue and the hash index atomically (one ledger makes this one step).
4. Only full blocks are hashed/published; a partial tail block is private to
   its writer; `PublishComputed` is idempotent per block.
5. Hash chains are immutable functions of (salt, group, tokens); equal
   digests imply equal (parent, tokens) — duplicates coexist, adoption picks
   deterministically (first).
6. Group tables satisfy: token at logical position p lives at
   `blocks[p / B]` slot `p % B` for every token-wise group, with positions
   below the retention floor (window/sink) pointing at `kNullBlockId` (never
   read or written).
7. `num_cached_tokens ≤ num_computed_tokens ≤ allocated capacity` per
   request; a hit never covers the final prompt token (n−1 rule).
8. Requests admitted before a `ResetPrefixCache` never publish after it
   (epoch guard, unchanged).
9. All-or-nothing `Reserve`: failure leaves every group's holdings and the
   pool availability unchanged.
10. A recurrent slot is assigned to at most one request at a time and is
    reset exactly once per assignment, before any kernel reads it; fork
    children start from copied state, never the parent's live slot.

### 4.10 Attention and runner integration

The runner contract stays "flat CSR + per-layer views" — the design adds one
dimension (group), not a new mechanism:

- Per token-wise group: `kv_indices` (concatenated per-seq block ids),
  `kv_indptr` (CSR offsets), `last_page_len`, and per-layer byte views from
  `LayerCache(layer)` that `models::Attention::Forward` reshapes into the
  K/V-shaped tensors kernels consume today. Slot mapping needs no new
  tensor: `WritePagedKv` already scatters by `positions` through the table
  (`src/models/layers/attention.cc:24`).
- Window semantics flow through the existing `AttentionParams.sliding_window`
  (FlashInfer `window_left`, generic-kernel window floor); the runner derives
  `window_left` from the model's per-layer window and `num_computed_tokens`
  — no new wire data. `kNullBlockId` rows are skipped by one-line guards.
- Recurrent groups produce per-request `state_slot` lists instead of tables;
  the GDN ops consume slot-indexed state tensors from `RecurrentStatePool`.
- CUDA-graph replay (A12) will pin each group's CSR into the fixed flat
  buffer `BlockTable`'s header already describes; no API change anticipated.

### 4.11 Memory planning and sizing

`CacheConfig` precedence is unchanged (utilization-bytes → explicit bytes →
block count, `model_runner.cc:263`). The planner becomes spec-aware:
`bytes_per_block = max over token-wise groups of
CacheGroupSpec::BytesPerBlock(block_size)` (uniform across groups because
ids share one space). `num_blocks = budget / bytes_per_block`; TP ranks keep
the existing all-reduce-min equalization. The profiling step itself is
unchanged. Recurrent pools size separately: `max_slots ≥ max_num_seqs` from
`SchedulerConfig`, bytes from `StateBytesPerSequence` — a small, explicit
budget line next to the KV budget.

### 4.12 Conventions

C++23, namespace `inferx`, `INFERX_*` include guards, `absl::Status` /
`StatusOr` with `INFERX_RETURN_IF_ERROR` / `INFERX_ASSIGN_OR_RETURN`, types
`PascalCase`, methods `PascalCase`, members `snake_case_`, constants
`kPascalCase` — **the repository's established convention, which is Google
style except that methods are PascalCase; consistency with 60+ existing
headers wins over the style guide's default**, and this deviation is recorded
here once. No exceptions; no new third-party dependencies; headers minimal
(forward-declare managers where a facade suffices). Thread-safety: nothing in
`src/cache` is thread-safe and none of it needs to be — single scheduler
thread, documented per class as today.

## 5. Attention mechanism support

### 5.1 How mechanisms map onto the design

- **MHA / MQA / GQA / causal / bidirectional**: one `FullAttentionManager`
  group; head-count and head-dim variation is `GqaKvComponent` data, already
  exercised by kernels (FlashInfer fast path + generic fallback).
- **MLA**: same manager, one `MlaLatentComponent` per layer; token-level
  prefix hashing is exact on latents (the latent *is* the per-token state);
  execution path is F12 (absorbed decode or `AssembleMlaCaches`).
- **Sliding window / local**: `SlidingWindowManager` retention + existing
  windowed kernels; window < block, window crossing block edges, and
  per-request reuse caps are all §2.5 arithmetic.
- **Hybrid SW+full (gpt-oss, Gemma-2/3-style)**: two token-wise groups, one
  coordinator, per-group tables; admission reserves the *max* over groups
  per token, which the planner prices into `bytes_per_block`.
- **Recurrent / linear (GDN, Mamba2)**: `RecurrentStateManager` over existing
  pools and ops; hybrid attention+recurrent (Qwen3-Next shape) composes
  through the coordinator (zero state hit ⇒ zero request hit in phase E).
- **Sinks (gpt-oss)**: a retention floor on the window manager
  (`CacheSpec.sink_tokens`); the leading sink blocks are prefix-cacheable
  like any other blocks — identical openings hash identically — and never
  roll off. Kernel numerics use the existing `AttentionParams.sinks` input
  to the generic kernel. Full-attention layers need no cache-side treatment.
- **Cross-attention / block-sparse / chunked-local**: not represented; config
  rejection with actionable errors (T04 P-04/P-08/P-09). The spec `Kind` enum
  and manager factory are the only seams a future family needs.
- **Quantized KV**: `DataType` already sizes the pool (P-10 storage-only);
  the quantized *write/read path* is orthogonal to this design and stays R.

### 5.2 Support matrix

Status vocabulary from T04: **S** supported and tested, **R** rejected at
startup with a tested actionable error, **P** planned by this design
(phase from §7). "Today" reflects the working tree as of 2026-10-19: the
cache layer (spec, pool ledger, group managers, facade) is implemented and
unit-tested; the engine wiring of §7.2 is still in flight.

| Mechanism | Storage today | Lifecycle today | After this design | T04 |
| --- | --- | --- | --- | --- |
| Full attention MHA/MQA/GQA, BF16 | S | S | S unchanged | P-01 |
| Sliding window | S (kernels) | S at manager/facade level (`ReleaseSkipped`, pads, skipped tokens); family config + kernel wiring pending | **P** (phase C) | P-02 |
| Attention sinks | S (generic kernel) | R (no model passes sinks) | **P** (phase C, with FX-sink) | P-03 |
| Chunked local (Gemma-3) | — | `ChunkedLocalManager` implemented and unit-tested; no family config yet | Manager exists (§6.4 factory); families later | P-04 |
| Hybrid multi-group | S (per-layer components) | S at manager/facade level (multi-group atomic Reserve); scheduler output wiring pending | **P** (phase D) | P-05 |
| Cross-layer KV sharing (YOCO/Gemma-3n) | — | R (unrepresented) | **P** (planned `SharedCache`/`LayerAlias`; pool + runner aliasing with phase D) | P-05 |
| MLA | S (latent layout + ops) | R (unwired) | **P** (phase F = F12) | P-06 |
| Recurrent/linear (GDN/Mamba) | S (pool + ops, unwired) | R | **P** (phase E; no cross-request state sharing, preemption recomputes) | P-07 |
| Cross-attention | — | R | R stays (`EncoderRetention` vocabulary, planner-gated) | P-08 |
| Sparse / selected-token | — | R | R stays | P-09 |
| Quantized KV | S (byte-level) | R (no RW path) | R stays | P-10 |
| Fork / beam branching | — | S at manager/facade level (shared refs, CoW, slot copies); runner device copies pending | **P** (phase G) | — |
| Prefix caching | S | S | S (per-group chains) | P-11 |

## 6. Detailed class interfaces

This section is the authoritative API specification. Signatures are the
contract; private members are omitted except `BlockRecord`, shown because its
fields *are* the ledger invariants. All classes: namespace `inferx`,
single-threaded with the scheduler, `Status`/`StatusOr` error handling, no
exceptions. File paths are new unless marked *modified*.

**Status (2026-10-19).** The cache layer matches this document and is
unit-tested end to end (spec/planner, pool ledger with the absorbed index,
group managers for all five retentions, the `CacheManager` facade, the
recurrent state pool; suites in §7.9): `cache_spec.cc` exists,
`kv_block_pool.cc` is byte-denominated on `CacheComponent` with the §6.2
ledger, `recurrent_state_pool.cc` is implemented off `CacheGroupSpec`, the
`CrossAttention()` factory still exists (it aliases `Encoder`; drop it when
the families migrate), and `prefix_cache.h/.cc` plus `BlockHashChain` are
deleted (absorbed per §6.2/§4.5). The tree still does not compile overall:
the engine migration of §7.2 step 2 is open — `scheduler.h` holds a
`CacheManager` by value it cannot construct, `model_runner.cc` still reads
the deleted `spec.state.paged_kv` shape, and `llama_config.h` still includes
`models/requirements.h`.

### 6.1 `include/inferx/cache/cache_spec.h` — specification and planning

Storage is byte-denominated and per-component; retention is a closed
variant. Geometry stays with the family that produced it — quantized
payloads with scales, sparse indexers, and MLA latents are all just bytes to
the cache, which is where vLLM itself went (opaque
`state_content_size_bytes`).

```cpp
namespace inferx {

/// \brief Storage requirements of one cache component of one layer, in
/// bytes. kPerToken components are paged; kFixedSlot components hold
/// per-sequence state. dtype is informational: kernels, not the cache,
/// interpret the bytes.
struct CacheComponent {
  enum class Growth : std::uint8_t { kPerToken = 0, kFixedSlot = 1 };

  Growth growth = Growth::kPerToken;
  int64_t bytes_per_token = 0;  ///< kPerToken only; > 0.
  int64_t bytes_per_slot = 0;   ///< kFixedSlot only; > 0.
  DataType dtype = DataType::kBFloat16;

  bool operator==(const CacheComponent&) const = default;
  static CacheComponent PerToken(int64_t bytes_per_token, DataType dtype);
  static CacheComponent FixedSlot(int64_t bytes_per_slot, DataType dtype);
};

/// GQA/MHA/MQA K+V component: 2 * kv_heads_per_rank * head_dim elements per
/// token; heads are rank-local (post-TP).
constexpr CacheComponent GqaKvComponent(int64_t kv_heads_per_rank,
                                        int64_t head_dim, DataType dtype);
/// MLA compressed latent: latent_dim elements per token; never TP-sharded.
constexpr CacheComponent MlaLatentComponent(int64_t latent_dim, DataType dtype);

/// \brief Retention policies; parameters are unrepresentable for the wrong
/// policy, and manager dispatch is exhaustive over the variant.
struct FullRetention {};                ///< Keep everything (MHA/MQA/GQA/MLA).
struct WindowRetention {                ///< Trailing window + optional sinks.
  int64_t window = 0;       ///< > 0.
  int64_t sink_tokens = 0;  ///< Multiple of block_size; 0 = none (gpt-oss).
};
struct ChunkRetention {                 ///< Vocabulary only; planner-gated.
  int64_t chunk_tokens = 0;
};
struct EncoderRetention {};             ///< Vocabulary only; planner-gated.
struct RecurrentRetention {             ///< Fixed state; boundaries defined.
  /// Positive multiple of block_size; state-validity granularity. No
  /// cross-request sharing in phase E (§6.7).
  int64_t boundary_tokens = 0;
};
using Retention = std::variant<FullRetention, WindowRetention, ChunkRetention,
                               EncoderRetention, RecurrentRetention>;

/// \brief One layer's cache: storage components + retention. Data only.
struct CacheSpec {
  static CacheSpec FullAttention(std::vector<CacheComponent> components);
  static CacheSpec SlidingWindow(std::vector<CacheComponent> components,
                                 int64_t window, int64_t sink_tokens = 0);
  static CacheSpec ChunkedLocal(std::vector<CacheComponent> components,
                                int64_t chunk_tokens);      // planner-gated
  static CacheSpec Encoder(std::vector<CacheComponent> components);  // gated
  static CacheSpec Recurrent(std::vector<CacheComponent> slots,
                             int64_t boundary_tokens);

  /// Non-empty; kPerToken for token-wise retention, kFixedSlot for
  /// RecurrentRetention.
  std::vector<CacheComponent> components;
  Retention retention = FullRetention{};

  bool operator==(const CacheSpec&) const = default;

  bool IsTokenWise() const;          // False iff RecurrentRetention.
  bool PrefixCacheable() const;      // Full/window/chunk: yes. Encoder/recurrent: no.
  int64_t BytesPerTokenPerLayer() const;  // Σ bytes_per_token; 0 when recurrent.
  int64_t StateBytesPerLayer() const;     // Σ bytes_per_slot; 0 when token-wise.

  /// \brief Resident-block upper bound for one sequence:
  /// cdiv(n, B) full; cdiv(W-1, B) + 1 window (misalignment slack);
  /// + cdiv(sink_tokens, B) leading sink blocks when sinking; 0 recurrent.
  int64_t MaxBlocks(int64_t num_tokens, int64_t block_size) const;

  /// \brief Per-spec validation: non-empty components, growth matching the
  /// retention, positive window/chunk, block-aligned sinks and boundaries,
  /// positive byte counts.
  Status Validate(int64_t block_size) const;

  /// Diagnostics, logging, error messages.
  std::string ToString() const;
};

/// \brief A layer that owns no cache (distinct from an invalid empty spec).
struct NoCache { bool operator==(const NoCache&) const = default; };

/// \brief A layer that reads another layer's cache (Gemma 3n, YOCO):
/// cross-layer sharing declared by the architecture, never inferred from
/// compatible layouts. The source must be an earlier storage-owning layer.
struct SharedCache {
  int64_t source_layer = -1;
  bool operator==(const SharedCache&) const = default;
};

/// \brief A model's per-layer cache requirements — `StateRequirements`
/// renamed and moved from models/ into the cache layer: models depending on
/// cache is the correct direction, and it removes cache's old upward
/// include of models/requirements.h. ModelSpec carries it as `cache`.
struct CacheRequirements {
  using LayerCache = std::variant<NoCache, CacheSpec, SharedCache>;
  std::vector<LayerCache> layers;  ///< One entry per layer, execution order.
};

/// \brief One manager's worth of layers: equal specs, so one spec serves all.
struct CacheGroupSpec {
  int64_t group_index = 0;      ///< Token-wise groups first, recurrent last.
  std::vector<int64_t> layers;  ///< Storage owners in model order; may
                                ///< interleave with other groups' layers;
                                ///< SharedCache layers are not listed.
  CacheSpec spec;

  int64_t NumLayers() const;
  /// Σ over layers of BytesPerTokenPerLayer × block_size; 0 when recurrent.
  int64_t BytesPerBlock(int64_t block_size) const;
  /// Σ over layers of StateBytesPerLayer; 0 when token-wise.
  int64_t StateBytesPerSequence() const;
  /// Startup headroom check without instantiating managers.
  int64_t MaxBytesPerSequence(int64_t num_tokens, int64_t block_size) const;
};

/// \brief A layer resolved to read another layer's cache.
struct LayerAlias {
  int64_t layer = 0;
  int64_t source_layer = 0;  ///< The storage-owning layer.
  bool operator==(const LayerAlias&) const = default;
};

/// \brief Planning result. Invariant (established by PlanCacheGroups):
/// token-wise groups precede the recurrent group; TokenWiseGroups() is a
/// prefix of `groups` and depends on that ordering.
struct CachePlan {
  std::vector<CacheGroupSpec> groups;
  std::vector<LayerAlias> aliases;
  int64_t block_size = 0;

  int64_t NumGroups() const;
  int64_t NumTokenWiseGroups() const;
  absl::Span<const CacheGroupSpec> TokenWiseGroups() const;
  /// Fast path predicate; precise about the optional trailing recurrent group.
  bool HasAtMostOneTokenWiseGroup() const;
  const CacheGroupSpec* RecurrentGroup() const;          ///< Or nullptr.
  /// \brief Group storing `layer`'s cache, following aliases; nullptr for
  /// layers without cache.
  const CacheGroupSpec* GroupOfLayer(int64_t layer) const;
};

/// \brief Merges a model's state requirements into cache groups.
///
/// Merge rule: layers with equal CacheSpec share one group regardless of
/// position (hybrid families interleave); recurrent layers form one
/// trailing group; NoCache layers drop out; SharedCache layers become
/// aliases. window or chunk >= max_model_len promotes to full attention
/// (std::nullopt disables promotion).
///
/// InvalidArgument: no cached layer at all; any spec failing
/// Validate(block_size); a SharedCache whose source is not an earlier
/// storage-owning layer; per-group page-size demands that differ
/// (phase-1 uniform block size, §4.3).
/// UnimplementedError: ChunkRetention and EncoderRetention — the spec
/// vocabulary exists, the managers do not (T04 P-04/P-08); rejection stays
/// at startup, actionable, exactly like today's config gates.
StatusOr<CachePlan> PlanCacheGroups(const CacheRequirements& requirements,
                                    int64_t block_size,
                                    std::optional<int64_t> max_model_len = std::nullopt);

}  // namespace inferx
```

`max_model_len` is the scheduler's resolved context length. Any window
`W >= 1` is otherwise accepted — `W < block_size` is legal retention math;
`W >= max_model_len` promotes the layer to full attention instead.

### 6.2 `include/inferx/cache/kv_block_pool.h` (modified) — storage + ledger

The free-list vector + `in_use` bitmap are replaced by the ledger;
`PrefixCache`'s index moves in. `KvLayout` is gone (§6.1): the pool stores
bytes per (layer, component) and stops understanding geometry — kernel views
become the model's job.

```cpp
namespace inferx {

class KvBlockPool {
 public:
  /// Reserved, never allocated, never written. Window-rolled table entries
  /// point here; kernels skip block 0 rows.
  static constexpr int32_t kNullBlockId = 0;

  /// \brief Carves `num_blocks` usable blocks (ids 1..num_blocks; id 0 is
  /// the null block) out of one allocation, with one region per token-wise
  /// group sized from the group's own layouts.
  ///
  /// \param token_wise_groups  Groups from CachePlan (recurrent group
  ///                           excluded — it owns no blocks).
  /// \return                   The pool, or ResourceExhausted if it will
  ///                           not fit / InvalidArgument on empty groups.
  static StatusOr<KvBlockPool> Create(int64_t num_blocks, int64_t block_size,
                                      absl::Span<const CacheGroupSpec> token_wise_groups,
                                      Device device);

  /// Legacy overload kept through phase B: one implicit full-attention
  /// group of `num_layers` layers sharing `component`. Removed once all
  /// callers migrate to the group-based Create.
  static StatusOr<KvBlockPool> Create(int64_t num_layers, int64_t num_blocks,
                                      int64_t block_size,
                                      const CacheComponent& component, Device device);

  // -- allocation and ledger ------------------------------------------------

  /// \brief Takes a free block, evicting LRU cached entries first when the
  /// free queue is empty. Never returns kNullBlockId.
  /// \return A block id, or ResourceExhausted when nothing is free or
  ///         evictable. Callers preempt on exhaustion, as today.
  StatusOr<int32_t> AllocateBlock();

  /// \brief +1 reference (adopt/fork/sharing). Error on a free block.
  Status AcquireBlock(int32_t block);

  /// \brief -1 reference. At zero: cached blocks stay in the index and join
  /// the LRU; uncached blocks return to the free queue (LIFO position).
  Status ReleaseBlock(int32_t block);

  /// \brief Releases a batch (finished sequence, unfork children).
  Status FreeBlocks(absl::Span<const int32_t> blocks);

  /// \brief Evicts up to `count` cached blocks, least-recently-used first.
  /// Pinned entries are never victims. Returns the number actually evicted.
  int64_t EvictCached(int64_t count);

  // -- prefix index (absorbed PrefixCache; semantics unchanged) -------------

  /// \brief Pins the longest cached prefix of `hashes`, stopping at the
  /// first miss; `max_blocks` caps the walk. Returns pinned block ids in
  /// logical order. Duplicated hashes: first entry wins, as today.
  std::vector<int32_t> AdoptBlocks(absl::Span<const BlockHash> hashes,
                                   int64_t max_blocks);

  /// \brief Registers one fully computed block under `hash`. The caller
  /// holds the resulting reference. Re-registering a block is a no-op.
  void PublishBlock(const BlockHash& hash, int32_t block);

  /// \brief Drops a block's index entry without freeing the block (used
  /// when a window group rolls a published block off and its last
  /// reference drops). No-op on uncached blocks.
  void ForgetBlock(int32_t block);

  /// \brief Makes every cached entry undiscoverable. Evictable entries are
  /// reclaimed now; pinned ones become ownership-only tombstones that free
  /// when their last holder releases (the manager-level epoch guard in
  /// ResetPrefixCache prevents pre-clear republishing).
  void ClearCachedEntries();

  /// \brief Cached and discoverable block for `hash`, if any.
  bool LookupCached(const BlockHash& hash, int32_t* block) const;

  /// \brief True while the ledger tracks `block` (entry or tombstone).
  bool Contains(int32_t block) const;

  /// \brief Discoverable entries; tombstones do not count.
  int64_t CachedBlocks() const;
  /// \brief Cached entries with zero references; the eviction pool's depth.
  int64_t EvictableBlocks() const;

  // -- geometry and kernel views ---------------------------------------------

  /// \brief One layer's byte region: a [num_blocks, block_size,
  /// bytes_per_token] uint8 view (K and V of a GQA layer are separate
  /// components, so each stays contiguous across blocks — the old
  /// entry-outermost decision, generalized). The owning model layer
  /// reshapes it into kernel geometry (Tensor::Reshape/Bitcast); the pool
  /// never interprets component bytes.
  StatusOr<Tensor> LayerCache(int64_t layer) const;

  /// \brief The group owning `layer`; groups are created by Create.
  const CacheGroupSpec& GroupForLayer(int64_t layer) const;
  int64_t GroupIndexOfLayer(int64_t layer) const;
  int64_t NumTokenWiseGroups() const;

  int64_t NumLayers() const;
  int64_t NumBlocks() const;               // usable blocks (null block excluded)
  int64_t BlockSize() const;
  int64_t FreeBlocks() const;              // free queue depth (excludes evictable)
  int64_t UsedBlocks() const;
  size_t Bytes() const;
  int64_t BlocksForTokens(int64_t tokens) const;

  /// \brief Bytes one block id occupies across ALL groups' layers — the
  /// sizing unit for num_blocks. Max over groups of per-group layer sums.
  static int64_t BlockBytes(absl::Span<const CacheGroupSpec> groups,
                            int64_t block_size);
};

/// \brief The per-block ledger (private; shown because its fields are the
/// invariants of §4.9). One entry per usable block id.
struct BlockRecord {
  /// Holders: request ownership, prefix pins, fork sharers. 0 exactly while
  /// the block is on the free queue.
  uint32_t ref_cnt = 0;

  /// Digest while cached; zeroed on eviction/forget.
  BlockHash hash{};

  /// Has a live index entry (true even for tombstones until reclaimed).
  bool cached = false;
  /// False once ClearCachedEntries() made the entry undiscoverable.
  bool discoverable = false;

  /// Intrusive links; membership is exclusive: free queue (never cached,
  /// ref 0), LRU queue (cached, ref 0), or neither (ref > 0).
  int32_t prev_free = -1;
  int32_t next_free = -1;
  bool on_lru = false;
};

}  // namespace inferx
```

Migration map (old `PrefixCache` method → new home; `prefix_cache_test.cc`
ported case-for-case — done, §7.2 step 4):

| `PrefixCache` (today) | `KvBlockPool` (phase A) | Difference |
| --- | --- | --- |
| `Adopt(hashes, max_blocks)` | `AdoptBlocks(hashes, max_blocks)` | Identical; pins via ledger `ref_cnt` |
| `Publish(hash, block)` | `PublishBlock(hash, block)` | Identical |
| `Release(blocks)` | `ReleaseBlock(block)` / `FreeBlocks` | Ref counting was per-entry, now per-block |
| `Evict(count)` | `EvictCached(count)` | Identical (LRU, pinned safe) |
| `Clear()` | `ClearCachedEntries()` | Identical (tombstones preserved) |
| `Contains` / `CachedBlocks` / `EvictableBlocks` | same names | Identical |
| (none) | `AcquireBlock` | New: fork/share pinning without a hash |
| `EvictFor` in `CacheManager` | removed | Eviction moved into `AllocateBlock` (lazy, vLLM-style) |

### 6.3 `include/inferx/cache/block_hash.h` — hashing

`prefix_cache.h` is deleted: the index moved into the pool (§6.2) and the
hash types live in `block_hash.h`, which needs nothing from the pool — the
old include cycle is gone.

```cpp
namespace inferx {

/// \brief Content digest identifying one block's KV state.
struct BlockHash {
  std::array<uint8_t, 32> bytes{};
  bool operator==(const BlockHash&) const = default;
  // AbslHashValue: the digest is already uniform; hash its first 8 bytes.
};

/// \brief One link of the facade-owned hash chain: the digest of `parent`
/// chained with one block of tokens, with optional `salt` bytes mixed into
/// the first link only, so a salt namespaces every later digest (§4.5).
/// Domain-separated and length-framed; the facade stores whole chains as
/// plain digest vectors and passes the parent explicitly, which is why this
/// is a free function rather than a tip-holding object.
BlockHash ChainHash(const BlockHash& parent, absl::Span<const TokenId> tokens,
                    absl::Span<const uint8_t> salt);

}  // namespace inferx
```

The group-id namespacing the old v2 sketch put in the chain seed lives in
the pool's `BlockKey` (digest + group index) instead.

### 6.4 `include/inferx/cache/cache_group_manager.h` — manager base

The implemented surface (see the header for full contracts):

```cpp
namespace inferx {

/// \brief A request's holdings in a token-wise group.
struct TokenWiseRequestState {
  /// Block table in logical order; positions below the retention floor
  /// hold kNullBlockId so the table stays indexable by logical block.
  std::vector<BlockId> blocks;
  /// Tokens adopted from the prefix cache at admission (whole blocks).
  int64_t num_cached_tokens = 0;
  /// Leading blocks already handled by PublishCompleted (publish cursor).
  int64_t num_published_blocks = 0;
};

/// \brief A request's holdings in the recurrent group.
struct RecurrentRequestState {
  int64_t slot = -1;  ///< Assigned state slot; -1 while unallocated.
};

/// Per-request, per-group bookkeeping. The alternative matches the group's
/// kind, so a recurrent state can never carry a block table.
using RequestGroupState = std::variant<TokenWiseRequestState, RecurrentRequestState>;

/// \brief Where a request is and where one step takes it.
struct GrowthTarget {
  int64_t num_computed_tokens = 0;  ///< Drives window and chunk roll-off.
  int64_t target_tokens = 0;        ///< Tokens the request holds after this step.
  int64_t encoder_tokens = 0;       ///< Encoder length, for cross-attention; 0 elsewhere.
};

/// \brief A prefix-cache hit found by a read-only search. Nothing is pinned.
struct CacheHit {
  std::vector<BlockId> blocks;  ///< Logical order; window hits lead with kNullBlockId.
  int64_t num_tokens = 0;       ///< Tokens covered; a multiple of the block size.
};

/// \brief One block duplication the runner must perform (copy-on-write fork).
struct BlockCopy { BlockId src = kNullBlockId; BlockId dst = kNullBlockId; };

/// \brief One recurrent-state device copy the runner must perform for a fork.
struct SlotCopy { int64_t parent_slot = -1; int64_t child_slot = -1; };

/// \brief What a successful Reserve handed to the request.
struct Granted {
  std::vector<BlockId> blocks;          ///< New blocks, logical order; null pads excluded.
  std::optional<int64_t> slot;          ///< Newly assigned recurrent slot, if any.
  std::vector<BlockCopy> copies;        ///< Copy-on-write duplications to perform.
};

/// \brief Lifecycle policy for one cache group. Implementations are `final`
/// and created only by CreateCacheGroupManager. Single-threaded with the
/// scheduler; methods run per request per step, never per token.
///
/// Hash-chain extension is facade-owned (the facade holds the per-group
/// chains and passes digests in). Managers own retention, hit search,
/// allocation, publication, and release of blocks and slots.
///
/// Per-step call order for one request, across all of its groups:
///   1. ReleaseSkipped  - free blocks that rolled off the retention window.
///   2. NumNewBlocksNeeded on every group, summed against pool capacity.
///   3. Reserve on every group; if one fails, Rollback the ones that
///      succeeded.
///   4. PublishCompleted once the step's blocks are computed.
class CacheGroupManager {
 public:
  virtual ~CacheGroupManager() = default;

  /// \name Identity
  ///\{
  virtual const CacheGroupSpec& group() const = 0;
  int64_t group_index() const;
  bool IsTokenWise() const;
  ///\}

  /// \brief Longest usable cached prefix of at most `max_tokens` tokens, as
  /// whole blocks. Read-only: takes no references, so a coordinator can call
  /// it repeatedly with shrinking caps until every group's hit agrees.
  /// Recurrent groups return an empty hit in phase E.
  virtual CacheHit FindLongestCacheHit(absl::Span<const BlockHash> hashes,
                                       int64_t max_tokens) const = 0;

  /// \brief Blocks (token-wise) or slots (recurrent: always 1) this group
  /// holds once the request reaches `target_tokens`.
  virtual int64_t NumBlocksHeldAt(int64_t target_tokens, int64_t encoder_tokens = 0) const = 0;

  /// \brief Installs a hit into the request state and pins its blocks.
  /// All-or-nothing; sets num_cached_tokens.
  virtual Status AdoptCached(RequestGroupState& req, const CacheHit& hit) = 0;

  /// \brief Frees blocks that rolled off the retention floor at
  /// num_computed_tokens, rewriting their table entries to kNullBlockId.
  /// Window and chunk groups only; a no-op elsewhere.
  virtual Status ReleaseSkipped(RequestGroupState& req, int64_t num_computed_tokens);

  /// \brief New blocks (or slots) Reserve would need right now. When
  /// `pending_hit` is set, its idle cached blocks count as consumed capacity.
  virtual int64_t NumNewBlocksNeeded(const RequestGroupState& req, const GrowthTarget& target,
                                     const CacheHit* pending_hit = nullptr) const = 0;

  /// \brief Grows the request's holdings to cover `target`; all-or-nothing.
  /// Pads positions below the retention floor with kNullBlockId.
  virtual StatusOr<Granted> Reserve(RequestGroupState& req, const GrowthTarget& target) = 0;

  /// \brief Undoes a successful Reserve: releases what `granted` handed out
  /// and removes it from the request (multi-group atomicity).
  virtual Status Rollback(RequestGroupState& req, const Granted& granted) = 0;

  /// \brief Publishes newly completed blocks to the prefix index, starting
  /// at the request's publish cursor. Skips null (rolled-off) entries.
  /// Encoder groups never publish; recurrent groups publish only at state
  /// boundaries (phase E: no-op).
  virtual Status PublishCompleted(RequestGroupState& req, absl::Span<const BlockHash> hashes,
                                  int64_t num_computed_tokens) = 0;

  /// \brief Releases the request's holdings and resets its state. Blocks
  /// are released tail first, so the free queue evicts tails before shared
  /// prefixes; slots return to the state pool.
  virtual Status Release(RequestGroupState& req) = 0;

  /// \brief Forks `parent` into `num_children` children. Token-wise:
  /// children share the parent's blocks (AcquireBlock); the first Reserve
  /// that writes a shared partial block duplicates it (a BlockCopy in
  /// Granted). Recurrent: fresh slots plus one SlotCopy per child for the
  /// runner to perform.
  virtual Status Fork(const RequestGroupState& parent, int64_t num_children,
                      std::vector<RequestGroupState>* children,
                      std::vector<SlotCopy>* slot_copies) = 0;

  /// \brief Tokens below the retention floor this group skips at
  /// num_computed_tokens. Feeds window_left.
  virtual int64_t NumSkippedTokens(int64_t num_computed_tokens) const;
};

using CacheGroupManagerPtr = std::unique_ptr<CacheGroupManager>;

/// \brief The only way to obtain a manager. Dispatches on the retention
/// alternative: FullRetention -> FullAttentionManager, WindowRetention ->
/// SlidingWindowManager, ChunkRetention -> ChunkedLocalManager,
/// EncoderRetention -> CrossAttentionManager, RecurrentRetention ->
/// RecurrentStateManager.
///
/// Both pools are borrowed and must outlive the manager. `state_pool` may
/// be null for token-wise groups; `pool` may be null for the recurrent
/// group. InvalidArgument when a required pool is null.
StatusOr<CacheGroupManagerPtr> CreateCacheGroupManager(const CacheGroupSpec& group,
                                                       KvBlockPool* pool,
                                                       RecurrentStatePool* state_pool);

}  // namespace inferx
```

### 6.5 `include/inferx/cache/full_attention_manager.h`

No additional public methods — the behavioral contract on the base
interface:

| Base method | `FullAttentionManager` contract |
| --- | --- |
| `FindLongestCacheHit` | Walk `hashes` left to right until first miss (prefix-cache off ⇒ 0); cap at `max_blocks`; pin via `AdoptBlocks` |
| `BlocksHeldFor(t)` | `cdiv(t, block_size)` — unchanged from today |
| `AdoptCached` | Append pinned blocks to `blocks`; set `num_cached_tokens` |
| `Reserve` | Grant `[blocks.Size(), cdiv(t, B))` from `AllocateBlock`; all-or-nothing (eviction now lazy inside the pool); CoW duplicate before granting a write into a block whose `ref_cnt > 1` |
| `Release` | `ReleaseBlock` each held block; clear table |
| `Fork` | `AcquireBlock` every parent block per child; children's tables equal the parent's |
| `NumSkippedTokens` | 0 (base) |

MLA groups are instances of this manager with one `MlaLatentComponent`
per layer; nothing in the contract branches on it.

### 6.6 `include/inferx/cache/sliding_window_manager.h`

```cpp
namespace inferx::sliding_window {

/// Blocks a window of `window` tokens occupies: cdiv(window-1, B) + 1
/// (the +1 covers a window start mid-block). vLLM parity.
int64_t RetainedBlocks(int64_t window, int64_t block_size);

/// Contiguous blocks a prefix hit must cover: cdiv(window-1, B).
int64_t HitBlocks(int64_t window, int64_t block_size);

/// Tokens below the window floor: max(0, num_computed - window + 1),
/// never below sink_tokens.
int64_t SkippedTokens(int64_t num_computed, int64_t window, int64_t sink_tokens);

/// First logical token still attended at `num_computed` (the inclusive
/// window floor, clamped by the sink region).
int64_t FirstRetainedToken(int64_t num_computed, int64_t window,
                           int64_t sink_tokens);

}  // namespace inferx::sliding_window

/// final class; created by CreateCacheGroupManager only.
class SlidingWindowManager final : public CacheGroupManager { /* ... */ };
```

With sinks, a group's resident bound composes both regions:
`RetainedBlocks(W, B) + cdiv(sink_tokens, B)` leading sink blocks — the
formula `CacheSpec::MaxBlocks` documents (§6.1) and the phase-A
implementation must honor; the checked-in header's comment omits the sink
term (§6.0, delta 2).

Behavioral contract:

| Base method | `SlidingWindowManager` contract |
| --- | --- |
| `FindLongestCacheHit` | Search right-to-left for `HitBlocks(W, B)` *contiguous* hash hits ending at the prompt's last full block; adopt those plus leading blocks intersecting the sink region when `sink_tokens > 0`; never adopt past `max_blocks` |
| `BlocksHeldFor(t)` | `RetainedBlocks(W, B)` once t ≥ W; `cdiv(t, B)` before the window first clips; + leading `cdiv(sink_tokens, B)` when sinking |
| `AdoptCached` | Rewrite table entries below `FirstRetainedToken` to `kNullBlockId` as the adopted blocks are installed |
| `Reserve` | Grant up to the retention cap; release blocks that rolled off (`ReleaseBlock`), rewriting their entries to `kNullBlockId`; table length never shrinks (position math, §4.4) |
| `Release` | As full attention |
| `Fork` | As full attention (null entries copy verbatim) |
| `NumSkippedTokens` | `sliding_window::SkippedTokens(num_computed, W, sink_tokens)` |

Publication (facade): all full blocks publish normally; when a published
block later rolls off and its ledger reference reaches zero, `Reserve`
forgets its hash (`ForgetBlock`) so stale below-window entries do not bloat
the index.

### 6.7 `include/inferx/cache/recurrent_state_manager.h`

```cpp
namespace inferx {

/// final class; created by CreateCacheGroupManager only. Owns no per-token
/// state: one slot per request, boundaries defined by
/// spec.state_boundary_tokens.
class RecurrentStateManager final : public CacheGroupManager {
 public:
  // Base contract:
  //   FindLongestCacheHit -> 0 (phase E: no cross-request state sharing).
  //   BlocksHeldFor(t)    -> 1 slot when t > 0.
  //   Reserve             -> grant slot if none; the slot is NOT reset here
  //                          (see SlotsNeedingReset); no-op when held.
  //   Release             -> return slot; -1 again.
  //   Fork                -> fresh slots per child; the device-side state
  //                          copy is the runner's job (below).
  //   IsTokenWise         -> false.

  /// \brief Slots assigned since the last drain; the facade collects them
  /// for the runner (CacheManager::TakeSlotsNeedingReset), which calls
  /// RecurrentStatePool::ResetSlot stream-ordered before its kernels read.
  absl::Span<const int64_t> SlotsNeedingReset() const;
  void ClearResetPending();

  /// \brief Slot pairs the runner must device-copy for Fork: one
  /// SlotCopy per child (§6.4). Drained alongside resets.
  absl::Span<const SlotCopy> ForkCopies() const;
  void ClearForkCopies();
};

}  // namespace inferx
```

Phase-E scope, explicitly: no cross-request state sharing (hybrid prefix
adoption therefore yields zero for hybrid models), and preemption recomputes
from scratch — `Release` frees the slot like any other resource. Boundary
resume and shared boundary-state caching are §9 items, not implied.

### 6.8 `include/inferx/cache/cache_coordinator.h`

```cpp
namespace inferx {

/// \brief One request's reconciled cache hit.
struct CacheHit {
  /// Agreed cached token count: min across groups (0 when any group misses).
  int64_t cached_tokens = 0;
  /// Per token-wise group: pinned blocks in logical order (group order).
  std::vector<std::vector<int32_t>> group_blocks;
  bool Empty() const { return cached_tokens == 0; }
};

/// \brief Owns the group list; turns per-group decisions into one
/// per-request answer. Created by CacheManager; not used directly by the
/// scheduler or runner.
class CacheCoordinator {
 public:
  /// \brief Validates the plan (token-wise groups before the recurrent
  /// group, block size uniform) and instantiates one manager per group.
  static StatusOr<std::unique_ptr<CacheCoordinator>> Create(
      const CachePlan& plan, KvBlockPool* pool,
      RecurrentStatePool* state_pool /* null when no recurrent group */);

  int64_t NumGroups() const;
  bool HasAtMostOneTokenWiseGroup() const;  // Mirrors CachePlan; fast path.
  const CacheGroupManager& Group(int64_t group_index) const;
  CacheGroupManager* MutableGroup(int64_t group_index);

  /// \brief Reconciled longest hit over the prompt's per-group hashes.
  ///
  /// Unitary: single FindLongestCacheHit call, no allocation for the
  /// fixed point. Hybrid: query every token-wise group, take the minimum
  /// (full-attention groups first), trim every group's adoption to the
  /// minimum, then intersect with the recurrent group (zero in phase E).
  /// Two passes suffice: hits only shrink.
  CacheHit FindLongestCacheHit(
      const std::vector<BlockHash>& chain,
      absl::Span<const TokenId> prompt_tokens,
      int64_t max_cached_tokens) const;

  /// \brief Per-group orchestration used by the facade. All-or-nothing
  /// across groups: dry-run BlocksHeldFor on every group, then commit.
  Status ReserveAll(std::vector<RequestGroupState>* req_states,
                    int64_t target_tokens,
                    std::vector<std::vector<int32_t>>* granted_per_group);
  Status ReleaseAll(std::vector<RequestGroupState>* req_states);
  Status ForkAll(const std::vector<RequestGroupState>& parent,
                 int num_children,
                 std::vector<std::vector<RequestGroupState>>* children);

  /// \brief Max skipped tokens across groups at `num_computed_tokens`
  /// (the value kernels' window_left must respect). 0 for unitary full
  /// attention.
  int64_t MaxSkippedTokens(
      const std::vector<RequestGroupState>& req_states,
      int64_t num_computed_tokens) const;
};

}  // namespace inferx
```

### 6.9 `include/inferx/cache/cache_manager.h` (modified) — facade

```cpp
namespace inferx {

class CacheManager {
 public:
  /// \brief Result of admitting a prompt to the prefix index.
  struct PrefixAdoption {
    /// Agreed whole-block cached tokens (min across groups).
    int64_t cached_tokens = 0;
    /// Group-0 block ids in logical order — the unitary view existing
    /// callers consume. Empty when prefix caching is off or nothing hit.
    std::vector<int32_t> block_ids;
    /// Per token-wise group block ids; size 1 and equal to block_ids for
    /// unitary models. Hybrid callers (phase D) read this.
    std::vector<std::vector<int32_t>> group_block_ids;
  };

  /// \brief Prefix-caching counters, moved from SchedulerStats bookkeeping
  /// so hybrid reuses one struct.
  struct Stats {
    int64_t prefix_queries = 0;
    int64_t prefix_hits = 0;
    int64_t prefix_queried_tokens = 0;
    int64_t prefix_hit_tokens = 0;
  };

  /// \brief Legacy constructor: one implicit group over the whole pool, no
  /// recurrent state (the pool must have been created by a legacy Create).
  /// Keeps every existing call site compiling through phases A-C.
  explicit CacheManager(KvBlockPool* pool, bool enable_prefix_caching = false);

  /// \brief Target constructor: manager per planned group.
  ///
  /// `state_pool` may be null when the plan has no recurrent group.
  CacheManager(KvBlockPool* pool, RecurrentStatePool* state_pool,
               const CachePlan& plan, bool enable_prefix_caching);

  ~CacheManager();
  CacheManager(const CacheManager&) = delete;
  CacheManager& operator=(const CacheManager&) = delete;

  // -- unchanged public contracts (docs from today's header carry over) -----

  /// Same contract as today: at most once per request, before any Reserve,
  /// full prompt, capped by `max_cached_tokens`, truncated to whole blocks,
  /// records chains for later publishing, salt namespaces the chain.
  PrefixAdoption AdoptPrefix(uint64_t request_id,
                             absl::Span<const TokenId> prompt,
                             std::string_view cache_salt,
                             int64_t max_cached_tokens);

  /// Same contract as today: publishes every newly covered full block of
  /// prompt+output, per group, through the group's chain; epoch-guarded.
  Status PublishComputed(uint64_t request_id,
                         absl::Span<const TokenId> prompt,
                         absl::Span<const TokenId> output);

  /// Same contract as today: all-or-nothing growth, never shrinks,
  /// ResourceExhausted only when eviction cannot help.
  Status Reserve(uint64_t request_id, int64_t target_tokens);

  /// Group-0 view (unitary mapping); invalidated by growth/release.
  absl::Span<const int32_t> Blocks(uint64_t request_id) const;

  /// Unknown/already-released ids are no-ops, as today.
  Status Release(uint64_t request_id);

  /// Same semantics as today: discoverability off; in-flight requests never
  /// republish.
  void ResetPrefixCache();

  // -- additions -------------------------------------------------------------

  /// \brief Token-wise group count (hybrid callers branch on this).
  int64_t NumGroups() const;

  /// \brief One group's block table for a request. Group 0 equals Blocks().
  absl::Span<const int32_t> GroupBlocks(uint64_t request_id,
                                        int64_t group_index) const;

  /// \brief One group's skipped tokens at `num_computed_tokens` (the
  /// runner computes window_left per layer from this; 0 for full groups).
  int64_t GroupSkippedTokens(uint64_t request_id, int64_t group_index,
                             int64_t num_computed_tokens) const;

  /// \brief Forks a live request (beam-style branching). Children share
  /// token-wise blocks (ref-counted, CoW on diverging writes) and copy
  /// recurrent state (runner-drained). Children must be fresh ids; the
  /// parent stays live. Each child must subsequently go through
  /// PublishComputed with its own token stream.
  Status Fork(uint64_t parent_request_id,
              absl::Span<const uint64_t> child_request_ids);

  /// \brief Slots the runner must reset (and fork state copies it must
  /// make) before this step's kernels run. Drains the pending sets.
  void TakeSlotsNeedingReset(std::vector<int64_t>* slots);
  void TakeForkStateCopies(std::vector<SlotCopy>* copies);

  // -- counters (same names as today) ----------------------------------------

  bool PrefixCachingEnabled() const;
  int64_t BlockSize() const;
  int64_t FreeBlocks() const;
  int64_t TotalBlocks() const;
  float Usage() const;
  int64_t CachedPrefixBlocks() const;
  const Stats& stats() const;
};

}  // namespace inferx
```

Private shape (for implementers): `KvBlockPool* pool_`;
`std::unique_ptr<CacheCoordinator> coordinator_`;
`absl::flat_hash_map<uint64_t, std::vector<RequestGroupState>> requests_`;
per-request chains `std::vector<BlockHash>` extended through `ChainHash`
and hashed-block counts; `Stats stats_`.

### 6.10 `include/inferx/cache/recurrent_state_pool.h` (modified) — state slots

The pool keeps its lifecycle rules and `ResetSlot` sequencing semantics; its
API becomes component-based to match §6.1 — the geometry-specific
`State`/`ConvState` views and the include of the deleted
`models/requirements.h` go away with the migration:

```cpp
class RecurrentStatePool {
 public:
  /// \brief One fixed-slot region per component, shared by every recurrent
  /// layer (the recurrent group's spec.components).
  static StatusOr<RecurrentStatePool> Create(
      const std::vector<CacheComponent>& slots, int64_t max_slots, Device device);

  /// \brief One component's byte region: [max_slots, bytes_per_slot] uint8.
  /// The GDN family reshapes it into the delta-rule / conv-ring geometries;
  /// the pool never interprets bytes.
  StatusOr<Tensor> ComponentView(int64_t component) const;

  /// \brief Unchanged: zeroes one slot across every component,
  /// stream-ordered, before the step's kernels read it.
  Status ResetSlot(ops::OpContext& ctx, int64_t slot);

  int64_t NumComponents() const;
  int64_t MaxSlots() const;
  size_t Bytes() const;
};
```

Usage split unchanged: the `RecurrentStateManager` assigns slots; the runner
drains `CacheManager::TakeSlotsNeedingReset` and calls `ResetSlot`
stream-ordered before the step's kernels.

### 6.11 Model, config, and engine integration interfaces

**`CacheRequirements` (moved to `cache_spec.h`; renamed from
`StateRequirements`, with `ModelSpec::state` renamed `cache`)** — the
planner input is a per-layer variant in the cache layer (§6.1): each of
`layers` is a
`NoCache`, a `CacheSpec` (storage + retention), or a
`SharedCache{source_layer}` (YOCO/Gemma-3n aliasing). `model_spec.h`
includes `inferx/cache/cache_spec.h` — models depending on cache is the
correct layer direction and removes the old upward include of
`models/requirements.h`. Family configs fill the variant from checkpoint
fields they already parse: qwen3/llama build
`CacheSpec::FullAttention({GqaKvComponent(...)})` (or `SlidingWindow(...)`
once phase C opens the gate); GDN families build `Recurrent` with
`FixedSlot` components and `boundary_tokens`.

**`SchedulerOutput` (modified, phase D)** — additive fields, empty (and thus
wire-identical) for unitary models:

```cpp
struct NewRequestData {
  // ... existing fields unchanged; block_ids stays the group-0 view ...
  /// Per token-wise group initial tables; filled only when NumGroups() > 1.
  std::vector<std::vector<int32_t>> group_block_ids;
};

struct CachedRequestUpdate {
  // ... existing fields unchanged ...
  /// Per token-wise group granted-block diffs; filled only when > 1 group.
  std::vector<std::vector<int32_t>> group_new_block_ids;
};
```

No window metadata crosses the wire: the runner holds `ModelSpec.state` and
`num_computed_tokens`, which is everything `window_left` needs (§4.10).

**Model families (modified, phases C/E/F)** — each family's config loader
stops rejecting and starts filling the variant: `qwen3_config.cc` /
`llama_config.cc` build
`CacheSpec::SlidingWindow({GqaKvComponent(...)}, window, sink_tokens)`
instead of rejecting; the GDN family (`qwen3_next`) builds
`CacheSpec::Recurrent` with fixed-slot components and `boundary_tokens`;
the MLA family builds `CacheSpec::FullAttention({MlaLatentComponent(...)})`.
The runner gate at `model_runner.cc:266` (pure-recurrent rejection) opens in
phase E.

**ModelRunner (modified, phases D/E)** — `RunStep` flattens per group (one
CSR triple per token-wise group; `ForwardContext` carries the group index so
`models::Attention::Forward` selects the right tables), drains
`TakeSlotsNeedingReset`/`TakeForkStateCopies` before `BeginAttentionStep`,
and passes per-request slot ids to the GDN ops. `AllocateKvPool` builds the
`CachePlan` from `ModelSpec.state` and `CacheConfig::block_size`, then calls
the group-based `KvBlockPool::Create`.

## 7. Implementation plan

### 7.1 File map and library layout

The cache layer lands as per-concept targets in `src/cache/CMakeLists.txt`
(`inferx::sha256`, `inferx::cache_spec`, `inferx::kv_block_pool`,
`inferx::recurrent_state_pool`, `inferx::cache_group_manager`,
`inferx::cache_manager`), mirrored `include/inferx/cache/` + `src/cache/`:

| File | Contents | Phase |
| --- | --- | --- |
| `cache_spec.h/.cc` | `CacheComponent`, `Retention` variant, `CacheSpec`, `CachePlan`, `PlanCacheGroups` | A (done) |
| `kv_block_pool.h/.cc` (modified) | ledger, null block, absorbed index; group regions pending | A (mostly done) |
| `block_hash.h` (+ impl in `cache_manager.cc`) | `BlockHash`, `ChainHash`; `prefix_cache.h/.cc` and `BlockHashChain` deleted | A (done) |
| `cache_group_manager.h/.cc` | base class, `RequestGroupState`, factory, all five managers (full/window/chunk/cross collocated; no per-manager files) | B/E (done) |
| `recurrent_state_pool.h/.cc` | slot storage, `ComponentView`, `ResetSlot` | E (done) |
| `cache_coordinator.h/.cc` | reconciliation (currently a stub; the facade carries the per-group loop) | B (open) |
| `cache_manager.h/.cc` (modified) | facade over the group managers | A→D (done at unit level) |

Dependency order inside `inferx_cache` is flat (spec → pool → managers →
coordinator → facade); no layer outside `core`/`models/requirements.h` is
pulled in. `recurrent_state_pool.h` already depends on `ops/op_context.h`
and stays as is.

### 7.2 Phase A — ledger pool, absorbed index, build fixes

Steps in order:

1. Delete the dangling `add_subdirectory(src/kvc)` (`CMakeLists.txt:41`),
   `plugins/kvcache/cuda` (`:45`), `tests/kvc` (`:60`), the
   `INFERX_BUILD_KVCACHE_PLUGINS` option, and the `plugins/kvcache/host`
   stub (§7.8). The tree must configure before anything else lands.
2. Finish the `cache_spec.h` landing (header is in the tree; deltas in
   §6.0): write `cache_spec.cc` — `Validate`, `MaxBlocks` with the sink
   term composed, `ToString`, the plan helpers; drop the unmappable
   `CrossAttention()` factory; add the planner's `ChunkRetention`/
   `EncoderRetention` gates. Migrate everything orphaned by the header's
   ahead-of-code deletions: `recurrent_state_pool.h` (to the §6.10 API)
   and `llama_config.h` off `models/requirements.h` (`model_spec.h`
   already migrated with the `CacheRequirements` rename, member
   `spec.state` → `spec.cache`); `src/cache/kv_block_pool.cc` off
   `KvLayout`; qwen3/llama families, the runner, and the diagnostic fill
   `spec.cache` with `CacheSpec::FullAttention({GqaKvComponent(...)})`.
   The tree compiles again at the end of this step.
3. `KvBlockPool`: add `BlockRecord` array, intrusive free queue + LRU,
   `kNullBlockId` reservation (usable ids 1..num_blocks), group-aware
   `Create` (the legacy overload wraps one implicit group), and the
   absorbed index methods. Delete `free_list_`/`in_use_`. `AllocateBlock`
   becomes eviction-aware; `CacheManager::EvictFor` is deleted (its tests
   move to the pool).
4. (done 2026-10-19) `prefix_cache.h/.cc` deleted; `BlockHashChain` folded
   into `ChainHash` (`block_hash.h`, implementation in `cache_manager.cc`);
   `prefix_cache_test.cc` ported case-for-case to `sha256_test.cc` (FIPS
   vectors, incremental), `kv_block_pool_test.cc` (pin safety, LRU order,
   tombstones-through-clear, double-free, duplicate hashes as the documented
   single-binding rebind), and `cache_manager_test.cc` (chain determinism,
   salt namespacing, mutation isolation, block-boundary framing).
5. `CacheManager`: rewire `AdoptPrefix`/`PublishComputed`/`Reserve`/
   `Release` onto pool methods, keeping the public contract and the epoch
   guard; keep the legacy constructor (implicit single group).

Gate: the existing `tests/unit/cache/*` and `tests/unit/engine/scheduler*`
suites pass with **no scheduler source changes**; unitary behavior is
byte-identical (block ids shift by the null block — pool-size constants in
tests adjust, productions sizes are large enough that one block is noise).

### 7.3 Phase B — managers and coordinator

1. `cache_group_manager.h/.cc`: base + factory; `FullAttentionManager`
   holds the logic phase A left in `CacheManager` (growth, adoption install,
   CoW duplication hook).
2. `cache_coordinator.h/.cc`: unitary fast path + `ReserveAll`/`ReleaseAll`/
   `ForkAll` orchestration (dry-run-then-commit across groups).
3. `CacheManager` delegates to the coordinator; legacy constructor builds a
   single-group plan internally.

Gate: same suites green; new `full_attention_manager_test.cc` and
`cache_coordinator_test.cc` (dry-run atomicity, unitary fast path equality
with direct manager calls).

### 7.4 Phase C — sliding window

1. `sliding_window_manager.h/.cc` with the four free functions; table
   rewrite-to-null semantics; hash-forget on rolloff (§6.6).
2. Family configs: `qwen3_config.cc` / `llama_config.cc` build
   `CacheSpec::SlidingWindow({GqaKvComponent(...)}, window, sink_tokens)`
   instead of rejecting; `PlanCacheGroups` validates (window/sink/boundary
   alignment rules).
3. Runner/models: per-layer `AttentionParams.sliding_window` flows from the
   model's per-layer `WindowRetention`; `window_left` computed from
   `num_computed_tokens`; kernels add the `kNullBlockId` skip guard
   (`src/ops/cuda/flash_attention.cu`, `generic_attention.cu`,
   `src/ops/attention.cc` host paths).
4. FX-swa / FX-sink fixtures (F13) drive the T04 P-02/P-03 cases.

Gate: T04 P-02 and P-03 case lists pass; P-01 suite unchanged (full
attention untouched).

### 7.5 Phase D — hybrid token-wise groups

1. `SchedulerOutput` grows `group_block_ids`/`group_new_block_ids` (§6.11);
   scheduler fills them when `NumGroups() > 1` from `GroupBlocks`.
2. `ModelRunner::RunStep` flattens one CSR triple per group;
   `ForwardContext` carries the group index; `models::Attention::Forward`
   selects its group's tables; `LayerCache(layer)` resolves through
   `GroupForLayer` and is reshaped into kernel geometry.
3. Hybrid admission: `ReserveAll` prices the max across groups; planner
   rejects differing page-size demands (already §6.1).
4. FX-hyb fixture; gpt-oss family registration (config parse →
   `CacheRequirements` with alternating windows + sinks) behind the same
   gate as its parity cases.

Gate: T04 P-05 cases; unitary wire format unchanged (asserted by a
regression test).

### 7.6 Phase E — recurrent state

1. `recurrent_state_manager.h/.cc`: slot grant/release, reset-pending and
   fork-copy queues (§6.7).
2. `CacheManager` target constructor (pool + state pool + plan);
   `TakeSlotsNeedingReset` / `TakeForkStateCopies`; scheduler passes the
   state pool it obtains beside `KvPool`.
3. Runner: drain resets/copies before `BeginAttentionStep`; per-request
   slot ids into `ForwardContext` for GDN ops; `RecurrentStatePool` sized
   from `max_num_seqs`.
4. GDN family (`qwen3_next`): config → `CacheSpec::Recurrent` fixed-slot
   components + `boundary_tokens`; open the pure-recurrent gate at
   `model_runner.cc:266` last, after the hybrid path is proven.

Gate: T04 P-07 subset (slot lifecycle, hybrid boundary behavior, per-family
shapes); explicitly out: cross-request state sharing, boundary-resume
preemption (documented R, §9).

### 7.7 Phase F — MLA execution (F12) and phase G — fork

F: latent-layout groups flow from the family config; absorbed decode or
`AssembleMlaCaches` wiring per F12; TP-replicated latent heads. G:
`Fork`/CoW on the ledger; beam-style branching consumer decides whether
`Fork` stays engine-internal or surfaces in the scheduler. Both phases carry
their own T04 gates (P-06; new fork suite).

### 7.8 Build changes

Phase A step 1 (dangling references) is the only *removal*; everything else
is additive to `src/cache/CMakeLists.txt` via `inferx_add_library`. No new
dependencies; no changes to `cmake/*` helpers.

### 7.9 Test matrix

| Suite | Cases (names indicative) | Phase |
| --- | --- | --- |
| `tests/unit/cache/kv_block_pool_test.cc` | Ported `prefix_cache_test` set + ledger: `NullBlockNeverAllocated`, `RefCountZeroFreesToLifo`, `EvictionSkipsPinned`, `TombstoneReclaimsOnLastRelease`, `ClearKeepsOwnership` | A |
| `tests/unit/cache/cache_spec_test.cc` | `UniformSpecSingleGroup`, `InterleavedWindowsTwoGroups`, `WindowPromotedToFull`, `SinkNotBlockAlignedRejected`, `BoundaryNotBlockMultipleRejected`, `ParallelVectorMismatchRejected`, `EmptyStateRejected` | A |
| `tests/unit/cache/cache_manager_test.cc` | Existing suite unchanged (regression) + `AdoptionEpochGuard`, `ReserveAtomicAcrossGroups` | A/B |
| `tests/unit/cache/full_attention_manager_test.cc` | Growth/adoption/release parity with pre-refactor behavior; `CoWDuplicatesSharedPartialBlock` (G) | B |
| `tests/unit/cache/sliding_window_manager_test.cc` | `RetentionMathTable` (W vs B sweeps incl. W < B, W crossing edges), `RightToLeftHitNeedsContiguous`, `RolledOffBlocksBecomeNull`, `SinkFloorNeverReleases`, `SkippedTokensFormula` | C |
| `tests/unit/cache/cache_coordinator_test.cc` | `UnitaryFastPathMatchesManager`, `HybridMinAcrossGroups`, `DisagreeingGroupsTrimToMin`, `DryRunAtomicity` | B/D |
| `tests/unit/cache/recurrent_state_manager_test.cc` | `SlotGrantedOnce`, `ResetPendingDrainedExactlyOnce`, `ReleaseReturnsSlot`, `ForkCopiesState`, `NoCrossRequestSharing` | E |
| `tests/unit/engine/scheduler_*` | Existing suites as phase gates throughout | A–E |
| `tests/unit/models/fx_*` + integration | T04 P-02/P-03/P-05/P-06/P-07 case lists on FX fixtures | C–F |

All bookkeeping suites run on `Device::Cpu()` pools with tiny geometries
(`GqaKvComponent(1, 8, kBFloat16)`, `kBlockSize = 4`), following the
existing `tests/unit/cache` conventions.

### 7.10 Usage examples

Unitary model (phases A–C; identical to today's call sites — the scheduler
does not change):

```cpp
// Startup (ModelRunner::AllocateKvPool):
const CachePlan plan = INFERX_ASSIGN_OR_RETURN(
    PlanCacheGroups(model_spec.cache, cache.block_size, max_model_len));
KvBlockPool pool = INFERX_ASSIGN_OR_RETURN(
    KvBlockPool::Create(num_blocks, cache.block_size,
                        plan.TokenWiseGroups(), device));

// Engine (Scheduler):
CacheManager cache_manager(&pool, cache.enable_prefix_caching);
const auto adoption = cache_manager.AdoptPrefix(
    req_id, request.Prompt(), request.CacheSalt(),
    /*max_cached_tokens=*/request.NumTokens() - 1);
INFERX_RETURN_IF_ERROR(
    cache_manager.Reserve(req_id, adoption.cached_tokens + chunk));
// Runner reads Blocks(req_id) + pool.LayerCache(layer), reshaped per layer.
INFERX_RETURN_IF_ERROR(
    cache_manager.PublishComputed(req_id, request.Prompt(), request.Output()));
INFERX_RETURN_IF_ERROR(cache_manager.Release(req_id));  // finish/abort/preempt
```

Hybrid model with window and recurrent groups (phases D/E):

```cpp
// Startup:
const CachePlan plan = INFERX_ASSIGN_OR_RETURN(
    PlanCacheGroups(model_spec.cache, cache.block_size, max_model_len));
// plan.groups = [SW(W=128, sink=4·B), full, recurrent(boundary=2·B)] e.g.
KvBlockPool pool = INFERX_ASSIGN_OR_RETURN(
    KvBlockPool::Create(num_blocks, cache.block_size,
                        plan.TokenWiseGroups(), device));
RecurrentStatePool states = INFERX_ASSIGN_OR_RETURN(
    RecurrentStatePool::Create(plan.RecurrentGroup()->spec.components,
                               /*max_slots=*/sched.max_num_seqs, device));
CacheManager cache_manager(&pool, &states, plan, cache.enable_prefix_caching);

// Admission: hybrid min across groups; recurrent group reports 0, so
// adoption.cached_tokens == 0 for attention+recurrent requests in phase E.
const auto adoption = cache_manager.AdoptPrefix(req_id, prompt, salt, cap);
INFERX_RETURN_IF_ERROR(cache_manager.Reserve(req_id, target_tokens));

// Step (ModelRunner, before kernels):
std::vector<int64_t> slots;
cache_manager.TakeSlotsNeedingReset(&slots);
for (const int64_t slot : slots) {
  INFERX_RETURN_IF_ERROR(state_pool.ResetSlot(op_ctx, slot));  // stream-ordered
}
// Per group g: flatten GroupBlocks(req_id, g) into that group's CSR; the
// window group's kernel gets window_left from GroupSkippedTokens(...).

// Cleanup / branching:
INFERX_RETURN_IF_ERROR(cache_manager.Fork(parent_id, {child_a, child_b}));
INFERX_RETURN_IF_ERROR(cache_manager.Release(req_id));
```

## 8. Trade-offs

- **One ledger, absorbed `PrefixCache`.** Folding the hash index into the
  pool churns ~300 tested lines to eliminate the two-counter bug class
  (prefix refs vs pool in-use) that fork sharing would otherwise force into
  existence. The alternative (keep `PrefixCache` beside a pool-level count)
  was rejected: two ledgers for one ownership fact.
- **Uniform page size in phase 1.** One scheduler block size across groups
  forfeits memory for the unlikely model whose groups want different page
  sizes; it keeps `BlockTable::Locate`, slot math, the CSR format, and the
  n−1 rule single-shaped. The planner *rejects* rather than silently
  mis-prices such models; per-group sizes are §9 with the planner already
  producing the group metadata they need.
- **Bytes-only components, model-shaped views.** The pool stopped
  understanding geometry; generality (quantized payloads, indexers, latents
  as plain bytes) trades away pool-side view shaping. Model code pays one
  `Reshape`/`Bitcast` per layer view and owns the shape's correctness — the
  same trade vLLM made with opaque `state_content_size_bytes`.
- **Null block over table compaction.** Keeping window-rolled entries in
  place (pointing at id 0) preserves position math in every kernel and
  CUDA-graph buffer; compacting tables saves a few ids per sequence and
  costs a rewrite of every index computation. vLLM made the same call.
- **Manager granularity = spec kind, not model.** Windows differing only in
  size are different groups (vLLM parity — different retention math); a
  model with three distinct windows is three groups, priced honestly at
  planning time.
- **Facade-owned hashing, manager-owned retention.** Chains and publication
  stay in `CacheManager` so managers stay small and the hash format changes
  once, in one file. The cost is that managers see hashes, not tokens, for
  hit search — which is all they need.
- **Recurrent phase E is deliberately narrow.** No cross-request state
  sharing and recompute-on-preempt keep the first recurrent milestone
  honest ("do not claim support unless semantics are implemented"); the
  coordinator's min-across-groups already yields the correct conservative
  behavior for hybrids. Boundary sharing is a contained §9 extension (it
  needs a state-keyed index, not structural change).
- **Fork via ref counts + lazy CoW, not eager copy.** Beam branching shares
  clean block-aligned prefixes and pays a copy only when a child writes a
  shared *partial* block — the common case (decode past the branch point)
  allocates fresh tail blocks and copies nothing.
- **No watermark admission.** InferX's preempt-newest-recompute under FCFS
  reaches a similar steady state; a watermark knob is trivially added to
  `Reserve` later without API change.
- **SHA-256 chains stay.** Slower than xxhash; deterministic across builds
  and processes, self-contained (`sha256.cc`), and profiled as noise next to
  the per-step GPU work. Recorded non-goal (P-11).

## 9. Future extension points

Ordered by expected value; each has a seam today and no consumer yet:

1. **Per-group page sizes** — planner already emits per-group metadata;
   `BlockTable` becomes per-group block-size aware.
2. **CUDA-graph block-table buffers** (A12) — the per-group flat CSR pins
   into the fixed buffer `BlockTable`'s header already reserves.
3. **KV connectors / offload** — the facade's publish/release points and the
   ledger's ref counts are the attach points; the no-async-lease contract is
   the boundary to relax first.
4. **Recurrent boundary-state sharing** — a state-keyed index over boundary
   states (hash of tokens up to a boundary → slot snapshot), enabling hybrid
   prefix adoption and preemption rewind-to-boundary; the manager interface
   already isolates `FindLongestCacheHit`.
5. **Cross-layer KV sharing** (YOCO/CLKV) — already planned as
   `SharedCache`/`LayerAlias` (§6.1); the remaining work is pool/runner
   aliasing of `LayerCache(layer)` onto the source layer's region.
6. **Quantized KV write path** (P-10) — `CacheComponent` bytes already size
   storage for any dtype; the op-layer scatter/grow is the missing half.
7. **Eviction policy knobs** (retention intervals, LFU experiments) behind
   the pool's `EvictCached`; metrics counters already flow through
   `CacheManager::Stats` into `SchedulerStats`.
8. **Speculative decoding interplay** (A13) — draft/verify block granting
   rides `Reserve`'s target-token headroom; `Fork` is the draft-branch
   primitive.

## Appendix A: vLLM → InferX concept map

| vLLM (`b8b302cde4`) | InferX (this design) |
| --- | --- |
| `KVCacheSpec` family | `CacheSpec` + `Retention` variant |
| `AttentionSpec.state_content_size_bytes` | `CacheComponent` bytes |
| `get_kv_cache_spec` per attention module | family configs fill `CacheRequirements` |
| `get_kv_cache_groups` | `PlanCacheGroups` |
| `KVCacheGroupSpec` | `CacheGroupSpec` |
| `KVCacheConfig` / `KVCacheTensor` | `CachePlan` + `KvBlockPool` group regions |
| `KVCacheBlock` + `FreeKVCacheBlockQueue` | `BlockRecord` + intrusive queues in `KvBlockPool` |
| `BlockPool` (+ `BlockHashToBlockMap`) | `KvBlockPool` with absorbed prefix index (was `PrefixCache`) |
| `BlockHash` chain / extra keys | `ChainHash` (salt in first link; group id in the pool's `BlockKey`) |
| `SingleTypeKVCacheManager` subclasses | `CacheGroupManager` subclasses |
| `find_longest_cache_hit` | `CacheGroupManager::FindLongestCacheHit` |
| `KVCacheCoordinator` (unitary/hybrid) | `CacheCoordinator` |
| `KVCacheManager` | `CacheManager` (facade, API-preserving) |
| `MambaManager` / `MambaSpec` | `RecurrentStateManager` / `RecurrentStatePool` |
| `kv_sharing_target_layer_name` (SharedBlockManager successor) | `SharedCache` / `LayerAlias` |
| `SinkFullAttentionSpec` | `WindowRetention.sink_tokens` |
| `null_block` | `KvBlockPool::kNullBlockId` |
| `CommonAttentionMetadata` (block table, slot mapping) | per-group CSR (`kv_indices`, `kv_indptr`, `last_page_len`) + `positions` |
| `determine_available_memory` / `get_kv_cache_configs` | `RunProfileStep` + spec-aware `AllocateKvPool` |
| `KVConnectorBase_V1` | none (future; §9.3) |
