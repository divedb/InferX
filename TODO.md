# KV-cache parity TODO — InferX vs vLLM / SGLang

Revision 2 (2026-10-14). Gap analysis derived from a line-level review of
vLLM main (v0.31.0-era sources, reviewed 2026-10-14) and SGLang main @
`4560da58` (v0.5.21, reviewed 2026-10-14), against InferX's two cache layers:

- **Engine layer** — `KvBlockPool` (one allocation, per-layer heterogeneous
  layouts, LIFO free list), `CacheManager` + `PrefixCache` (SHA-256
  block-hash chain, refcounted index, LRU lazy eviction), `Scheduler`
  (FCFS, chunked prefill with block-aligned clipping under caching,
  recompute preemption newest-first, token-granular hit stats).
- **Provider layer** — the `kvc` SDK 0.1 (C ABI + C++ facade + canonical
  identity helper), host and CUDA tiered reference providers with
  single-session in-flight write dedup, `kvc.stats`, and an in-flight
  `kvc.management` extension. The engine's serving loop does **not** call
  providers (P1-6 below).

This revision supersedes revision 1 (commit `c165c43`). `docs/kvcache_plan.md`
was removed from the working tree; its §27 68-item catalog is recoverable
from the git index and is referenced below as "plan #N". Each item is one
deliverable: implementation + tests + a dedicated commit. Status is updated
as work lands.

## Corrections to revision 1

- **`KvCacheBridge` never existed.** Rev 1 (P1-6) claimed the bridge "exists
  and is unit-tested". It does not exist in the tree or in git history; only
  `docs/extensions/kvcache.md:273` references `inferx/cache/kv_bridge.h`, and
  the documented `inferx run-bench` provider-driving command is equally
  phantom (only throughput/latency bench commands exist). The real — and
  only — integration point today is `adapters/inferx/provider_memory.cc`
  (`ImportProviderMemory`), exercised solely by `tests/kvc/adapter_test.cc`.
  P1-6 therefore includes building the bridge, not just wiring it.
- **The engine does not use the P0-2 identity helper.** The engine hashes
  with its own `inferx.kv.blockhash.v1` chain (block-size-bound, tokens
  only, salt in the seed) while the SDK ships the portable
  `kvc.token.v1` token-level chain with an auxiliary-digest slot. Two
  divergent chains — new P1-7 unifies them.
- **Uncommitted work in the tree** (on top of `e54303e`): the
  `kvc.management` extension (header + host/CUDA providers + conformance
  test) and an engine fix dropping preempted request state in
  `ModelRunner` (`requests_.erase` for `preempted_request_ids`). Tracked
  under P2-8; commit independently of the engine work.

## Landed in revision 1 (kept for history)

| Item | Commit | Summary |
|------|--------|---------|
| P0-1 prefix caching | `3d28da5` | vLLM-style block-hash chain, refcounted LRU index, adoption (n−1 cap), lazy eviction, salt |
| P0-2 kvc identity | `6b2208f` | domain-separated SHA-256, token-level prefix chain, block keys, golden vectors |
| P1-3 preemption | `c91d0e2` | recompute-only, newest-first victim, cached prefixes survive, `kPreempted` surfaced |
| P1-4 hit accounting | `ce2dec6` | token-granular queries/hits counters in `SchedulerStats` |
| P1-5 chunk alignment | `60bedef` | prefill chunks clip to block boundaries under caching |
| P2-7 in-flight dedup | `e54303e` | single-session attach-to-founding-transaction in all three providers |

## Priorities and dependencies

```text
P1-6 engine↔provider loop ──┬─> P2-15 persistence   P2-14 host eviction (quick)
 (bridge; nothing uses       ├─> P2-16 remote/L3     P2-8  management (in flight)
  providers today)           └─> P3 PD disagg
P1-7 engine identity unification ──┬─> P1-10 fine-grained boundaries/CoW
                                  └─> P2-11 cross-block-size reuse
P1-8/9/11 engine behaviors ── independent
P2-18 SWA / P2-19 recurrent ── triggered by model ports (gpt-oss staged)
```

Citation note: both engines reorganized heavily during 2026. vLLM's
`hybrid_kv_cache_manager.py` split into `v1/core/kv_cache_coordinator.py` +
`single_type_kv_cache_manager.py`; connectors moved to
`distributed/kv_transfer/kv_connector/`; `SharedStorageConnector` was
replaced by the `v1/kv_offload/` framework. SGLang's
`hierarchical_cache.py`/`hicache.py` became `mem_cache/hybrid_cache/` +
`pool_host/` + `storage/`; `evictor.py` is now `evict_policy.py`;
`token_kv_cache.py` folded into `radix_cache.py` (page_size=1); the
standalone SWA/Mamba trees were deleted in favor of `UnifiedRadixCache`
components with a Rust tree core (default since v0.5.21).

---

## P0 — the blocker: nothing calls providers

### P1-6. Engine↔provider publication loop (`KvCacheBridge` in serving)
**Status: open** — *elevated from P1: this is the plan's own "second
blocker"; every sharing/offload/persistence use case is dead code until it
lands.*

The engine never calls providers from its serving loop: zero `kvc::`
references in `src/` or `apps/`; `EngineGateway` owns only
`Scheduler` + `ModelExecutor`. vLLM routes everything external through the
KVConnector lifecycle (scheduler side: `get_num_new_matched_tokens` →
`update_state_after_alloc` → `build_connector_meta` →
`update_connector_output` → `request_finished`; worker side:
`start_load_kv` → `wait_for_layer_load` → `save_kv_layer` →
`wait_for_save` → `get_finished`, plus `handle_preemptions` and
`has_pending_block_frees` so transfer completion can replace preemption).
SGLang hooks HiCache into the scheduler event loop (`HybridCacheController`:
L2 transfer engine, write-through with node-anchored locks, L3 prefetch
tickets, `checkpoint()` at every chunk boundary).

Build `KvCacheBridge` and wire an optional provider session into
`EngineGateway`/`ModelRunner`:

- Lookup before admission; provider hits merge with local `PrefixCache`
  hits (longest wins; both counted in stats with source attribution).
- Async load into allocated blocks with scheduling deferred until the
  transfer completes (vLLM's `reserved_blocks` + `full_sequence_must_fit`
  admission pattern) — the CUDA provider's `poll`/`wait` and the async
  fixture exist for exactly this.
- Publish completed blocks after each step (whole-block, post-commit
  receipt only); release pins at finish; propagate preemption
  (`handle_preemptions` analog: a victim with an in-flight provider load
  aborts the load or waits, never frees mid-transfer blocks).
- Failure policy `recompute|fail` (vLLM `kv_load_failure_policy`); a failed
  block load falls back to local compute of that extent.
- Fix the stale docs (`kv_bridge.h`, `run-bench`) as part of landing.

Edge cases: provider slower than recompute (budget/timeout on load, SGLang
`hicache-storage-prefetch-policy` timeout pattern); partial adoption
rollback on mid-layer load failure; publication dedup vs the engine's own
prefix index (avoid double-counting hit tokens); shutdown drain
(`KVC_BUSY` semantics); block-identity agreement between engine and
provider (P1-7).

Tests: end-to-end serving through the host provider; merged local+remote
hit accounting; load-failure fallback via the async fixture's `fail_store`;
preemption mid-load; no-provider default path unchanged bit-for-bit.

## P1 — engine behaviors

### P1-7. Unify engine hashing with the kvc identity chain
**Status: open**

vLLM's block hashes are block-size-*independent* by construction
(`hash_block_size` = GCD of prefix-cacheable groups; `BlockHashListWithBlockSize`
rescales one chain to any coarser block because the last sub-hash inside a
target block *is* that block's hash), which is what later enables
cross-block-size reuse, sub-block matching, and per-group granularity. Our
engine chain extends per block with `token_count` folded into the digest —
block-size-bound — and hashes token ids only. The SDK already ships the
right primitive (`kvc.token.v1`: `P(i+1) = H("kvc.token.v1", P(i), token,
position, auxiliary)`, block-size-independent, auxiliary slot for LoRA /
multimodal / prompt-embed digests — vLLM's `extra_keys` analog).

Switch `BlockHashChain`/`PrefixCache` onto `kvc::runtime` identities:
per-token chain with an auxiliary digest, salt folded via the seed (or
block-0 auxiliary — keep the current isolation semantics either way), block
digests derived as boundary values of the token chain at the configured
block size. Golden vectors for the new profile; old entries simply
invalidate (no persistence exists yet).

Edge cases: determinism across processes/builds (SHA-256, no seeded
Python-`hash`-style pitfalls); auxiliary must bind LoRA name+path and mm
identifier+offset (vLLM `("mm", id, offset)` spans multi-block items) once
those features exist — leave the slot wired now; salt-in-seed vs
salt-in-block-0 documented explicitly.

Tests: identity golden vectors; engine adoption unchanged behaviorally;
block digests at 16 vs 32 boundary consistency (same chain, two views).

### P1-8. KV watermark and admission headroom
**Status: open**

vLLM reserves a `watermark` fraction of blocks as headroom applied when
other requests are already scheduled, and v0.31 reworked the waiting queue
(#58947) so requests already holding KV blocks schedule before new
admissions — the admit→preempt→re-admit thrash we can produce today. SGLang
uses `--schedule-conservativeness` (new-token-ratio estimation feeding
retraction decisions). InferX admits new requests up to hard `Reserve`
failure and then preempts; nothing keeps running requests from being
thrashed by fresh admissions.

Add a watermark fraction to `CacheConfig`: new admissions gated when
`(used + watermark) > capacity`; growth of already-running requests and
re-admission of preempted requests is *not* gated (they already hold or
held blocks — the v0.31 #58947 rule). Keep preemption as the backstop.

Edge cases: watermark too high strands capacity (make it tunable, small
default); must not deadlock with an impossible request (P1-3's
admission-time error still fires first); interaction with prefix eviction
(`Reserve` evicts cached blocks before failing — watermark check should
count cached-but-evictable blocks as available).

Tests: pressure scenario admits fewer but preempts zero; preempted request
re-admits past the watermark; no regression at low load.

### P1-9. Cache-aware and priority scheduling policies
**Status: open**

SGLang's default-adjacent policies are cache-aware (`lpm` longest-prefix-
match over a simulated waiting-queue tree, `dfs-weight`, plus `hrrn`,
`shortest-prefill-first`, and priority scheduling with
`--disable-priority-preemption`); vLLM stays FCFS but orders preemption
victims by priority (lowest first, then largest). InferX implements FCFS
only — `SchedulerConfig::Policy` carries unused `kPriority`/`kFair`/
`kWeightedFair` stubs, and preemption is purely newest-first.

Implement `kPriority` (waiting-queue ordering + preemption victim
ordering: lowest priority, then newest) and an `lpm` policy that ranks the
waiting queue by longest prefix-cache hit when `enable_prefix_caching` is
on (bounded: SGLang re-ranks every 32 passes, caps prefix-match work at
128 waiters).

Edge cases: priority inversion under preemption (a low-priority victim
must not starve — bound requeue position); lpm recomputation cost vs
waiter count; interaction with chunked prefill budget.

Tests: priority ordering respected under pressure; preemption picks
lowest-priority-newest; lpm measurably reorders identical-prefix waiters.

### P1-10. Fine-grained prefix boundaries and partial-tail copy-on-write
**Status: open** — depends on P1-7 (token-level chain).

Only full blocks are hashable today, so prompts shorter than `block_size`
share nothing and any reuse that ends mid-block re-computes the whole tail
block. vLLM recently closed exactly this gap: `prefix_match_unit` sets a
sub-block hash granularity (validated against every group's block size);
`cache_partial_block` registers the *final prompt tail* hash inside a
not-yet-full block; a later hit on that partial boundary redirects the
shared tail block to a private CoW block (`kv_cache_block_copies` device
copies, both endpoints pinned until the copy drains). SGLang covers the
same ground by running the radix tree at `page_size=1` where the model
allows it.

Deliver in two steps: (a) hash-boundary granularity decoupled from
allocation block size — publish/adopt at the fine boundary while adoption
still snaps allocation to whole blocks (hit accounting already
token-granular from P1-4); (b) partial-tail publication of the last prompt
block with CoW privatization when a second request extends a shared tail.

Edge cases: CoW copy must be ordered before the diverging writes (device
copy queue, pin both endpoints); n−1 recompute rule enforced at token
granularity; duplicate partial-hash replacement (vLLM
`replace_existing_hashes` for re-keying); no double-counting when the same
request re-adopts post-preemption (P1-3 invariant preserved).

Tests: two prompts differing inside the final block share everything
before the divergence point via CoW; sub-block hit tokens counted
correctly; copy-before-write race pinned with the async fixture.

### P1-11. Surface the cache controls (CLI, API, endpoints)
**Status: open** — cheap; do alongside P2-8.

Config fields exist but are unreachable: `--enable-prefix-caching` and
`--gpu-memory-utilization` are not wired in `apps/cli/args/engine_args.h`
(only `--num-kv-blocks/--block-size/--kv-cache-memory-bytes` are);
`Request::cache_salt` has a setter nothing calls (no server/gateway path
sets it — the multi-tenant isolation feature is engine-internal only).
vLLM exposes `cache_salt` per request, `POST /reset_prefix_cache?
reset_running_requests=&reset_external=` and `/reset_mm_cache`; SGLang
exposes `/flush_cache` and `--enable-cache-report` (per-request
`cached_tokens` in responses).

Wire the two CLI flags; plumb `cache_salt` through the server request
schema; with P2-8, add a flush/reset endpoint that clears both the engine
prefix cache and the attached provider (when P1-6 lands); expose
`cached_tokens` per request in responses behind a flag.

Tests: flag toggles behavior end-to-end; salt isolation through the API;
flush makes the next identical prompt recompute.

## P2 — provider/ABI work

### P2-8. Management surface: enumeration, bulk clear, namespace reset
**Status: done (providers; engine surface under P1-11)**

`kvc.management` v1.0 landed: `enumerate` (null-out sizing, record
fidelity) and idempotent `clear` (pinned generations survive; the CUDA
provider reclaims sole-owner storage under the session lock so block
destruction cannot re-take it), in both bundled providers, with
conformance tests. `SessionConfigure` now negotiates
`required_extensions` properly (stats/management acceptable, unknown
schemas `Unsupported`), pinned by `TestRequiredExtensionNegotiation`.
`PrefixCache::Clear()` under live adoptions is pinned by a unit test
(pinned blocks stay allocated; release-after-clear frees them). Engine
flush/reset endpoint remains under P1-11.

Prior art for semantics: vLLM `reset_prefix_cache` refuses while running
requests wait on remote transfers and can optionally preempt-first;
SGLang `/flush_cache` takes a timeout and clears storage tiers, emitting
`AllBlocksCleared`.

### P2-14. Host-provider eviction (quick win)
**Status: open**

The host reference provider hard-fails `KVC_RESOURCE_EXHAUSTED` at its
64 MiB budget; it records LRU ticks on hit but never reclaims. Every tier
in both engines evicts (vLLM CPU offload `lru|arc`; SGLang host pools LRU
per component with duplicate-copy reclaim). Add LRU reclaim on budget
pressure to the host provider (victim = committed, unpinned, oldest tick),
mirroring the CUDA provider's `MakeHostRoomLocked` pattern it already
 interoperates with.

Tests: budget pressure evicts LRU-unpinned first; pins survive; lookups
miss after eviction; stats counters move.

### P2-9. Publication event feed
**Status: open**

vLLM's `KVEventsConfig` publishes `BlockStored`/`BlockRemoved`/
`AllBlocksCleared` batches over ZMQ and has grown a `replay_endpoint`
(PUB/ROUTER replay buffer), connector-emitted events tagged with medium
(GPU/CPU/STORAGE) and locality (LOCAL/REMOTE), and
`kv_cache_report_mode="full"` which additionally emits `BlockStored` for
*reused* blocks so gateways learn about hits. SGLang's `--kv-events-config`
feeds Dynamo/sgl-router cache-aware routing with per-page chained hashes
and a load-publish endpoint. The kvc ABI is polled counters only; push
channels and callbacks are deliberate exclusions (plan §25).

Add a polling-buffered event extension: bounded ring of
stored/removed/cleared records with namespaced filtering, drained on
`poll`; events tagged with tier and hit-vs-store. Feeds P3 cache-aware
routing.

Tests: ring overflow drops oldest and reports loss; namespace filter;
clear emits a single cleared record.

### P2-10. Entry expiry and eviction-priority hints
**Status: open**

SGLang now ships a rich policy menu — lru/lfu/fifo/mru/filo/priority,
T-LRU (TTFT-budget-aware: nodes past the budget are reported infinitely
old and drained first), SLRU (probationary/protected segments), with JSON
config and per-node priority — and vLLM's CPU offload tier is lru|arc
(pluggable out-of-tree). The ABI carries no lifetime or priority signals.

Add optional per-transaction hints (expiry deadline, eviction priority) to
the write path; honor them in both reference providers' victim selection
(TTL sweep on allocation, priority-ordered victims). Engine-side surfacing
can wait for P1-6.

Tests: expired entries are evicted before LRU victims; higher priority
survives; deadline-precision boundaries.

### P2-11. Cross-block-size reuse (re-blocking)
**Status: open** — blocked on P1-7 for the engine side.

vLLM re-views one hash chain at any coarser block size
(`BlockHashListWithBlockSize` rescaling); SGLang's KV-Shard grafts
rotation-base-chained trees across ranks and sizes (`tp_lcm_size` for
heterogeneous TP). Our publication units are group-fixed
`tokens_per_block`, so a 16-token provider cannot serve a 32-token
consumer. With a token-level chain (P1-7) a block digest at size 2N is a
boundary value of the same chain: deliver a re-blocking view in lookup
(serve a 32-token block from two adjacent 16-token records whose
dependencies chain) plus negotiated per-group re-blocking.

Tests: 16→32 and 32→16 round views; dependency-chain mismatch is a miss,
never a wrong hit.

### P2-12. Zero-copy provider-to-provider transfer
**Status: open**

vLLM's `NixlPushConnector` writes producer KV directly into
consumer-preallocated memory (with the KV lease design); SGLang's external
linker moves device-pool ↔ L3 with no host tier and dedups replicated MLA
KV on rank 0 (TTFT 3.85→2.23s at 256K prefix). The ABI's `import_memory`
+ optional `map_read` (declared, implemented nowhere) are the hooks.
Deliver region grants or a peer-transfer extension so a mapping from
session A feeds a store into session B without an engine-mediated staging
copy. Gateway: PD disaggregation (P3).

Tests: grant lifecycle vs pin/refcounts; revoked grant fails closed; CUDA
provider interop.

### P2-13. Session-affinity soft retention
**Status: open**

SGLang's session radix cache now does per-component session ref tracking
(eviction prefers unreferenced → fewer-referenced → policy order; soft
protection, not pins; `/close_session` dereferences) and streaming
sessions go further (the session owns its KV row and tree lock across
turns; requests borrow). Add a soft-retention preference to the ABI
(evict last, not never) distinct from pins, plus session-scoped
enumeration via P2-8.

Tests: session-tagged entries outlive untagged under pressure; close
releases; interaction with P2-10 priorities.

### P2-15. Persistence across restart
**Status: open** — practical value gated on P1-6.

The identity helper (P0-2) was the stated blocker and is done. Deliver a
file-backed store in the host provider: deterministic layout derived from
block keys, enumerate-on-open (SGLang's file backend scans existing pages
into a metadata cache), budget-bounded directory, and namespace/model
binding already carried by the keys. vLLM's `fs` offload tier shares
cross-process via deterministic hash seeds — same principle.

Tests: restart restores hits (no re-publication); corrupted/stale files
are dropped, never served; cross-restart salt isolation.

### P2-16. Shared/remote storage tier (L3)
**Status: open** — gated on P1-6 + P2-15 patterns.

The ABI reserves `REMOTE_HOST/REMOTE_DEVICE/FABRIC` memory types;
nothing implements them. Both engines converged on shared L3 backends —
Mooncake transfer-engine stores, NIXL (io_uring/GDS/3FS/S3), HF3FS, file,
FlexKV — for cross-instance prefix sharing. Deliver one reference remote
backend for the host provider (file-over-NFS first, then Mooncake or
NIXL), including the consistency rule for concurrent writers (see
P2-17).

Tests: two sessions share via the backend; network failure = miss (never
a wrong hit); retirement propagates.

### P2-17. Cross-process in-flight write dedup (finish P2-7)
**Status: open (remainder)**

P2-7 closed single-session dedup. Across processes both engines still
recompute; our `begin_write` returns `Conflict` for a second writer. With
P2-16 in place, add attach-across-process: a writer that finds the key
committed-or-reserved by another process attaches/waits instead of
conflicting, with fencing (plan §26 transaction-recovery open question).

Tests: two processes race the same manifest; one commits, the other
attaches; crashed writer's reservation eventually reclaims (lease).

### P2-18. Sliding-window and hybrid window management *(model-triggered)*
**Status: open — trigger: an SWA model port* (`models/gpt-oss-20b` weights
are already staged; `llama_config.cc:55` returns Unimplemented for
sliding-window configs).**

The pool supports per-layer layouts but has no window semantics: KV blocks
are pure causal append, so an SWA model would hold full-length memory.
vLLM: `SlidingWindowManager` scans right-to-left for `cdiv(window, block)`
contiguous blocks, frees out-of-window tokens every step
(`remove_skipped_blocks`), retains `extra_retained_tokens` for MTP
re-prefill; newer variants include R-SWA (O(prefix+window) with gap
eviction) and DeepSeek-V4.1 bounded replay (SWA layers simply
`prefix_cacheable=False` and rebuild the window after a hit). SGLang: SWA
as a tree component with tombstones (structure survives, in-window data
rebuilt from fresh full KV), window-segment locks released early
(`dec_swa_lock_only`), out-of-window slot freeing with a retain floor at
the last checkpoint, and branching-point caching (shared-prefix token hit
43.8%→60.8% on DSV4-Flash).

Deliver when the model lands: window field per layer group in `KvLayout`/
`StateRequirements`, per-step out-of-window frees, adoption capped at the
window (choose bounded-replay semantics: no SWA-layer prefix adoption, full
attention layers adopt normally — vLLM's hybrid coordinator reconciles
per-type hit lengths to a fixed point; we need the same reconciliation
across groups), preemption interaction (window makes recompute cheap).

Tests: memory stays O(window) for SWA layers under long decode; hybrid
adoption reconciles full+SWA groups; window frees interact with chunked
prefill boundaries.

### P2-19. Recurrent-state checkpoint reuse *(model-triggered)*
**Status: open — trigger: a GDN/SSM-hybrid port* (`RecurrentStatePool`
exists with slot-zeroing only; no sharing, no save/restore).**

vLLM treats a mamba "block" as a full state snapshot at a block boundary
(align mode: interior positions are null blocks, ~2+spec+checkpoint live
states, prefill checkpoints at hash-aligned boundaries, boundary states
offered to connectors, `enable_mamba_shared_prefix_checkpoint` for shared
junctions). SGLang's MambaComponent stores int8-compressed checkpoints
(~2× capacity) with donate/fork semantics and demotes states to host on
write-back instead of dropping. InferX's hybrid models currently get zero
prefix benefit from recurrent layers.

Deliver: state snapshot at block boundaries keyed by the prefix chain,
adoption restores state into a slot, recompute-from-checkpoint on
preemption, and per-group hit reconciliation with the paged groups (a hit
must be a hit in *every* group, vLLM fixed-point style).

Tests: second identical prompt restores state (no re-scan); checkpoint at
chunk boundaries under aligned chunked prefill; preemption resumes from
the last checkpoint.

## P3 — deliberate deferrals (tracked, not planned)

- **PD disaggregation** (plan #58/59/60): vLLM NIXL pull *and* push
  connectors (`kv_role`, bootstrap, `kv_load_failure_policy`), Mooncake,
  decode-side KV offload manager; SGLang mooncake/nixl/mori/ascend engines
  with page-granular sends and retraction backup. Gated on P1-6 + P2-12/16.
- **Speculative decoding × cache** (plan #45): vLLM eagle-drop alignment
  (`eagle_proof_margin` at hash-unit granularity), mamba+EAGLE checkpoint
  alignment, n-gram/GPU n-gram; SGLang bigram radix keys (draft/target
  share boundary tokens, O(1) view flip), fused draft pools (one slot id
  covers target+draft). Our roadmap: `docs/speculative_decode.md` phase 3
  requires transactional target KV with rollback and CoW before
  speculative writes.
- **Quantized KV menu**: vLLM `kv_cache_dtype` spans fp8 e4m3/e5m2,
  int8/fp8/int4 per-token-head, nvfp4, turboquant, ultraquant, per-layer
  skip patterns; SGLang adds mxfp8/nvfp4 with a prefill dequant dtype.
  Our pool accepts 1-byte dtypes but nothing sets them and kernels lack
  dequant paths — revisit with the quantized-weights workstream.
- **Cross-layer KV sharing** (YOCO / Gemma3n `kv_sharing`, plan #47) and
  **encoder/cross-attention + multimodal caches** (vLLM encoder cache +
  `/reset_mm_cache` + EC-transfer framework; SGLang `multimodal_cache`,
  embedding store): no such models wired.
- **Cache-aware routing** (vLLM events→gateway replay; SGLang sgl-router
  power-of-two, Dynamo, Aug-2026 recoverable-KV-placement RFC): gated on
  P2-9.
- **Distributed KV**: SGLang KV-Shard logical-page placement, DCP/PCP
  context-parallel hashing (vLLM), TP-replica publication dedup (plan
  §28.12), elastic resharding (plan #33). Our TP is single-scheduler
  mirrored; no divergence to solve yet.
- **CUDA-graph decode**: `CaptureGraphs` returns Unimplemented while the
  block-table design is already graph-stable in intent (fixed `kv_indices`
  buffer). Perf parity, not cache semantics — both engines lean on graphs
  for decode.
- **Transfer checksums** (SGLang PR #39500): provider-internal (plan #61).
- **Cross-format FP8↔BF16 load conversion**: intentional exclusion
  (plan #38).
- **Host-resident sparse KV / DSA pools / late-interaction pools**
  (vLLM HiSparse, QUEST; SGLang DSA/NSA pools): far-future model classes.
- **Stale artifacts**: `vllm_parity_probe_test` fixtures recorded
  2026-10-09, before the model-layer rebuild in `f741b8f` — re-record or
  relax to agreement-where-decided margins; `docs/extensions/kvcache.md`
  `run-bench`/`kv_bridge.h` claims — fixed by P1-6 (or immediately).

## Edge cases to pin as tests where features already exist

Both engines treat these as invariants; InferX implements most but does
not pin all of them:

- n−1 last-token recompute, salt isolation, chunk-boundary publication,
  preemption re-adoption without double-counted hits, duplicate-hash
  coexistence, impossible-request admission error — all tested today;
  keep them green.
- **Zero-length and sub-block prompts** (SGLang returns a root-anchored
  empty match): adoption path untested for `prompt_len < block_size` and
  `prompt_len == 0`.
- **Two requests adopting the same blocks in the same step** (concurrent
  identical admission): refcounts should cover it; untested.
- **`Clear()` while adoptions are pinned** (see P2-8): untested.
- **`kv_cache_usage` semantics**: counts allocated blocks including pinned
  cached ones — document or split into allocated/evictable/cached gauges
  (vLLM `hits_by_source`, SGLang `kv_evictable_tokens` are the models).
