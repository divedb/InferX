# KV-cache parity TODO — InferX vs vLLM / SGLang

Gap analysis derived from a line-level review of vLLM (main @ `f4b8d43`,
2026-10-13) and SGLang (main @ `2428d0f5`, 2026-10-14), against InferX's two
cache layers:

- **Engine layer** — `KvBlockPool` (one allocation, LIFO free list, per-layer
  layouts), `CacheManager` (per-request reservations), `Scheduler` (FCFS +
  chunked prefill). No prefix cache, no preemption, no hit accounting.
- **Provider layer** — the `kvc` SDK 0.1 (C ABI + C++ facade), host and CUDA
  tiered reference providers, `KvCacheBridge` adapter. See
  `docs/kvcache_plan.md` §27–29 for the full 68-item use-case catalog; this
  file tracks the actionable subset in implementation order.

Each item below is one deliverable: implementation + tests + a dedicated
commit. Status is updated as work lands.

## Priorities and dependencies

```text
P0-1 engine prefix cache ──┬─> P0-3 preemption (recompute)
                           ├─> P1-5 block-aligned chunked prefill
                           └─> P1-6 hit accounting
P0-2 kvc identity helper ──┬─> P2-kvc sharing/persistence items
P1-* independent
```

---

## P0 — engine core

### P0-1. Automatic prefix caching in the serving loop
**Status: done** — vLLM-style block-hash prefix cache landed: `BlockHashChain`
(SHA-256, salt-in-seed, deterministic across processes), `PrefixCache`
(refcounted hash index with LRU eviction and duplicate coexistence),
`CacheManager::AdoptPrefix`/`PublishComputed` (n−1 cap, whole-block
publication of prompt and decode blocks), scheduler adoption at admission
and publication after each completed step, `Request::cache_salt`.

The engines' central cache feature; vLLM V1 keys every block by a hash chain
(`hash_block_tokens`: `h = H(prev, tuple(token_ids), extra_keys)`, extra keys =
LoRA / multimodal / `cache_salt` on block 0 / prompt-embeds; only full blocks
hashed; SHA-256 default) and SGLang keys radix-tree nodes by page-aligned
token runs with per-page chained SHA-256. InferX's scheduler frees every
block on request finish and recomputes identical prompts from scratch.

Implement vLLM-style block-hash prefix caching over `KvBlockPool`:

- `PrefixBlockIndex`: hash-chain per block (salt folded into block 0's hash),
  `hash → {block ids}` map, per-block ref counts, LRU order maintained on
  touch/hit (vLLM: hit blocks are removed from the free queue; freed cached
  blocks rejoin the tail = FIFO/LRU, non-cached rejoin the head = LIFO).
- `CacheManager` learns hit adoption: on admission, walk the request's block
  hashes, count the longest prefix hit, cap at `num_tokens − 1` (the last
  prompt token must recompute for logits — both engines do this), pin hit
  blocks (ref count), prefill the block table, and set
  `num_computed_tokens` so the model runner skips the cached prefix.
- Finished/aborted requests release blocks into the eviction pool instead of
  the free list when their hash entries are live; allocation first drains the
  true free list, then evicts LRU cached blocks (lazy eviction on
  allocation, like vLLM's `_maybe_evict_cached_block`).
- Gated by `CacheConfig::enable_prefix_caching` (flag already exists; default
  true, matching both engines' defaults).
- Edge cases from the engines: identical finished prefixes collapse onto one
  set of blocks; hash insertion happens only for blocks whose tokens are
  fully computed (partial tail blocks are never hashed); double-counting hit
  tokens must not happen when the same request is re-admitted after
  preemption; deterministic hashing (fixed seed) so restarts and processes
  agree — vLLM seeds `NONE_HASH` per-process for Python `hash()`, but its
  SHA-256 modes are deterministic; we always use SHA-256.

Tests: hash-chain known-answer vectors; hit adoption pins and skips tokens;
second identical prompt reuses blocks (pool `FreeBlocks` unchanged while
cached); LRU eviction under pressure; salt isolation; `n−1` cap; eviction
respects pinned blocks.

### P0-2. Canonical identity helper for the kvc SDK
**Status: open**

`docs/kvcache_plan.md` §8 specifies portable identities: SHA-256 over
domain-separated, canonically encoded records; a token-level prefix chain
(`P(i+1) = H("kvc.token", Pi, token_id, position, aux_digest)`); block keys
derived from namespace + model identity + group + extent + dependency digest.
Today every caller (including our own conformance tests) invents ad-hoc keys,
which the SDK docs explicitly forbid for production. This blocks persistence
across restart (catalog 20), cross-session sharing (17), and safe
multi-process dedup (28/29) — the plan calls it out as one of two blockers
for all sharing use cases.

Deliver `kvc/identity.h` + `kvc_runtime` sources: a small SHA-256, canonical
record encoding (domain tag + fixed-width fields, no floating point), prefix
chain over a token sequence, and block-key derivation. Deterministic across
processes and builds; no Abseil/CUDA dependency.

Tests: NIST SHA-256 vectors; identical inputs → identical digests across
constructed-differently sequences; any token/position/namespace/aux change
changes the chain from that point on; empty-sequence seed.

## P1 — engine behaviors the engines consider table stakes

### P1-3. Preemption: recompute path
**Status: open**

vLLM V1 is recompute-only (V0 swap is gone): when a running request cannot
get blocks, the scheduler preempts the *newest* running request (`running[-1]`
under FCFS), frees all of its blocks (cached hashes stay), resets
`num_computed_tokens` to 0, and prepends it to the waiting queue; the freed
cached blocks are usually re-hit on resume. SGLang instead retracts the
request with *fewest generated tokens first* and evicts radix entries to make
room. InferX's scheduler currently *skips* an under-resourced request forever
(`continue`), which can starve it behind larger requests and never makes room.

Implement vLLM-style recompute preemption in `Scheduler::Schedule`: on
`Reserve` failure for a running request, preempt running requests from the
back until the reservation fits or only the victim itself remains; reset
watermark, release blocks (hash entries survive), requeue at the waiting-queue
head, surface `RequestStatus::kPreempted` (the enum value exists unused), and
count preemptions in stats. Beam/parallel groups arrive later with spec
decoding; nothing to do for them yet.

Tests: victim selection is newest-first; preempted request resumes and
completes; cached blocks survive preemption (re-hit on resume); the protected
minimum (never preempt the only request into unrunnable state) holds;
`preempted_req_ids` surface in stats.

### P1-4. Prefix-cache hit accounting
**Status: open**

vLLM tracks `PrefixCacheStats` (requests/queries/hits, token-granular,
recorded at admission; separate counters for post-preemption re-hits) and
SGLang exposes hit-rate gauges feeding Prometheus. InferX's `SchedulerStats`
has no cache-hit fields at all.

Add hit/miss token counters (and queries/requests) to `SchedulerStats`,
recorded at admission; expose them wherever `SchedulerStats::kv_cache_usage`
already flows.

Tests: counters distinguish fresh admission from post-preemption re-admission
hits; token-granular (a 5-block hit on a 20-block prompt counts 80 tokens).

### P1-5. Block-boundary-aligned chunked prefill under caching
**Status: open**

Both engines stop prefill chunks at block/page boundaries when caching is on
(vLLM `_mamba_block_aligned_split` clips chunks to the next block boundary
and the fine-grained hash boundary; SGLang `_get_prefill_admission` computes
`extend_len = chunk_tokens_limit // page_size * page_size` and aligns the
end). Unaligned chunk ends leave a partially-filled block whose hash cannot
be registered, so a chunked long prompt publishes nothing until its final
chunk. InferX cuts chunks purely by token budget.

Clip the scheduler's chunk to the next block boundary when
`enable_prefix_caching` is on (leave unaligned when off — vLLM also skips
this for non-caching runs to avoid wasted padding).

Tests: chunks end on multiples of `block_size`; a two-chunk prompt publishes
the first chunk's blocks after chunk 1 completes; alignment off → old
behavior.

### P1-6. Engine→provider publication loop (`KvCacheBridge` in serving)
**Status: open**

The plan (§27 item 25/26, §28 tail) names the second blocker: the engine
never calls providers from its serving loop — `KvCacheBridge` exists and is
unit-tested, but `EngineGateway` does not instantiate it, so no external
cache (host provider, CUDA tiered provider) participates in serving. vLLM
does this through the KVConnector lifecycle (`get_num_new_matched_tokens` →
`start_load_kv` → `wait_for_layer_load` → `save_kv_layer` → `wait_for_save` →
`request_finished`), SGLang through HiCache hooks in the scheduler event
loop.

Wire an optional provider session into `EngineGateway`/`ModelRunner` using
`KvCacheBridge`: lookup before admission, async load into allocated pages,
publish completed blocks, release pins at finish. Blocked on nothing
technical; substantial surface (async load gating, failure policy
`recompute|fail`, preemption interaction `handle_preemptions`).

## P2 — kvc ABI extensions (plan §28 gaps)

Ordered by the plan's gap analysis; each is provider-side work first in the
host reference provider, plus conformance tests in `tests/kvc`.

### P2-7. In-flight write dedup (catalog 28/29; §28.6)
**Status: open**

vLLM/SGLang dedup concurrent identical publications in-process (append-only
hash maps / radix insert splits). Across processes both engines currently
recompute; our `begin_write` returns `Conflict` for the second writer, which
must recompute or poll. Add an attach-to-committed fast path: a lookup that
finds the key committed while a writer was in flight treats the write as
done (dedup), and let `begin_write` wait/poll instead of failing when the
same manifest is already reserved (attach). Blocked on P0-2 for safe keys.

### P2-8. Management surface: enumeration, bulk clear, namespace reset (catalog 37; §28.5/28.9)
**Status: open**

SGLang exposes hicache storage attach/detach/list endpoints and
`/flush_cache`; vLLM exposes `reset_prefix_cache` (needed after weight
updates, RLHF). The ABI removes keys one digest at a time only. Add a
`kvc.management` extension table: enumerate stored keys (paged), bulk
remove by namespace, hit-rate/lifetime counters reset.

### P2-9. Publication event feed (catalog 54; §28.11)
**Status: open**

vLLM `KVEventsConfig` publishes `BlockStored`/`BlockRemoved`/
`AllBlocksCleared` batches over ZMQ with replay; SGLang `kv-events` feeds
`sgl-kv-indexer` for `cache_aware` routing. The ABI has polled counters
only. Add a polling-buffered event extension (namespaced filtering, bounded
ring) — callbacks stay excluded by plan §25.

### P2-10. Entry expiry and eviction-priority hints (catalog 34/35; §28.5)
**Status: open**

Neither engine exposes consumer TTL over a shared cache; SGLang ships
pluggable eviction (lru/lfu/fifo/mru/filo/priority/T-LRU/SLRU — per-request
`priority` propagates into node priority) while vLLM is LRU-only. The ABI
carries no lifetime or priority signals. Add optional per-transaction hints
(expiry deadline, eviction priority) to the write path and honor them in the
reference providers' victim selection.

### P2-11. Cross-block-size reuse (catalog 30; §28.2)
**Status: open**

vLLM re-views one hash chain at any coarser block size
(`BlockHashListWithBlockSize` rescaling) because its hashes are
block-size-independent; SGLang page-truncates keys. Our publication units
are group-fixed `tokens_per_block`, so a 16-token provider cannot serve a
32-token consumer. Requires a re-blocking operation (serve a 32-token block
from two adjacent 16-token records whose dependencies chain) or negotiated
per-group re-blocking in lookup.

### P2-12. Zero-copy provider-to-provider transfer (catalog 40; §28.1)
**Status: open**

Region grants or peer-transfer extension so a mapping from session A can
feed a store into session B without an engine-mediated staging copy.

### P2-13. Session-affinity soft retention (catalog 55)
**Status: open**

SGLang's session radix cache keeps evictable-per-session references
(`--enable-session-radix-cache`), released on session close. Add a soft
retention preference (evict last, not never) distinct from pins.

## P3 — deliberate deferrals (tracked, not planned here)

- Hybrid/SWA/Mamba cache groups in the engine (catalog 9/31/46): our model
  layer has no sliding-window or recurrent model wired yet; the pool already
  supports per-layer layouts. Revisit when such a model lands.
- Speculative decoding interactions (45), COW for parallel sampling n>1
  (41 — vLLM V1 itself dropped the dedicated n>1 COW path in favor of
  refcount sharing, which our design gets from P0-1), streaming sessions
  (63), PD disaggregation (59/58/60), checksums (61 — provider-internal).
- vllm_parity_probe_test fixtures are stale (recorded 2026-10-09, before the
  model-layer rebuild in f741b8f); re-record the vLLM+InferX serve results or
  relax to agreement-where-decided margins matching the rebuilt kernels.
