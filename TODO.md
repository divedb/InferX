# KV cache review and implementation backlog

Reviewed 2026-10-15. Scope: `include/inferx/cache/`, `src/cache/`, their unit tests, and the scheduler/model-runner/executor integration needed to establish actual supported scenarios. No history files or Git history were read. Findings below come from source inspection, not from running upstream servers or reproducing failures on a GPU. “Bug” identifies a concrete inconsistent code path; acceptance checks describe work still to do.

Upstream comparisons are pinned to the source snapshots retrieved for this review:

- **vLLM V1:** [`5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1`](https://github.com/vllm-project/vllm/tree/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1).
- **SGLang:** [`0cf683ea8d27041da981f7875e1755b9f7772799`](https://github.com/sgl-project/sglang/tree/0cf683ea8d27041da981f7875e1755b9f7772799).

This is a capability comparison, not a proposal to reproduce every upstream class or storage backend. Backend/model restrictions matter: a feature existing upstream does not imply every combination of quantization, hybrid state, speculative decoding, graph execution, and transfer works there. Keep unsupported combinations explicit in InferX until validated.

## What already exists

| Area | InferX today | Upstream comparison and remaining gap |
| --- | --- | --- |
| Paged storage | Fixed rank-local pool, integer block IDs, per-layer K/V tensor views, O(1) free list and double-free detection | Basic paged allocation is present; see P0 for lifecycle defects and P1 for richer layouts. |
| Prefix reuse | SHA-256 chained token hashes, versioned encoding, salt isolation, full-block matching, shared references, duplicate hashes, lazy LRU eviction | Similar to vLLM's hash-index design [V1, V2]. SGLang uses a radix tree with page-aligned splits and protected paths [S1]. A radix-tree rewrite is not itself a requirement. |
| Request growth | Append-only, per-request reservation; normal `Release` retains cacheable blocks | No truncate, fork, tentative reservation/commit, or transfer ownership [V1, S2]. |
| Serving integration | Prefix adoption with an `N-1` logits cap, chunked prefill, completed-step publication, recompute-preemption machinery | These exist; correctness gaps below prevent treating all edge cases as supported. |
| Model layouts | Uniform or heterogeneous `KvLayout`; one-entry latent storage possible; separate recurrent-state buffer | Different shapes are supported at storage level, but cache groups, recurrent checkpoint reuse, and complete MLA execution are absent [V3–V5, S3–S5]. |
| Precision | Byte-addressable dtypes accepted by pool; sub-byte types rejected | Built-in model creation executes BF16. FP8 byte storage is not an implemented quantized-KV pipeline [V6, S6]. |
| Capacity | Block count, explicit bytes, or memory-utilization profiling; TP minimum budget reduction | Already present in `ModelRunner::Initialize`; configuration, peak allocation, and admission gaps remain. |
| Parallel execution | Rank-local KV buffers; executor dispatches one logical block mapping to TP runners | This is existing TP support, not cross-worker cache transfer, context parallelism, or distributed cache residency. |
| Observability | Occupancy, cached-block count, scheduler prefix query/hit/token counters | No residency breakdown, eviction events, transfer metrics, or independent cache diagnostics. |

Local implementation anchors: [manager](src/cache/cache_manager.cc), [prefix index](src/cache/prefix_cache.cc), [KV pool](src/cache/kv_block_pool.cc), [recurrent pool](src/cache/recurrent_state_pool.cc), [scheduler](src/engine/scheduler.cc), [runner](src/models/model_runner.cc), [executor](src/engine/model_executor.cc), [configuration](include/inferx/config/cache_config.h), [model registry](src/models/registry.cc).

## Priority and dependency order

- **P0:** Incorrect KV reuse, memory ownership, or serving progress. Fix before expanding cache behavior.
- **P1:** Foundations and high-value model/workload coverage. Some require model/kernel work outside `cache`.
- **P2:** Advanced serving, offload, distributed reuse, and measured performance improvements.
- **P3:** Specialized optimizations and broader backend coverage.

Each item lists prerequisites by ID; “none” means independently actionable. Within a priority, work in dependency order. Suggested sequence: **C01–C08 → F01–F04 → F05–F09**, with diagnostics **F10** alongside foundation work; then select advanced work for the intended deployment. **A01 → A02 → A03** is the speculative/branching path; **F02 + F04 + F06 → A04 → A05/A06** is the transfer path. **F05 → F07/F08** is the hybrid-model path. P0 items do not depend on adopting an upstream architecture.

## P0 — correctness and progress

### C01 — Preserve shared ownership across prefix-cache clear

- [x] **Bug:** `PrefixCache::Clear` removes `by_block_`, including live reference counts. If requests A and B share a block, `CacheManager::ClearPrefixCache(); Release(A)` then frees that block because `Contains` is false, although B still references it. B can read overwritten KV; its later release can double-free. The existing clear test checks one adoption and immediate pool counts, not two manager-owned sharers followed by reuse.
- **Change:** Separate allocation ownership from hash discoverability, or reject clear while requests are active. Define whether already-running requests may republish after a clear; use an epoch for invalidation if needed. vLLM rejects a prefix reset with live blocks [V2].
- **Accept:** Two sharers plus private tails; clear, release one, allocate another request, and verify the remaining sharer stays intact. Exercise clear twice, continued publication, cancellation, and clear after a weight change.
- **Dependencies:** none.
- **Decision D1 (recorded 2026-10-15, with the C01 fix):** ownership and discoverability are separated inside `PrefixCache`: `Clear` frees and destroys unreferenced entries, turns still-pinned entries into ownership-only tombstones (kept in `by_block_`, removed from `by_hash_`), and `Release` frees a tombstone's block when its last holder lets go. `ClearPrefixCache` is permitted with live requests rather than vLLM's reject-with-live-blocks stance, because the tombstone ledger preserves correctness. A manager-level cache epoch invalidates publication rights: requests admitted before a clear keep their blocks but never publish afterwards, so pre-clear KV (e.g. computed under old weights) cannot re-enter the index. `CacheManager::Release` and the destructor snapshot untracked blocks *before* dropping cache references, because releasing a tombstone's last reference frees its block through the cache. Covered by `ClearPreservesSharedBlockOwnership`, `InFlightRequestDoesNotRepublishAfterClear`, `ClearIsRepeatableAndFreshRequestsAdoptAfterwards`, and `ClearKeepsOwnershipOfPinnedBlocksUntilLastReference` (debug + release).

### C02 — Make manager teardown release each physical block exactly once

- [ ] **Bug:** `CacheManager::~CacheManager` calls `pool_->FreeBlocks` on every request table even after decrementing prefix references. Shared blocks are freed repeatedly. Cache-only blocks left after all requests were released are never returned to a surviving borrowed pool: `PrefixCache` has no draining destructor.
- **Change:** Use the same ownership-aware release path as normal operation, then drain unreferenced cache entries; handle errors without relying solely on debug assertions. Preserve the documented requirement that device work has finished.
- **Accept:** Destroy a manager with two active sharers; with only cached entries; with mixed cached/private blocks; and after clear. The borrowed pool must end entirely free with no duplicate IDs. Existing destructor coverage uses caching disabled.
- **Dependencies:** C01.

### C03 — Publish only fully computed, accepted KV

- [ ] **Bug:** `PublishComputed` bounds publication by token-span length and allocated block count, not a completed-token watermark. The scheduler passes the entire prompt after every chunk. For block size 4, prompt length 8, and a first chunk of 1 token, one allocated block is immediately published as if all four tokens were computed. `clip_chunk` deliberately allows an unaligned chunk when the remaining budget is smaller than a block. Reserving capacity ahead of execution has the same hazard.
- **Change:** Pass an explicit successful-computation/accepted-token boundary; publish only blocks below it. Keep scheduled, allocated, computed, and accepted lengths distinct. Reject inconsistent spans rather than silently capping them. Upstream APIs explicitly carry computed lengths and exclude uncommitted speculative tokens [V1, S1, S2].
- **Accept:** Budgets 1 through `block_size-1`; nonaligned resumed chunks; over-reservation; failures before/within execution; prompt/output boundary blocks; and a second request arriving after the short chunk. No uncomputed token may become a cache hit.
- **Dependencies:** none.

### C04 — Correct recurrent-slot reset addressing

- [ ] **Bug:** `State` and `ConvState` expose `[all slots' matrix states][all slots' convolution states]` per layer. `ResetSlot` instead zeros `slot * (matrix_bytes + conv_bytes)` as one contiguous region. With multiple slots and nonzero convolution state, it clears parts of another slot and fails to clear the intended convolution state.
- **Change:** Zero the two slot slices at their actual offsets, or consistently change storage and views. Keep operations ordered on the execution stream.
- **Accept:** Fill all matrix/conv slots with distinct sentinels, reset first/middle/last slots, and verify exactly the selected slot changed across heterogeneous layers. Cover `conv_kernel_size == 1`, one slot, invalid slots, and CPU/CUDA memory implementations.
- **Dependencies:** none. This is a latent storage bug; the current built-in runner does not wire recurrent state into execution.

### C05 — Repair preemption state transitions and replay the accepted sequence

- [ ] **Bugs:** Requests start `kWaiting`, but admission never changes them to `kRunning`; `Request::Preempt` consequently rejects normal scheduler victims. Independently, preemption retains `output_`, while readmission hashes only the original prompt and `NewRequestData` reconstructs the runner from prompt tokens alone. Previously accepted output and sampling progress are not restored. Subsequent publication can associate accepted-output hashes with KV recomputed for a different continuation.
- **Evidence (2026-10-15, found while verifying C01):** `inferx_scheduler_preempt_test` crashed on unmodified HEAD in debug builds — `assert(transition.ok())` at `src/engine/scheduler.cc:236` (`PreemptNewestAfter`) — and release builds masked it because the assert was compiled out.
- **Status (2026-10-15, same day):** the transition bug is fixed — admission performs `kWaiting → kRunning` via `Request::MarkRunning`, `Preempt`/`MarkWaiting` transitions are validated, and readmission replays prompt **plus retained accepted output** (the watermark advances toward `NumTokens()`, and the readmission plan hands the runner the full replay sequence). Covered by `PressurePreemptsNewestAndVictimResumes`, `PreemptedRequestReadoptsItsOwnCachedPrefix`, and `PreemptedVictimReplaysAcceptedOutput` (debug + release). Remaining for full C05 closure: seeded-sampling/RNG restore against an uninterrupted reference and the all-prefix-evicted / partially-retained / salted / repeated-preemption acceptance cases.
- **Change:** Introduce a valid admission transition and replay prompt plus accepted output, excluding any token without KV from the cached watermark. Restore sampling/RNG and stopping state without emitting old tokens twice. Reuse available generated-prefix KV when valid. vLLM's allocation path explicitly accounts for resumed requests replaying output [V1].
- **Accept:** Preempt after several sampled tokens, including mid-block; compare resumed greedy output and seeded sampling to an uninterrupted reference. Test all-prefix-evicted, partially retained, salted, repeated-preemption, and aborted-victim cases.
- **Dependencies:** C01, C03.

### C06 — Initialize prefix state for short prompts and enforce append-only publication

- [ ] **Gap/latent bug:** `AdoptPrefix` returns before creating hash state when the prompt has fewer than one block. Such requests never publish later full blocks formed during decode. `PublishComputed` can also lower `hashed_blocks` on a shorter input without rewinding `chain`, corrupting future hashes if the API is used for rollback. Existing hashes are not checked against changes to previously supplied tokens.
- **Change:** Initialize the salted chain even on a zero-hit/short prompt; explicitly validate monotonic committed length and immutable committed tokens, or provide a separate rollback API. Define empty prompts, repeated adoption, `Reserve` before adoption, request-ID reuse, and negative adoption caps. The `N-1` limit is currently caller-enforced, not intrinsic to `AdoptPrefix`.
- **Accept:** Prompt sizes 0, 1, `B-1`, `B`, `B+1`; decode across the first full block; salted short prompts; shrinking/mutated publication spans; zero reservation; and reuse of a released ID. Preserve the valid uncapped low-level API if intentionally supported.
- **Dependencies:** C03.

### C07 — Validate geometry and arithmetic before allocating or indexing

- [ ] **Bugs/gaps:** Pool creation multiplies signed dimensions without overflow checks and narrows block IDs to `int32_t` without a block-count bound. `BlocksForTokens` can overflow its rounding addition. `BlockTable::Locate(-1)` can return block 0 with a negative slot because signed division truncates toward zero. `BlockTable` permits a zero block size; `LayoutFor` has unchecked indexing. Recurrent sizing has similar unchecked products.
- **Change:** Checked byte/element arithmetic, valid dtype and geometry checks, bounded block/token indices, and explicit checked versus internal unchecked accessors. Guard scheduler/runner `int` and device-table conversions too. Define supported entry counts; values above two are currently allocated but have no general entry-view accessor.
- **Accept:** Negative/zero dimensions and positions; `INT32_MAX`/`INT64_MAX` boundaries without huge real allocations; invalid layers/dtypes; heterogeneous stride bounds; and recurrent-size overflow. Invalid input must fail before allocation or memory access.
- **Dependencies:** none.

### C08 — Guarantee progress under impossible or fragmented capacity demands

- [ ] **Gap:** A prompt can be admitted incrementally even if it can never fit the pool. Once the oldest/only running request cannot grow, the scheduler has no victim and simply retries on later steps. An impossible waiting head can also block later requests. Empty prompts and a zero effective token budget need defined admission outcomes.
- **Change:** Validate context/capacity limits, distinguish “retry after reclaim” from “cannot fit,” and add headroom/full-sequence admission policy where appropriate. Account for shared pins, recurrent slots, and future lookahead separately. vLLM checks reservation headroom and optionally full-sequence fit [V1].
- **Accept:** A prompt one token beyond pool capacity; decode crossing the last available page; all blocks pinned; zero-progress queue head with a smaller request behind it; zero/negative configuration; and constrained chunk budgets. Every accepted request must advance, yield through a defined policy, or fail explicitly.
- **Dependencies:** C03, C05, C07.

## P1 — lifecycle foundations and model coverage

### F01 — Make failed reservation behavior match its contract

- [ ] `Reserve` promises unchanged pool availability on failure, but `EvictFor` can discard cached entries before discovering that insufficient capacity remains. Preflight against free plus evictable blocks before destructive eviction, or explicitly revise the contract and expose the effect. Separate capacity estimation from adoption/pinning for scheduler planning. Define partial-failure behavior of `FreeBlocks`, which currently frees preceding IDs before returning an error on a later ID.
- **Accept:** Failed growth with both pinned and evictable cache entries preserves the chosen contract; failed admission unwinds all pins; invalid/duplicate IDs in bulk release cannot leave ambiguous ownership.
- **Dependencies:** C01, C07. **Reference:** vLLM allocation preflight [V1].

### F02 — Add completion-aware leases and robust request cleanup

- [ ] The single-threaded, synchronous contract is legitimate today, but cannot safely support overlapped execution, transfer, asynchronous cancellation, or concurrent cache callers. Introduce explicit in-flight read/write leases or equivalent stream-event fences; delay physical reuse until the last consumer completes. Specify scheduler-owner serialization instead of assuming a mutex alone solves GPU lifetime.
- **Accept:** Cancel, preempt, clear, fail, and shut down while writes/reads are outstanding; stale completion after request-ID reuse; two streams; and repeated cleanup. No early publication or recycled block access. Maintain the simple synchronous path.
- **Dependencies:** C01–C03, C05. **References:** deferred publication in [V1], transfer lifecycle in [V7], request handles/locks in [S2].

### F03 — Retain reusable ancestors when evicting a sequence

- [ ] Current `PrefixCache::Release` adds blocks to LRU in forward sequence order. Pressure therefore evicts the earliest block first, making retained suffix hashes unreachable because adoption stops at the first miss. Release tail-first, or adopt a policy that preserves useful prefix paths while respecting recency across requests. vLLM frees in reverse order; SGLang evicts leaves [V2, S1].
- **Accept:** Cache `A/B/C`, reclaim one block, then request `A/B/D`; retain the largest useful prefix. Cover overlapping requests, duplicate hashes, pinned ancestors, and release order ties. Measure reuse under pressure rather than only entry count.
- **Dependencies:** C01, C02.

### F04 — Make cache identity cover all KV-affecting inputs

- [ ] Salt and token IDs suffice for a single immutable text model, but not LoRA/adapters, multimodal placeholders, prompt embeddings, position offsets/RoPE configuration, model revisions, or externally persisted KV. Add typed, versioned identity metadata and a model/cache epoch. Keep storage compatibility metadata distinct from semantic identity. Do not hash sampling settings that cannot change already-computed KV.
- **Accept:** Identical token placeholders with different images/embeddings/adapters cannot share; identical effective inputs can. Cover a multimodal span crossing blocks, a model reload, changed positional semantics, and incompatible imported dtype/layout/sharding. Preserve deterministic serialization and salt isolation; add request-path salt propagation where serving APIs are introduced.
- **Dependencies:** C01, C06. **References:** vLLM extra hash keys [V8]; SGLang `extra_key` and `cache_salt` [S1].

### F05 — Introduce typed cache groups with independent allocation semantics

- [ ] Per-layer geometry is already supported, but all layers share one logical block sequence and grow for every token. Add group specifications for full attention, windowed/local attention, recurrent state, and non-prefix-cacheable scratch state; independent block mappings/capacities and common valid-hit boundaries. Include layer-to-group mapping and explicit layer KV aliasing where a model reuses another layer's K/V.
- **Accept:** Unequal layer counts/dtypes/head dimensions; different block sizes; per-group exhaustion with atomic rollback; differing group hits; shared physical buffers; and aliases without double allocation/free. Report padding and unusable capacity. Do not copy an old “exactly two attention types” restriction: current vLLM coordinator source has generalized group iteration even though its design document is older [V3, V4].
- **Dependencies:** C07, F01, F04. **References:** typed specs and managers [V3–V5], SGLang component separation [S3, S4].

### F06 — Add explicit physical layout and backend capability contracts

- [ ] The current layout is fixed to `[layer][entry][block][token][head][dim]`. Add layout descriptors, strides, alignment, padding, and supported block-size validation at the backend boundary. Provide auxiliary/packed planes when needed rather than overloading `entries_per_token`. One-entry `KeyCache` is useful latent storage but does not supply all MLA, positional-key, indexer, or quantization buffers.
- **Accept:** MHA/MQA/GQA, nonsharded MLA latent geometry, unequal K/V dimensions where supported, supported NHD/HND or packed layouts, and rejection of incompatible kernels. Every tensor view and transfer descriptor must agree on bytes and strides.
- **Dependencies:** C07; F05 for grouped layouts. **References:** [V5, V6, S5, S6].

### F07 — Support bounded sliding-window, local-chunk, and sink retention

- [ ] No cache operation currently retires old request blocks while a request continues. Implement per-group out-of-window reclamation, logical-position mapping, and window-aware prefix eligibility. Full attention must retain the full prefix; local attention can retain only the necessary window/chunk. Attention sinks and circular scratch buffers need distinct retention and cacheability rules.
- **Accept:** Window smaller than a page; window/chunk boundaries inside pages; prefill chunks larger than the window; pinned shared blocks; long decode with bounded local memory; and a hit whose early full-history pages were evicted. Never retire pages still needed by an in-flight or speculative step.
- **Dependencies:** C03, F02, F05, F06. **References:** `SlidingWindowManager`, `ChunkedLocalAttentionManager`, sink/circular specs [V4, V5]; SWA components and pure-SWA cache [S4].

### F08 — Complete recurrent/SSM state ownership and prefix checkpoints

- [ ] `RecurrentStatePool` only stores and resets matrix/conv tensors; it has no slot allocator, request mapping, checkpoint capture/restore, copy-on-write, or prefix-index integration. The runner rejects models with no paged state and does not construct a recurrent pool. Wire the lifecycle and backend state descriptors before enabling hybrid/pure-recurrent models.
- **Accept:** Restore exact matrix **and** convolution state at the same boundary as reused attention KV; branch into independent mutable slots; evict snapshots separately from active state; recover after preemption; handle unequal state dtypes/shapes and independent slot exhaustion. Test checkpoint alignment, missing checkpoint despite a longer attention hit, and pure-recurrent execution. Keep prefix caching disabled/rejected where complete state restoration is unavailable.
- **Dependencies:** C04, C05, F02, F05. **References:** vLLM `MambaManager` [V4]; SGLang Mamba pool and checkpoint copy-on-write [S3, S5]. vLLM's inspected Mamba path requires align mode for APC and explicitly rejects DCP/PCP; do not promise those combinations by analogy.

### F09 — Implement quantized KV as a complete storage/read/write path

- [ ] Add cache dtype selection independent of model dtype, K/V scales and their granularity/lifetime, calibration or supported dynamic scaling, append quantization, attention dequantization, and mixed-precision layer overrides. Then support packed sub-byte KV with explicit scale/packing metadata if target kernels justify it. The existing FP8-capable byte allocation is only a foundation.
- **Accept:** FP8 format/device compatibility; zero/large/NaN inputs and scale validation; cached-versus-uncached numerical tolerance; scales preserved through sharing/copy/offload; mixed layer formats; packed tails and padding. Reject unsupported combinations at startup. Quantized recurrent checkpoints are a separate decision with their own quality tests.
- **Dependencies:** F04, F06; F05 for mixed groups. **References:** [V6, S5, S6].

### F10 — Expose meaningful capacity, hit, and lifecycle diagnostics

- [ ] Split physical occupancy into active/pinned, cached-evictable, truly free, private tail waste, and later in-flight/host/storage categories. `usage()` currently counts evictable cached blocks as used; it can be near 100% without allocation pressure. Preserve existing scheduler counters but distinguish retries/preemptions from unique requests and successful usable hits.
- **Accept:** Accounting sums to physical capacity through share/evict/clear/failure; report eviction count, allocation failures, recomputed tokens, duplicate blocks, hash time, and hit tokens by source/group. Add optional invariant checks and stored/removed/cleared events for routing consumers; no raw prompt leakage in events by default.
- **Dependencies:** C01–C03, F01; extend with later tiers. **References:** vLLM stats/events [V1, V2]; SGLang protected/evictable sizes and metrics [S1, S2].

### F11 — Fix startup allocation peaks and budget validation

- [ ] Heterogeneous `KvBlockPool::Create` first allocates a uniform “probe,” then allocates the actual pool before replacing the probe. This transiently requires both allocations and can OOM although the final layout fits. Validate without allocating, compute offsets once, and allocate once.
- [ ] Validate explicit-byte versus utilization precedence, finite utilization in range, negative budgets, and minimum capacity. A negative profiled `kv_bytes` currently falls back to configured blocks, defeating the intended cap. Include future recurrent/checkpoint/scale/graph/transfer overhead in budgeting; the current profile does not establish those costs.
- **Accept:** Final pool fits but two pools do not; heterogeneous first layer larger than others; budget below one block; usage already over budget; exact-byte override; and unequal rank budgets. Preserve existing profiling and TP minimum reduction.
- **Dependencies:** C07, F06; F08/F09 for their extra budgets. **References:** cache configuration [V6], SGLang pool construction separation [S7].

## P2 — advanced serving and operational capabilities

### A01 — Add speculative reservation, commit, and rollback

- [ ] `Reserve` only grows and `Release` drops everything. Add tentative lookahead allocation, accepted-token commit, rejected-tail truncation, and separate draft/target state where required. Restore recurrent state as well as KV on rejection; never publish draft-only hashes.
- **Accept:** Accept none/some/all; rejection inside a shared page; EOS during speculation; allocation failure; target/draft differing layouts; and repeated rollback without leaks or hash-chain drift.
- **Dependencies:** C03, C06, F01, F02; F08 for recurrent models. **References:** vLLM lookahead/commit rules [V1, V4], SGLang speculative Mamba state [S5].

### A02 — Support request branching and mutable-tail copy-on-write

- [ ] Identical full prefixes already share, but there is no fork/clone operation for multi-sample generation, beam/tree search, or explicit branch continuation. Share immutable complete pages; copy only mutable tails/state. Keep branch-local lengths and sampling metadata. This is a workload requirement, not a claim that current vLLM V1 exposes a generic beam-search fork API.
- **Accept:** Fork on/off a page boundary; branches append different tokens; prune parent/child in any order; one branch aborts; copy fails; and recurrent state remains isolated. Hashes must describe each branch's actual KV.
- **Dependencies:** C01, F02, A01; F08 for recurrent branches. **References:** shared-block ownership [V2], state copy support [S3, S5].

### A03 — Improve prefix granularity without sharing writable bytes

- [ ] Match granularity is tied to the physical block size; common prefixes ending inside a block and full-hit last-token recomputation lose reuse. Evaluate configurable smaller pages or separate match units/checkpoints, with tail copying/recomputation where necessary. SGLang can match tokenwise at page size 1 and page-aligns otherwise; current vLLM exposes a finer `prefix_match_unit` for supported groups [S1, V6]. Neither implies arbitrary partial-page sharing is universally safe.
- **Accept:** `B-1/B/B+1` matching lengths, full prompt hit requiring logits, non-divisible group granularities, and branch tails. Compare extra metadata/copy cost against saved prefill before changing defaults.
- **Dependencies:** C03, F05, F06, A02 where writable tails are reused.

### A04 — Add host-memory offload and restore

- [ ] A CPU test pool is not a GPU↔host cache tier. Add pinned-host capacity management, block transfer descriptors, async backup/load states, residency tracking, transfer pinning, and recomputation fallback. Coordinate partial hits across groups and ranks. Keep active compute and transfer ownership separate.
- **Accept:** GPU pressure, host exhaustion, cancellation while loading/storing, failed copies, partial restore, stale completion, shutdown, and restored KV equivalence. Do not free a source before the transfer completes or expose destination KV before readiness.
- **Dependencies:** F02, F04, F06, F10. **References:** native vLLM CPU offload [V7]; SGLang host-cache interfaces [S2, S8].

### A05 — Add persistent/external cache tiers and prefetch policy

- [ ] Define versioned serialization, integrity/compatibility checks, content lookup, storage admission, eviction, and asynchronous prefetch/backup. Start with one backend; avoid tying the cache core to a specific transport. Support deadlines, partial-hit fallback, bounded queues, and backpressure; choose write-through versus write-back based on workload.
- **Accept:** Restart/reload; missing/corrupt/truncated objects; stale model/adapter/layout; storage unavailable; partially fetched prefix; namespace isolation; and writes completing after cancellation. External storage must not turn a normal miss into an indefinite wait.
- **Dependencies:** A04, F04, F10. **References:** vLLM tiering manager and offload keys [V7]; SGLang HiCache L1/L2/L3 design [S8]. SGLang L2 is instance-private; cross-instance reuse requires a suitably shared L3 backend.

### A06 — Support disaggregated prefill/decode and remote KV handoff

- [ ] There is no export/import or connector interface. Add producer/consumer metadata, receiver reservations, ready/failed completion, ownership handoff, layerwise transfer if useful, and cancellation/timeouts. Keep locally cached, remotely supplied, and newly computed lengths distinct.
- **Accept:** Partial local plus remote hit; producer dies; decoder disconnects; mismatched model/adapter/dtype/page layout; duplicate transfers; and last-token/logits handoff. Different TP layouts require an explicit resharding implementation or rejection, not reinterpretation of remote block IDs.
- **Dependencies:** F02, F04, F06, F10; A04's transfer foundation. **References:** vLLM connector/deferred publication [V1, V9]; SGLang HiCache deployment and transfer design [S8].

### A07 — Validate distributed cache agreement and add context-parallel layouts

- [ ] Existing TP mapping is useful, but `ModelExecutor::CreateGroup` only compares vocabulary sizes, not pool capacities/block sizes/layout compatibility. Validate that shared logical IDs are valid on every rank. Future independent cache residency needs agreement on usable hit length and allocation success, plus coordinated failure cleanup.
- [ ] Add pipeline/context-parallel cache partition descriptions and logical-to-local token mapping only when those execution modes exist. Distinguish replicated MQA/MLA state from sharded heads/tokens and account for transport metadata.
- **Accept:** Unequal capacities; missing shard; one-rank allocation/transfer failure; reset epoch disagreement; non-divisible sharding; and process restart. Include a multi-rank no-hang test. Do not call current TP support missing.
- **Dependencies:** C07, F04, F05, F06; A04/A06 for distributed transfers. **References:** DCP/PCP handling [V1, V3, V4]; multi-rank transfer agreement [S8].

### A08 — Support encoder/cross-attention and non-generation reuse rules

- [ ] There is one decoder-style token/block chain. Encoder-decoder serving needs separately sized encoder/cross-attention state and encoder-input identity; decoder token equality alone is insufficient. Prefix reuse may need to be bypassed when prompt logprobs, all-token embeddings, or other intermediate outputs must be recomputed.
- **Accept:** Same decoder text with different encoder inputs; independent encoder/decoder lengths/lifetimes; multimodal identity; and cache-on/off equivalence for requested prompt outputs. Do not claim encoder/cross-attention prefixes are generally cached upstream: vLLM allocates these groups but excludes them from ordinary prefix-hit estimation [V1, V4, V5].
- **Dependencies:** C03, F04–F06; model/runner output support outside `cache`.

### A09 — Add safe operational controls and cache epochs

- [ ] Expose controlled flush/reset with a result, per-request lookup bypass, and explicit policy for publication during bypass. Add targeted model/adapter/tenant invalidation if deployment needs it. Define reset behavior for live requests and all storage tiers, including reload/online weight updates.
- **Accept:** Reset refused or safely deferred when required; old-epoch completions never republish; disabled cache still retains each request's own decode KV. Correct the misleading `CacheConfig::enable_prefix_caching` comment: disabling cross-request reuse does not mean recomputing every sequence on every step.
- **Dependencies:** C01, C03, F04, F10; A05 for persistent invalidation. **References:** reset and lookup-bypass behavior [V1, V2]; SGLang cache lifecycle [S2].

### A10 — Add cache-aware admission and reuse-aware retention policies

- [ ] FCFS scheduling and release-recency LRU cannot prioritize shared-prefix batches, hot system prompts, or long-lived sessions. Expose a nonmutating hit/cost estimate, then evaluate prefix-aware scheduling with fairness/aging. Add optional retention priority, soft session references, admission thresholds, and per-namespace budgets where workloads justify them.
- **Accept:** No starvation of cold prompts; session close/cancel releases preferences; hot sessions cannot permanently pin all memory; repeated short requests do not inflate metrics. Compare policy benefits with F03's simple tail-first baseline before adding complexity.
- **Dependencies:** F01, F03, F10. **References:** vLLM hit estimation [V1], SGLang eviction strategies and soft session references [S9, S10].

### A11 — Reduce duplicate storage and hot-path metadata overhead

- [ ] Duplicate full-block hashes are deliberately allowed, as in vLLM V1; this is not inherently incorrect. Measure their cost, then consider canonicalization at safe boundaries. SGLang insertion can free duplicate slots and rewrite the request's mapping [S1]. Mapping changes require device synchronization/COW rules.
- [ ] Avoid hashing the entire prompt into candidates and then hashing it again from a fresh chain on publication. Reuse validated hash prefixes, stop unnecessary lookup work at the allowed cap, and consider preallocated block metadata/batched allocation. Current `Extend` allocates a byte vector per block; cache entries and LRU nodes allocate on the host.
- **Accept:** Concurrent identical misses, thousands of duplicate entries, long prompts, cache disabled, and forced eviction; measure host allocations, hash time, and scheduler latency. Preserve deterministic SHA-256 and valid duplicate behavior unless evidence supports a change.
- **Dependencies:** C01, C03, C06, F02, F10. **References:** [V2, V8, S1].

### A12 — Make device block-table updates graph-safe and reusable

- [ ] `BlockTable` documents fixed device buffers, but the current runner rebuilds/uploads metadata tensors each step and explicitly returns `Unimplemented` for CUDA graphs. Add stable buffers, batched/delta updates, capacity bounds, and replay-safe slot changes along with graph support.
- **Accept:** Different sequence/page counts within capture limits; prefix adoption, preemption, branch remapping, and slot reuse between replays; no stale page IDs. Guard dummy/padded entries so kernels never read or write an unrelated real block.
- **Dependencies:** C03, C07, F02, F06; runner/graph implementation outside `cache`. **References:** SGLang request-to-token pool and paged allocation [S5, S11].

## P3 — specialized workloads and efficiency

### X01 — Add sparse/selected-token and compressed auxiliary cache families

- [ ] Dense full-prefix block mappings cannot represent sparse attention's selected-token residency, retrieval/indexer caches, compressed multi-rate state, or hot/cold token pools. Add explicit auxiliary-state roles and token-position translations only for supported model families; preserve required full/compressed source state when approximate hot sets change.
- **Accept:** Selected tokens crossing pages; original positional indices; independently exhausted auxiliary pools; compression boundaries; and well-defined exact versus approximate output quality. Avoid treating a one-entry latent pool as complete sparse MLA support.
- **Dependencies:** F05–F07, F09, F10; A04 for hierarchical sparse residency. **References:** sparse roles/specs/managers [V4, V5] and specialized SGLang memory-pool families [S5, S7].

### X02 — Evaluate elasticity, backend portability, and session-tail persistence

- [ ] Fixed allocation is an intentional steady-state design, but cannot shrink/yield GPU cache memory for colocated workloads or preserve movable allocations through sleep/resume. Evaluate logical indirection or virtual-memory backing with explicit quiescence rather than introducing steady-state `cudaFree`.
- [ ] Define backend capability coverage beyond the current CUDA serving path; CPU pool bookkeeping tests alone do not prove CPU/ROCm/other-device model serving. Session workflows that retain non-page-aligned private tails need explicit ownership and expiry in addition to ordinary full-prefix reuse.
- **Accept:** Resize/suspend only at safe boundaries; address stability where graphs require it; resumed data integrity; backend-specific dtype/layout validation; and session timeout/branch/abort cleanup.
- **Dependencies:** F02, F06, F10, A02, A09; deployment/model backend work. **References:** backend-aware SGLang pools/configuration [S5–S7] and session ownership interfaces [S2]. These are optional extensions, not required for the current fixed-pool contract.

## Validation work accompanying implementation

- [ ] **T01 — Ownership/state-machine regression suite:** Extend `tests/unit/cache/` with the C01–C08 repro cases and randomized adopt/reserve/publish/release/clear/evict sequences checked against a small ownership oracle. Check uniqueness of free IDs, no evictable pinned blocks, hash-to-byte consistency, and full reclamation after teardown. Run debug and release builds so assertions do not hide behavior. **Depends on:** corresponding C fixes; tests can be written first.
- [ ] **T02 — Integration correctness:** Extend scheduler tests beyond mocked samples with runner-backed cache-on/off comparisons for chunk budgets, shared prefixes, cancellation, preemption after output, and TP agreement. Add sentinel-buffer tests for heterogeneous layouts/recurrent resets, and numerical tests for each new model/precision family. **Depends on:** C03–C08 and the relevant F/A feature.
- [ ] **T03 — Workload benchmarks:** Measure long shared documents, short prompts with long generation, divergent continuations, repeated multi-turn prefixes, duplicate concurrent misses, mixed hot/cold traffic, and forced pressure. Track TTFT, decode latency, recompute work, memory/metadata overhead, and hit tokens. Add tier bandwidth/timeouts only after transfer exists. **Depends on:** F10; establish baseline before A03/A10/A11/X01.
- [ ] **T04 — Attention-mechanism parity matrix:** Execute the appended section *“Attention-mechanism parity verification (T04)”*: land the fixture families, run every P-case to a recorded S/R/D verdict, and close the gap register (new items F12/F13/A13 plus the recorded amendments). Coverage target: every attention mechanism and cache configuration in the pinned vLLM/SGLang snapshots is supported, explicitly rejected with a tested error, or deferred with the mirrored upstream restriction cited. **Depends on:** the delivering items named per family; fixtures (F13) first.

Existing tests already cover basic SHA vectors, salted hashes, hit/miss adoption, pinned eviction, duplicate registrations, ordinary reservation failure, and disabled-cache behavior. Preserve them. Their presence is not evidence for live-clear ownership, fully computed chunk publication, recurrent-slot reset, or post-generation preemption correctness. No runtime tests were run for this documentation-only review.

## Pinned upstream source index

The implementation files below, rather than only feature descriptions, informed the comparison. SGLang's current source layout uses unified Full/SWA/Mamba components; older `hiradix_cache.py`/`mamba_radix_cache.py` paths are not the reviewed implementation. Upstream design prose can lag source behavior.

- **V1:** [vLLM KVCacheManager](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/core/kv_cache_manager.py) — lookup, allocation, computed lengths, lookahead, deferred publication, stats/events, connector hooks.
- **V2:** [vLLM block pool](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/core/block_pool.py) and [prefix design](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/docs/design/prefix_caching.md) — references, free queue, duplicates, reset, and eviction order.
- **V3:** [vLLM coordinator](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/core/kv_cache_coordinator.py) — cross-group hit agreement and granularity.
- **V4:** [vLLM attention-type managers](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/core/single_type_kv_cache_manager.py) — full/window/local/recurrent/cross-attention/sparse lifecycle rules.
- **V5:** [vLLM cache specifications](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/kv_cache_interface.py) — state types, geometry, quantization modes, and grouping.
- **V6:** [vLLM cache configuration](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/config/cache.py) and [quantized KV documentation](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/docs/features/quantization/quantized_kvcache.md).
- **V7:** [vLLM offload contracts](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/kv_offload/base.py), [CPU manager](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/kv_offload/cpu/manager.py), and [tier manager](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/kv_offload/tiering/manager.py).
- **V8:** [vLLM hash utilities](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/vllm/v1/core/kv_cache_utils.py) — adapter, multimodal, embedding, and salt identity.
- **V9:** [vLLM disaggregated prefill](https://github.com/vllm-project/vllm/blob/5d5c3d7c4360d777a4bb9c75b6be250dc8d812d1/docs/features/disagg_prefill.md).
- **S1:** [SGLang radix cache](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/radix_cache.py) — matching, split/insert, duplicate reconciliation, locks, and eviction.
- **S2:** [SGLang prefix-cache contracts](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/base_prefix_cache.py) — request ownership, checkpointing, transfers, and capabilities.
- **S3:** [SGLang Mamba component](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/unified_cache/components/mamba.py) and [checkpoint pool](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py).
- **S4:** [SGLang SWA component](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/unified_cache/components/swa.py) and [pure-SWA cache](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/pure_swa_radix_cache.py).
- **S5:** [SGLang device/request/state pools](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/memory_pool.py).
- **S6:** [SGLang cache dtype resolution](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/kv_cache_dtype.py).
- **S7:** [SGLang memory-cache organization](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/README.md).
- **S8:** [SGLang HiCache design](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/docs/docs/advanced_features/hicache_design.mdx).
- **S9:** [SGLang eviction policies](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/evict_policy.py).
- **S10:** [SGLang session-aware caching](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/docs/docs/advanced_features/session_radix_cache.mdx) and [session reference tracker](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/unified_cache/session_ref_tracker.py).
- **S11:** [SGLang paged allocator](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/allocator/paged.py) and [token allocator](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/mem_cache/allocator/token.py).

---

# Implementation plan

Appended 2026-10-15. Derived solely from the backlog above; no other files were read for this plan. The plan converts the priority/dependency graph into ordered phases, waves, and concrete per-item steps. Item IDs, bug descriptions, acceptance criteria, and dependency edges are those of the backlog; where the backlog leaves a genuine design choice open, the plan flags it as a decision to record in Phase 0 rather than choosing silently.

## Ground rules

1. **Dependency-first ordering.** An item starts only after every listed dependency has passed its acceptance criteria. Within a phase, independent items are grouped into waves and may proceed in parallel.
2. **Test-first for every defect.** Each C-item begins with a failing regression test built from its Accept list (this is also the seed of T01). The fix lands only with that test; the test is never weakened to make the fix pass.
3. **Per-item gate.** Acceptance criteria demonstrated in both debug and release builds (assertions must not be load-bearing), all pre-existing tests still green, existing tests preserved verbatim unless an item explicitly supersedes behavior.
4. **Unsupported combinations stay explicit.** Any capability that lands without full validation is rejected at configuration/startup time, never silently enabled. This applies to quantization hybrids (F09), transfer resharding (A06/A07), and recurrent APC restrictions (F08).
5. **One ownership model.** C01's ownership-versus-discoverability decision is the foundation for C02, F02, F03, and A02. Make and record it before any other P0 code.
6. **Scope anchors.** Cache work touches `src/cache/`, `include/inferx/cache/`, and `tests/unit/cache/`; integration items additionally touch `src/engine/scheduler.cc`, `src/models/model_runner.cc`, `src/engine/model_executor.cc`, `include/inferx/config/cache_config.h`, and `src/models/registry.cc` as named in the backlog.

## Phase 0 — Harness, baselines, and recorded decisions

Goal: everything needed so P0 fixes land test-first, plus the design decisions that unblock parallel work. No behavioral change ships in this phase.

### Step 0.1 — Build the ownership oracle (T01 foundation)

1. Add a test-only oracle in `tests/unit/cache/` that independently tracks, per physical block: owning request set (reference counts), cache-entry state (discoverable / evictable / pinned), and free-list membership.
2. Add invariant checkers run after every operation: free IDs are unique and disjoint from live IDs; no evictable entry is pinned; every hash index entry maps to bytes whose ownership the oracle agrees with; `free + active-pinned + cached-evictable == physical capacity`.
3. Add a randomized sequence driver over `adopt / reserve / publish / release / clear / evict` with fixed seeds, comparing manager state to the oracle after each step.
4. Run the suite in debug and release builds from the start (per T01).

### Step 0.2 — Record baselines

1. Run the full existing unit suite in both build types; record results.
2. Snapshot current scheduler counters (prefix query/hit/token) and pool occupancy reporting; these are the comparison baseline for F10 and later T03 work.

### Step 0.3 — Record the open design decisions as short ADRs

- **D1 (feeds C01, C02, F03):** ownership vs. hash discoverability split; whether `Clear` refuses under active requests (vLLM behavior [V2]) or invalidates via epoch; post-clear republication rules.
- **D2 (feeds C03, C05, C06, A01):** the distinct lengths — scheduled, allocated, computed, accepted — their API types, and who owns each watermark.
- **D3 (feeds C05):** the request state machine, including the `kWaiting → kRunning` admission transition and preemption/re-admission transitions.
- **D4 (feeds C06, A01):** strict append-only validation with rejection of shrinking spans, versus an explicit rollback API.
- **D5 (feeds C08):** "cannot fit" explicit failure policy vs. headroom/full-sequence admission, and the admission outcome for empty prompts / zero effective budgets.

**Gate G0:** oracle harness merged and green on the current (buggy) code paths where applicable; baselines recorded; D1–D5 written.

## Phase 1 — P0 correctness and progress (C01–C08)

Goal: no incorrect KV reuse, no ownership defect, guaranteed serving progress. All waves gate on the per-item acceptance lists in the backlog.

### Wave 1 — independent items (parallelizable): C01, C03, C04, C07

**Step 1.1 — C01: preserve shared ownership across prefix-cache clear**

1. Write the failing test: two manager-owned requests share a block with private tails; `ClearPrefixCache()`, release request A, allocate a third request; assert the shared block was not freed (B's KV intact) and no double-free occurs on B's release.
2. Per D1: separate allocation ownership (live reference counts) from hash discoverability (the `by_block_` lookup `Clear` currently destroys), or make `Clear` fail while requests hold blocks.
3. Add a cache epoch; publication after a clear carries the new epoch so stale in-flight completions cannot republish into the cleared index.
4. Extend tests per Accept: clear twice in a row; continued publication after clear; cancellation racing clear; clear after a weight change; the existing single-adoption clear test must keep passing.

**Step 1.2 — C03: publish only fully computed, accepted KV**

1. Write the failing test: block size 4, prompt length 8, first chunk 1 token; assert the allocated block is *not* published/hittable (today one block is published as fully computed).
2. Per D2: introduce an explicit accepted/computed-token boundary and thread it scheduler → manager → `PublishComputed`; publication covers only blocks entirely below the watermark.
3. Keep scheduled, allocated, computed, and accepted lengths distinct in the API; `PublishComputed` rejects inconsistent spans (e.g., boundary beyond allocated) instead of silently capping.
4. Cover the `clip_chunk` unaligned-tail path and capacity-reserved-ahead-of-execution explicitly in tests.
5. Extend per Accept: budgets 1 through `block_size-1`; non-aligned resumed chunks; over-reservation; failure before/within execution; prompt/output boundary blocks; a second request arriving after the short chunk must never hit uncomputed tokens.

**Step 1.3 — C04: correct recurrent-slot reset addressing**

1. Write the failing sentinel test: fill all matrix and conv slices of every slot with distinct sentinels across heterogeneous layers; reset first/middle/last slots; assert exactly the selected slot's matrix **and** conv regions were zeroed (today the contiguous zeroing clobbers a neighbor slot and misses conv state).
2. Fix `ResetSlot` to zero the two slices at their actual offsets (`slot * matrix_bytes` and `matrix_total + slot * conv_bytes` per layer), or restructure storage/views consistently — pick one representation and align `State`/`ConvState` with it.
3. Keep operations ordered on the execution stream; cover `conv_kernel_size == 1`, a single slot, invalid slot indices, and both CPU and CUDA memory implementations.

**Step 1.4 — C07: validate geometry and arithmetic before allocating or indexing**

1. Add checked-arithmetic helpers (element/byte products with overflow detection) and use them in pool creation, `BlocksForTokens`, and recurrent sizing; invalid input must fail *before* allocation.
2. Bound block/token indices: reject block counts that cannot fit the narrowed `int32_t` IDs; make `BlockTable::Locate` reject negative positions (today `-1` can yield block 0 with a negative slot); require block size ≥ 1; bounds-check `LayoutFor`.
3. Audit scheduler/runner `int` and device-table conversions for the same narrowing/truncation hazards.
4. Per Accept, test with `INT32_MAX`/`INT64_MAX`-style boundaries using checks that fail before any huge real allocation; invalid layers/dtypes; heterogeneous stride bounds; recurrent-size overflow; define the supported entry count above two (currently allocated but without a general entry-view accessor).

### Wave 2 — after C01/C03: C02, C05, C06

**Step 1.5 — C02: teardown releases each physical block exactly once**

1. Failing tests: destroy a `CacheManager` with (a) two active sharers, (b) only cached entries, (c) mixed cached/private blocks, (d) after a clear. Assert the borrowed pool ends entirely free with no duplicate IDs.
2. Replace the destructor's per-table `FreeBlocks` with the same ownership-aware release path used in normal operation (built on C01), then drain remaining unreferenced cache entries back to the pool.
3. Handle release/drain errors with real error propagation, not only debug assertions; preserve the documented "device work has finished" precondition in the destructor contract.

**Step 1.6 — C05: repair preemption transitions and replay the accepted sequence**

1. Add the admission transition so admitted requests become `kRunning` (per D3); `Request::Preempt` then accepts normal scheduler victims.
2. Change preemption to retain prompt **plus accepted output**; re-admission hashes the replayed sequence only up to the cached/valid watermark so accepted-output hashes cannot attach to KV recomputed for a different continuation.
3. Restore sampling/RNG state and stopping conditions without re-emitting old tokens; reuse still-valid generated-prefix KV where present instead of full recompute.
4. Per Accept: preempt after several sampled tokens including mid-block; compare resumed greedy output and seeded sampling against an uninterrupted reference; all-prefix-evicted, partially retained, salted, repeated-preemption, and aborted-victim cases.

**Step 1.7 — C06: initialize prefix state for short prompts; enforce append-only publication**

1. Failing test: prompt shorter than one block; decode crosses the first full block; assert the block publishes (today `AdoptPrefix` returns before creating hash state, so it never can).
2. Initialize the salted hash chain even on zero-hit/short prompts.
3. Per D4: either validate monotonic committed length and immutability of committed tokens in `PublishComputed` (rejecting shrinking/mutating spans), or add a separate rollback API; do not leave the current silent `hashed_blocks` decrease that corrupts `chain`.
4. Define and test: empty prompts, repeated adoption, `Reserve` before adoption, request-ID reuse, negative adoption caps; keep the valid uncapped low-level API if intentionally supported, with the `N-1` cap still caller-enforced.

### Wave 3 — after C03/C05/C07: C08

**Step 1.8 — C08: guarantee progress under impossible or fragmented demands**

1. Admission-time validation: reject (per D5) prompts that can never fit the pool — including via incremental chunked admission — with an explicit failure rather than silent retry loops.
2. Distinguish "retry after reclaim" from "cannot fit" in scheduler outcomes; add headroom or full-sequence-fit admission where configured; keep an impossible queue head from permanently blocking smaller later requests (or make that an explicit documented policy).
3. Account separately for shared pins, recurrent slots, and future lookahead when deciding fit; define outcomes for empty prompts and zero/negative effective budgets.
4. Per Accept: prompt one token beyond capacity; decode crossing the last available page; all blocks pinned; zero-progress head with a smaller request behind it; zero/negative configuration; constrained chunk budgets. Every accepted request must advance, yield through a defined policy, or fail explicitly.

**Gate G1 (P0 complete):** all C01–C08 acceptance lists pass in debug and release; the Phase 0 oracle suite is green over randomized sequences; T01's C-item repro cases are all checked in. Do not start Phase 2 before G1.

## Phase 2 — P1 lifecycle foundations and model coverage (F01–F11)

Order follows the backlog's suggested sequence: F01–F04 with F10 alongside, then F05–F09, then F11. F01–F04 are mutually independent once their C-dependencies hold; F05 and F06 serialize the rest.

**Step 2.1 — F01: failed-reservation behavior matches its contract**

1. Failing test: growth fails with both pinned and evictable cache entries present; today `EvictFor` destroys evictable entries before discovering capacity is still insufficient.
2. Preflight `free + evictable ≥ needed` before any destructive eviction — or explicitly revise the contract and expose the eviction effect; pick one and document it.
3. Add a non-mutating capacity-estimation API for scheduler planning, separated from adoption/pinning.
4. Make `FreeBlocks` atomic in ownership terms: validate all IDs (invalid/duplicate) before freeing any, so bulk failure cannot leave ambiguous partial ownership.

**Step 2.2 — F02: completion-aware leases and robust request cleanup**

1. Introduce in-flight read/write leases (or stream-event fences) so physical reuse waits for the last consumer; specify scheduler-owner serialization rather than relying on a mutex for GPU lifetime.
2. Preserve today's simple synchronous path as the degenerate case; it must not regress.
3. Per Accept: cancel/preempt/clear/fail/shutdown with outstanding reads or writes; stale completion arriving after request-ID reuse; two streams; repeated cleanup; no early publication and no recycled-block access.

**Step 2.3 — F03: retain reusable ancestors when evicting**

1. Failing/behavioral test: cache `A/B/C`, reclaim one block, request `A/B/D`; today forward-order LRU insertion evicts the earliest block and strands the suffix hashes (adoption stops at the first miss).
2. Change `PrefixCache::Release` to insert tail-first (vLLM frees in reverse order [V2]) or adopt a path-preserving policy respecting cross-request recency.
3. Cover overlapping requests, duplicate hashes, pinned ancestors, release-order ties; measure achieved *reuse* under pressure, not just entry counts.

**Step 2.4 — F04: cache identity covers all KV-affecting inputs**

1. Define typed, versioned identity metadata: LoRA/adapters, multimodal placeholders, prompt embeddings, position offsets/RoPE configuration, model revisions, externally persisted KV — plus a model/cache epoch. Keep storage-compatibility metadata (dtype/layout/sharding) distinct from semantic identity. Never hash sampling settings that cannot affect computed KV.
2. Per Accept: identical token placeholders with different images/embeddings/adapters must not share; identical effective inputs must; multimodal span crossing blocks; model reload; changed positional semantics; incompatible imported dtype/layout/sharding rejected; deterministic serialization and salt isolation preserved; request-path salt propagation added where serving APIs introduce it.

**Step 2.5 — F10 (alongside 2.1–2.4): capacity/hit/lifecycle diagnostics**

1. Split occupancy into active/pinned, cached-evictable, truly free, and private tail waste (later tiers: in-flight/host/storage); fix `usage()` so evictable cached blocks no longer read as allocation pressure.
2. Keep existing scheduler counters but distinguish retries/preemptions from unique requests and successful usable hits; add eviction count, allocation failures, recomputed tokens, duplicate blocks, hash time, hit tokens by source/group.
3. Add optional invariant checks and stored/removed/cleared events for routing consumers; no raw prompt content in events by default.

**Step 2.6 — F05: typed cache groups with independent allocation semantics** *(after C07, F01, F04)*

1. Introduce group specifications — full attention, windowed/local, recurrent state, non-cacheable scratch — with independent block mappings/capacities and a common valid-hit boundary across groups.
2. Add layer→group mapping and explicit layer KV aliasing (a model reusing another layer's K/V allocates once, frees once).
3. Per-group exhaustion must roll back atomically; groups may report different hit lengths; report padding and unusable capacity.
4. Cover unequal layer counts/dtypes/head dimensions, different block sizes per group, shared physical buffers. Do **not** bake in a two-attention-types restriction — follow the generalized group iteration of current vLLM source [V3, V4], not the older design document.

**Step 2.7 — F06: explicit physical layout and backend capability contracts** *(after C07; F05 for grouped layouts)*

1. Add layout descriptors with strides/alignment/padding and supported block-size validation at the backend boundary, replacing the fixed `[layer][entry][block][token][head][dim]` assumption.
2. Provide auxiliary/packed planes as first-class descriptors rather than overloading `entries_per_token`; the one-entry `KeyCache` stays latent storage, not a claim of full MLA/indexer/quantization buffers.
3. Per Accept: MHA/MQA/GQA geometries, non-sharded MLA latent, unequal K/V dimensions where supported, NHD/HND/packed layouts, and rejection of incompatible kernels; every tensor view and transfer descriptor must agree on bytes and strides.

**Step 2.8 — F07: bounded sliding-window, local-chunk, and sink retention** *(after C03, F02, F05, F06)*

1. Implement per-group out-of-window reclamation while the request continues: logical-position mapping and window-aware prefix eligibility (full attention retains everything; local attention retains only the window/chunk).
2. Define distinct retention/cacheability rules for attention sinks and circular scratch buffers.
3. Per Accept: window smaller than a page; boundaries inside pages; prefill chunks larger than the window; pinned shared blocks; long decode with bounded local memory; a hit whose early full-history pages were evicted; never retire pages needed by an in-flight or speculative step.

**Step 2.9 — F08: recurrent/SSM state ownership and prefix checkpoints** *(after C04, C05, F02, F05)*

1. Give `RecurrentStatePool` a slot allocator, request mapping, checkpoint capture/restore, copy-on-write, and prefix-index integration; make the runner construct the pool and accept paged-state models instead of rejecting them.
2. Alignment rule: restore exact matrix **and** convolution state at the same boundary as reused attention KV; branch into independent mutable slots; evict snapshots separately from active state; recover after preemption; unequal state dtypes/shapes; independent slot exhaustion.
3. Keep prefix caching disabled/rejected where complete state restoration is unavailable; do not promise DCP/PCP-style combinations by analogy with vLLM's align-mode-only Mamba APC [V4].

**Step 2.10 — F09: quantized KV as a complete storage/read/write path** *(after F04, F06; F05 for mixed groups)*

1. Cache dtype selection independent of model dtype; scale granularity/lifetime; calibration or supported dynamic scaling; append-side quantization and attention-side dequantization; mixed-precision layer overrides.
2. Only then, packed sub-byte KV with explicit scale/packing metadata if target kernels justify it.
3. Startup-reject unsupported format/device/kernel combinations. Per Accept: zero/large/NaN inputs, scale validation, cached-vs-uncached tolerance, scales preserved through sharing/copy/offload, mixed layer formats, packed tails/padding. Quantized recurrent checkpoints are a separate decision with their own quality tests.

**Step 2.11 — F11: startup allocation peaks and budget validation** *(after C07, F06; revisit budgets as F08/F09 land)*

1. Replace heterogeneous `KvBlockPool::Create`'s probe-then-allocate pattern (transiently two full pools in memory) with validate-without-allocating, compute offsets once, allocate once.
2. Enforce explicit-byte vs. utilization precedence, finite in-range utilization, non-negative budgets, minimum capacity; a negative profiled `kv_bytes` must not silently fall back to configured blocks.
3. Include recurrent/checkpoint/scale/graph/transfer overhead in budgeting as those features land; preserve existing profiling and the TP minimum reduction.

**Gate G2 (P1 complete):** F01–F11 acceptance lists pass; T02 extended per below; existing profiling/TP behavior preserved.

## Phase 3 — P2 advanced capabilities (select per deployment)

Three dependency tracks from the backlog, plus standalone serving items. A deployment should pick tracks by need; each track's steps are internally ordered.

### Track A — speculative decoding and branching (A01 → A02 → A03)

1. **A01 reservation/commit/rollback** *(after C03, C06, F01, F02; F08 for recurrent models)*: tentative lookahead allocation; commit of accepted tokens; truncation of the rejected tail; separate draft/target state where required; restore recurrent state as well as KV on rejection; never publish draft-only hashes. Cover accept none/some/all, rejection inside a shared page, EOS during speculation, allocation failure, differing draft/target layouts, repeated rollback without leaks or hash-chain drift.
2. **A02 fork and mutable-tail copy-on-write** *(after C01, F02, A01)*: share immutable complete pages, copy only mutable tails/state; branch-local lengths and sampling metadata; hashes describe each branch's actual KV. Cover fork on/off page boundaries, divergent appends, pruning parent/child in any order, one branch aborting, copy failure, isolated recurrent state.
3. **A03 finer prefix granularity** *(after C03, F05, F06, A02 where writable tails are reused)*: evaluate configurable smaller pages or separate match units with tail copy/recompute; `B-1/B/B+1` matching lengths, full-prompt hits still requiring logits, non-divisible group granularities. Compare metadata/copy cost against saved prefill (needs the T03 baseline) before changing defaults.

### Track B — offload, persistence, and transfer (F02+F04+F06 → A04 → A05 → A06)

1. **A04 host-memory offload/restore** *(after F02, F04, F06, F10)*: pinned-host capacity management, block transfer descriptors, async backup/load states, residency tracking, transfer pinning, recomputation fallback; coordinate partial hits across groups and ranks. Never free a source before its transfer completes or expose destination KV before readiness.
2. **A05 persistent/external tiers and prefetch** *(after A04, F04, F10)*: versioned serialization, integrity/compatibility checks, content lookup, storage admission/eviction, async prefetch with deadlines, partial-hit fallback, bounded queues and backpressure; one backend first, cache core stays transport-agnostic; write-through vs. write-back chosen by workload. Storage unavailability must not turn a miss into an indefinite wait.
3. **A06 disaggregated prefill/decode and remote handoff** *(after A04's transfer foundation plus F02, F04, F06, F10)*: export/import connector interface, producer/consumer metadata, receiver reservations, ready/failed completions, ownership handoff, optional layerwise transfer, cancellation/timeouts; keep locally cached, remotely supplied, and newly computed lengths distinct; mismatched TP layouts require explicit resharding or rejection — never reinterpret remote block IDs.

### Track C — serving and operational items (independent; order by deployment need)

1. **A07 distributed agreement and (later) context-parallel layouts** *(after C07, F04–F06; A04/A06 for transfers)*: extend `ModelExecutor::CreateGroup` beyond vocabulary comparison to pool capacities/block sizes/layout compatibility; validate shared logical IDs on every rank; coordinated failure cleanup; a multi-rank no-hang test; add CP partition descriptions only when those execution modes exist.
2. **A08 encoder/cross-attention and non-generation reuse rules** *(after C03, F04–F06; output support outside `cache`)*: separately sized encoder/cross-attention state with encoder-input identity (decoder token equality alone is insufficient); bypass reuse when prompt logprobs/embeddings/intermediate outputs must be recomputed.
3. **A09 operational controls and cache epochs** *(after C01, C03, F04, F10; A05 for persistent invalidation)*: controlled flush/reset returning a result; per-request lookup bypass with an explicit publication policy during bypass; targeted model/adapter/tenant invalidation; defined reset behavior for live requests and all tiers; fix the misleading `enable_prefix_caching` comment.
4. **A10 cache-aware admission and retention** *(after F01, F03, F10)*: non-mutating hit/cost estimate; prefix-aware scheduling with fairness/aging; optional retention priority, soft session references, admission thresholds, per-namespace budgets — each only where workloads justify it, and always compared against F03's tail-first baseline; no starvation of cold prompts, no permanent full-memory pinning by hot sessions.
5. **A11 duplicate-storage and hot-path metadata reduction** *(after C01, C03, C06, F02, F10)*: measure duplicate-hash cost before any canonicalization (duplicates are deliberately allowed and not inherently wrong); reuse validated hash prefixes instead of re-hashing the full prompt at publication; stop lookup work at the allowed cap; preallocate block metadata / batch allocation; replace `Extend`'s per-block byte vector and host-side entry/LRU allocations where measurements justify it.
6. **A12 graph-safe, reusable device block tables** *(after C03, C07, F02, F06; runner/graph work outside `cache`)*: stable device buffers instead of per-step rebuild/upload; batched/delta updates with capacity bounds; replay-safe slot changes for adoption/preemption/branch remapping/slot reuse; guard dummy/padded entries so kernels never touch a real unrelated block.

## Phase 4 — P3 specialized workloads (X01–X02)

1. **X01 sparse/selected-token and compressed auxiliary families** *(after F05–F07, F09, F10; A04 for hierarchical residency)*: explicit auxiliary-state roles and token-position translations for supported model families only; preserve required full/compressed source state when approximate hot sets change; define exact-vs-approximate output quality; do not present one-entry latent storage as complete sparse MLA.
2. **X02 elasticity, backend portability, session-tail persistence** *(after F02, F06, F10, A02, A09)*: evaluate logical indirection or virtual-memory backing with explicit quiescence (no steady-state `cudaFree`); define backend capability coverage beyond the CUDA serving path (CPU bookkeeping tests alone prove nothing); explicit ownership and expiry for non-page-aligned session tails.

## Validation integration (mapping T01–T03 onto the phases)

- **T01** starts in Phase 0 (oracle) and absorbs each C-item's failing test as it lands; extend it with each F-item's ownership-relevant cases. Run debug and release throughout.
- **T02** gates Phase 2 and every advanced feature: runner-backed cache-on/off comparisons for chunk budgets, shared prefixes, cancellation, preemption-after-output, and TP agreement; sentinel-buffer tests for heterogeneous layouts and recurrent resets; numerical tests added with each new model/precision family (F08, F09) and each track item.
- **T03** baselines are captured immediately after F10 lands and **before** any of A03/A10/A11/X01 begins (their acceptance depends on measured deltas); tier bandwidth/timeout metrics are added only after A04/A05 exist.

## Milestone gates

| Gate | Reached after | Exit criteria |
| --- | --- | --- |
| G0 | Phase 0 | Oracle harness green on current code; baselines recorded; decisions D1–D5 written |
| G1 | Phase 1 (C01–C08) | All P0 acceptance lists pass in debug and release; randomized oracle suite green; T01 C-cases checked in |
| G2 | Phase 2 (F01–F11) | All P1 acceptance lists pass; T02 extended; profiling and TP-minimum behavior preserved |
| G3a / G3b / G3c | Track A / B / C selections | Per-item acceptance lists of the chosen track items |
| G4 | Phase 4 (X01–X02) | Specialized-item acceptance lists, with T03 deltas measured against pre-change baselines |

## Execution risks and watchpoints

- **C01's decision dominates.** Every later ownership feature (C02, F02, F03, A02, A04) inherits D1; keep the epoch mechanism minimal and avoid revisiting it mid-phase.
- **C03's watermark threads through the serving stack.** It changes scheduler↔manager↔runner interfaces; land it before C05/C06 to avoid reworking their tests twice.
- **F02 must not tax the hot path.** Benchmark the scheduler loop before/after leases; the synchronous single-threaded path remains the default contract.
- **F05/F06 are the serialization point of Phase 2** — F07/F08/F09/F11 and Tracks B/C all wait on them; prioritize their review.
- **Keep upstream claims honest.** Where the backlog notes upstream restrictions (vLLM Mamba align-mode-only APC; SGLang L2 instance-private; vLLM V1 lacking generic beam-search fork), InferX must not advertise those combinations without its own validation — reject them explicitly until then.

---

# Attention-mechanism parity verification (T04)

Appended 2026-10-15 (second addendum, after the implementation plan). Question this section answers: **once every item above — including the new F12/F13/A13 items this section adds — is implemented, does InferX support every attention mechanism and cache configuration available in the pinned vLLM and SGLang snapshots, or reject it explicitly?** Inputs: the pinned upstream files, one new upstream anchor, and the InferX tree itself (no runtime tests were run for this addendum either).

New upstream anchor:

- **S12:** [SGLang attention backend registry](https://github.com/sgl-project/sglang/blob/0cf683ea8d27041da981f7875e1755b9f7772799/python/sglang/srt/layers/attention/attention_registry.py) — backend names, MLA/sparse/hybrid-linear wrappers, and platform allowlists.

InferX anchors describing today's state, cited throughout: model registry `src/models/registry.cc:24` (exactly Qwen3, Qwen3-MoE, Llama; BF16 execution only); kernel surface `src/ops/attention.cc:22` (FlashInfer geometry gate: head_dim 64/128/256, GQA ratio ≤ 32, no sinks), `src/ops/cuda/flash_attention.cu:113` (FlashInfer `window_left`), `src/ops/cuda/generic_attention.cu:64` and `:112` (generic window floor and denominator-only sinks), `src/ops/mla.cc` (unwired MLA assembly op), `src/ops/gdn.*` (unwired GDN ops); cache `include/inferx/cache/kv_block_pool.h:24` (`KvLayout.entries_per_token`; one-entry latent reads through `KeyCache()` only); capability gates `src/models/qwen3/qwen3_config.cc:113` and `src/models/llama/llama_config.cc:52` (sliding-window checkpoints rejected), `src/models/model_runner.cc:266` (models with no paged state rejected), `src/models/model_runner.cc:289` (CUDA graphs unimplemented), `src/engine/model_executor.cc:55` (TP requires eager), `include/inferx/config/scheduler_config.h:25` (`enable_chunked_prefill` parsed but never enforced); CLI block sizes `apps/cli/args/engine_args.h:56` ({16, 32, 64}); a local production sink+window checkpoint exists at `models/gpt-oss-20b`.

## Verdict rules

1. **Parity means cache-behavior parity plus observable output equivalence**, not porting upstream kernels, backend names, or flag names: allocation units, retention math, prefix-hit eligibility and granularity, preemption/recompute treatment, quantized-storage contracts, sharding obligations, and transfer eligibility. Numerical gates reuse the existing harnesses — synthetic-checkpoint FP32 CPU reference (`tests/unit/models/qwen3_forward_test.cc`), golden trace with logits-capture tie escape (`tests/integration/models/model_test.cc`), TP equivalence (`tests/integration/models/tensor_parallel_test.cc`), cache-on/off comparison (T02) — with ownership checked by the T01 oracle and residency by F10 accounting. Freeing behavior is never verified by outputs alone.
2. Every case carries one verdict: **S** (supported; case passes), **R** (rejected at startup/configuration with a tested, actionable error — the rejection itself is a test), or **D** (deferred: upstream itself restricts the combination and InferX mirrors the restriction explicitly). Verdicts describe the finished system; today's InferX gates may already satisfy an R verdict and must open when the delivering item lands. No case passes by silently disabling the mechanism under test (e.g. forcing prefix caching off to avoid a reuse bug).
3. Combinations upstream itself rejects stay R/D in InferX with the upstream restriction cited — DCP/PCP assertions on the window/local/mamba managers [V4], SGLang's Blackwell hybrid-GDN backend allowlist [S12], Mamba APC align-mode-only [V4] — never "supported by analogy" (ground rule 4 of the plan).

## Fixture families (prerequisite for every section below)

One tiny synthetic family per mechanism, extending the synthetic-checkpoint-plus-FP32-reference pattern; TP variants use the Loopback world of `tensor_parallel_test.cc`, and graph variants exist where A12 applies. Fixtures for windowed/sink/CLA/hybrid/recurrent/MLA families are the deliverable of F13 (FX-cross, FX-sparse, FX-alias belong to A08, X01, F05).

| Fixture | Mechanism | Shape sketch (tiny) | Delivered by |
| --- | --- | --- | --- |
| FX-full | full attention | 2–4 layers; kv_heads ∈ {heads, heads/4, 1}; head_dim 64/128/256 | exists (Qwen3/Llama/Qwen3-MoE) |
| FX-swa | pure sliding window | all layers windowed; window ∈ {1, 8, 17, 64} × block {16, 32} | F07, F13 |
| FX-sink | sinks; sink+window | sink_len 4·B; window 8; production anchor `models/gpt-oss-20b` | F07, F13 |
| FX-cla | chunked local | chunk 32 × block 16; hybrid 5 local : 1 global | F07, F13 |
| FX-hyb | full+window hybrid | 2 full + 2 window layers, unequal group geometry | F05, F07, F13 |
| FX-mamba | attention+mamba2 hybrid; pure recurrent | per-layer matrix+conv state, unequal shapes/dtypes | F08, F13 |
| FX-gdn | full+GDN (Qwen3-Next-like) | GDN ops already exist (`src/ops/gdn.*`) | F08, F13 |
| FX-mla | MLA latent | kv_lora 16 + rope 8; `entries_per_token = 1`; dense+MLA hybrid variant | F06, F12 |
| FX-cross | encoder–decoder | encoder len ≠ decoder len; cross-attention layers | A08 |
| FX-sparse | latent + indexer/top-k | selected-token plane + source plane | X01 |
| FX-alias | cross-layer KV sharing | two layers aliasing one group's storage | F05 |

## P-01 — Full attention (MHA / MQA / GQA)

**Upstream surface:** `FullAttentionSpec` (with optional recorded `sliding_window` when treated as full, `non_causal` for prefix-LM) and `FullAttentionManager` — full retention, left-to-right chained-hash hits, fine-grained partial-block probing, partial-tail caching [V4, V5]; `MHATokenToKVPool` with asymmetric `v_head_dim` and NHD/HND layouts [S5]; the default mechanism of every MHA backend [S12]. **InferX delivery:** C01–C08, F01–F06, F09–F11; A01/A04–A06/A12 where enabled. **Today:** S for dense BF16 Qwen3/Llama/Qwen3-MoE (golden-trace and TP gates already exist).

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-01.1 | Geometry sweep: MHA, GQA ratios 4/8, MQA; head_dim 64/128/256 on the FlashInfer path | Outputs match FP32 reference within existing tolerance (S) |
| P-01.2 | GQA ratio > 32 (`src/ops/attention.cc:22`) and head_dim ∉ {64,128,256} route to the generic kernel | Dispatch is exercised; outputs still match reference; no silent wrong-kernel path (S) |
| P-01.3 | Asymmetric K/V head dims (`head_size_v` / `v_head_dim`) | Supported where the F06 backend contract allows, else R at startup with the geometry named (S/R per F06) |
| P-01.4 | block_size 16/32/64 (CLI) and library-level 1/4 (as scheduler tests) | Token→block mapping, `[block][token][head][dim]` views verified with per-slot sentinels; invalid sizes rejected (S; C07/F06) |
| P-01.5 | Prefix caching on vs off, identical prompts | Outputs equal within tolerance; with caching on, the second prompt schedules zero prefill tokens (F10 hit-token counter), full-prompt hit still computes the last-token logits under the N−1 cap and matches uncached logits (S) |
| P-01.6 | Chunked prefill budgets < block, = block, unaligned tail (`clip_chunk`) | No uncomputed KV published or hit (C03); outputs equal the unchunked run (S) |
| P-01.7 | Preemption mid-prompt and after N sampled tokens, incl. mid-block | Resumed greedy and seeded sampling equal the uninterrupted reference (C05) (S) |
| P-01.8 | TP ∈ {2, 4}: heads and kv_heads divisibility; TP > kv_heads | Shard when divisible, replicate or reject per F06/A07 contract; TP=2 logits ≈ TP=1 (existing pattern); per-rank KV bytes = rank kv_heads × dims (S) |
| P-01.9 | CUDA-graph decode replay with changing sequence/page counts | Graph and eager token streams identical; no stale page IDs (A12) (S; R while A12 open — test the existing error) |
| P-01.10 | Prompt one token beyond pool capacity; decode crossing the last free page | Explicit failure or defined preemption/stop — never a silent retry loop (C08) (S) |
| P-01.11 | Softcap variant (Gemma-2 lineage; kernel exists `src/ops/cuda/flash_attention.cu:169`) | FX with softcap matches reference for supported geometry; others R (S/R) |
| P-01.12 | `non_causal` (prefix-LM) prompts: KV of a position depends on the whole chunk | Reuse/adoption rules defined and tested or R at config (F04 amendment; upstream adjusts chunked prefill and prefix caching [V5]) (S/R) |

## P-02 — Sliding-window attention (bounded window)

**Upstream surface:** `SlidingWindowSpec` (`sliding_window`, `extra_retained_tokens`) and `SlidingWindowManager` — skip = `num_computed − window + 1 − extra_retained_tokens`, right-to-left hit requiring `ceil((window−1)/B)` contiguous cached blocks, DCP/PCP asserted 1 [V4, V5]; `RSWASpec` — prefill globally visible, decode keeps only `rswa_window`, gap blocks freed, memory O(prefix + window) [V4, V5]; SGLang SWA component — `full_window_pages = ceil(window/page_size)`, page-aligned cacheability, per-request SWA reuse capped at ≈ one window (match validator stops at `sliding_window_size`), out-of-window slot freeing with retain floor, tombstoned SWA nodes under intact full-KV ancestors [S4]; `swa_bounded_replay` config [V6]. **InferX delivery:** F07 (retention + eligibility), F05/F06 (group geometry), F02 (no retirement of pages needed in flight), A03 (page-1 granularity). **Today:** R — window checkpoints rejected at config (`src/models/qwen3/qwen3_config.cc:113`, `src/models/llama/llama_config.cc:52`) although kernels support masking.

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-02.1 | FX-swa decode to length ≫ window | Resident window blocks per request bounded by `ceil((window−1)/B) + 1` (plus alignment); F10 shows steady per-group occupancy; outputs equal a no-free reference build (S) |
| P-02.2 | window < block (8 × 16) | No partial-block freeing: retention rounds to whole blocks covering the window; rounding documented and tested; outputs equal (S) |
| P-02.3 | window crossing a block boundary (2B+1, k·B+1) | Exactly the covering blocks retained; blocks below the window start retired mid-request; never a page needed by an in-flight/speculative step (F02 leases) (S) |
| P-02.4 | Window-group hit semantics | Adoption equals a suffix of `ceil((window−1)/B)` contiguous cached blocks scanned right-to-left; pre-window blocks never returned as hits for the window group [V4] (S) |
| P-02.5 | Per-request SWA reuse cap | Cross-request window-layer reuse limited to ≈ one window of tokens (validator cap analog [S4]); longer full-attention hits cannot extend the window-group hit (S) |
| P-02.6 | Hybrid boundary (FX-hyb): full group hit 200, window group hit 40 | Adoption = min across groups [V3]; recompute starts at the common boundary; both groups' F10 hit counters report their own lengths (S) |
| P-02.7 | Preemption of a windowed request | Replay retains only the window; resumed output equals the uninterrupted reference (C05+F07) (S) |
| P-02.8 | window ≥ max_model_len | Behaves as full attention: nothing retired, full prefix cacheable (S) |
| P-02.9 | Pure-SWA model, block/page size 1 | Tokenwise retirement and tokenwise matching (A03, SGLang page-1 parity [S1, S4]); memory bounded per token (S) |
| P-02.10 | Prefill chunk larger than the window | Intermediate out-of-window KV retired before request completion; outputs equal a no-free build (S) |
| P-02.11 | `extra_retained_tokens` with speculative re-prefill (A01/A13) | Retained tail tokens are present-but-not-attended; correctness after multi-module re-prefill [V5] (S with A13; D until then) |
| P-02.12 | RSWA and bounded-replay SWA (incl. `SlidingWindowMLASpec.bounded_replay`, DSv4) | Implement the pinned retention math (O(prefix+window), replay tokens = window) or R citing the missing mechanism — no partial silent approximation (S/R; recorded in the F07/X01 amendments) |
| P-02.13 | Window mechanism × DCP/PCP | R mirroring upstream assertions [V4] (D/R) |

## P-03 — Attention sinks

**Upstream surface:** `SinkFullAttentionSpec` (`sink_len > 0`, asserted a positive multiple of block_size) and `SinkFullAttentionManager` — otherwise full-attention behavior [V4, V5]; gpt-oss-style sink+window local layers. **InferX delivery:** F07 retention rules, F05 grouping; numerics via the existing generic kernel (`src/ops/cuda/generic_attention.cu:112`, denominator-only sinks). **Today:** R — no model passes `AttentionParams.sinks`; `gpt_oss` unregistered (`tests/unit/models/registry_test.cc:31`).

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-03.1 | sink_len = 4·B fixture | Sink blocks pinned for the request lifetime, never retired, cacheable as ordinary prefix blocks; `SinkFullAttention` alignment rule honored [V4] (S) |
| P-03.2 | sink + window (FX-sink; production anchor `models/gpt-oss-20b`) | Resident set = sink blocks + window-covering blocks; sink pages excluded from window retirement; outputs equal a no-free reference (S) |
| P-03.3 | sink_len not a positive multiple of block_size | R at startup mirroring the upstream assertion [V4] (R) |
| P-03.4 | Sink numerics | Denominator-only sink math equals a reference implementing additive bias on the softmax denominator (S) |
| P-03.5 | Sinks select the generic kernel | Presence of `sinks` routes away from the FlashInfer path (`src/ops/attention.cc:22`); dispatch tested explicitly (S) |
| P-03.6 | Sink blocks shared by concurrent requests via prefix cache | Reference counts and eviction exclusion verified by the T01 oracle (S) |

## P-04 — Chunked local attention (Gemma-3-style local/global interleave)

**Upstream surface:** `ChunkedLocalAttentionSpec` (`attention_chunk_size`); manager skips `(num_computed // chunk) · chunk` tokens, nulls blocks entirely before the current chunk start, looks up from the chunk-start block, requires block_size == alignment_tokens, EAGLE/DCP/PCP unsupported [V4, V5]. **InferX delivery:** F07 (chunk retention), F05 (grouping). **Today:** R — no family, no retention.

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-04.1 | chunk = k·B (32 × 16) | Previous-chunk blocks retired at chunk boundaries only; outputs equal a no-free build (S) |
| P-04.2 | chunk not a multiple of block size | R at startup citing the alignment requirement [V4] (R) |
| P-04.3 | Decode crossing a chunk boundary | Retirement happens exactly at the boundary; the in-flight chunk never loses needed pages (F02) (S) |
| P-04.4 | Hit semantics | Lookup proceeds from the chunk-start block to the first miss; pre-chunk blocks are nulls [V4] (S) |
| P-04.5 | Hybrid 5:1 local:global (FX-cla) | Common adoption = min(local group, full group) [V3]; local-group capacity cannot starve the full group's prefix (per-group pools, F05) (S) |
| P-04.6 | Prefill chunk larger than the chunk size | Intermediate chunks retired during prefill; outputs equal (S) |
| P-04.7 | CLA × EAGLE/DCP/PCP | R mirroring upstream [V4] (D/R) |

## P-05 — Hybrid multi-group models and cross-layer KV sharing

**Upstream surface:** per-type managers coordinated by the KV cache coordinator (valid hit = agreement across groups) [V3]; `UniformTypeKVCacheSpecs` groups same-type layers — different window sizes are different types; `KVCacheTensor` layer/block strides let physical bytes alias across groups, sound because a block ID has one owning group at a time [V5]; hybrid backend wrappers and platform allowlists (GDN, Mamba2, KDA, Lightning RWKV, short-conv) [S12]. **InferX delivery:** F05 (typed groups, layer→group mapping, aliasing), F06 (layouts), F11 (single-allocation heterogeneous pool). **Today:** heterogeneous per-layer `KvLayout` exists at storage level only; all layers share one logical block sequence.

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-05.1 | Unequal per-group layer counts, dtypes, head dims, block sizes (FX-hyb) | Independent mappings and capacities; one group's exhaustion rolls back atomically without corrupting the other (F05) (S) |
| P-05.2 | Three-way hybrid full + window + sink (gpt-oss shape) | Common valid-hit = min across all three; per-group hits reported separately (F10) (S) |
| P-05.3 | Same-type grouping rule | Two window layers with different window sizes form two groups, not one [V5]; grouping decision tested at config (S) |
| P-05.4 | Layer aliasing (FX-alias; `kv_sharing_fast_prefill` YOCO/Gemma-3n analog [V6]) | Aliased layers allocate once, free once, count once in occupancy; stride/offset arithmetic agrees on bytes everywhere a view or transfer descriptor is built (S) |
| P-05.5 | Cross-group identity | A window group's blocks are never published as full-attention prefix hashes (or vice versa); hash identity encodes group/type (F04) (S) |
| P-05.6 | Hybrid GDN end-to-end (FX-gdn; Qwen3-Next-like) | Prefill+decode through the real scheduler/runner; ops already exist (`src/ops/gdn.*`); FP32-reference tolerance (S after F08+F13) |
| P-05.7 | Mechanism × kernel allowlists | Unvalidated combinations (e.g. quantized window KV without a kernel; GDN on an unsupported geometry) R at startup with the constraint named — mirroring SGLang allowlists [S12], ground rule 4 (R) |
| P-05.8 | Heterogeneous pool startup | Final pool fits in one allocation; no probe-then-allocate double peak (F11); first layer larger than the rest (S) |

## P-06 — Multi-head latent attention (MLA)

**Upstream surface:** `MLAAttentionSpec` — one latent vector per token, `head_size_v = 0` (no separate V), its own `cache_dtype_str`/alignment/model_version, sparse-cache roles and index-group leader flag, optional `non_causal_multi_token_decode` [V5]; `SlidingWindowMLASpec` bounded replay (DSv4) [V5]; `HiddenStateCacheSpec` for hidden-state extraction [V5]; MLA backends (trtllm/cutedsl/tokenspeed/flashmla/flashinfer-MLA) [S12]; `fp8_ds_mla`/`nvfp4_ds_mla` cache dtypes [V6]. **InferX delivery:** F06 (latent + rope planes as first-class descriptors — the one-entry `KeyCache()` latent is storage only), F05 (dense+MLA groups), F09 (quantized latent), **F12 (new — execution wiring)**, A07 (replication semantics). **Today:** R — latent storage and `AssembleMlaCaches` (`src/ops/mla.cc`) exist but no model family, runner path, or TP test reaches them.

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-06.1 | FX-mla geometry | Per-token latent = kv_lora + rope dims; cache size independent of query-head count; `ValueCache()` contract error respected; K/V planes byte-exact under sentinels (S) |
| P-06.2 | Execution (F12) | Absorbed-decode path — or decompressed assembly via `AssembleMlaCaches` — matches the FP32 reference within tolerance for prefill and decode (S after F12; R before) |
| P-06.3 | TP with replicated latent | Latent bytes identical and replicated on every rank (never divided); query side sharded; TP=2 logits ≈ TP=1; `max_tp_shards` replica semantics per A07 (S) |
| P-06.4 | Prefix cache on latent tokens | Token-granular hash chain (no head dimension); hit/miss/publish/duplicate behavior identical to P-01.5 semantics; eviction preserves ancestors (F03) (S) |
| P-06.5 | Dense + MLA hybrid (first k layers full, rest latent) | Two groups, min-boundary adoption; unequal block sizes per group allowed (F05) (S) |
| P-06.6 | Quantized latent (F09) | FP8 latent + scale plane: cached-vs-uncached tolerance bounded; scales preserved through sharing and (with A04) offload; `fp8_ds_mla`-style role of the scale plane documented (S) |
| P-06.7 | Indexer/sparse roles (`cache_role`, `is_index_group_leader`, DSA) | Handled with P-09; index group sized and evicted independently (S/R per X01) |
| P-06.8 | Sliding-window MLA / bounded replay (DSv4) | S only if P-02.12 semantics are implemented for the latent; otherwise R citing the missing mechanism (S/R) |
| P-06.9 | Hidden-state cache requests (`extract_hidden_states`) | Bypass reuse and return exact hidden states (A08), or R (S/R) |
| P-06.10 | MLA × unsupported geometry/backends | R at startup (SGLang blocks several MLA backend×feature combos [S12]) (R) |

## P-07 — Recurrent / SSM / linear attention (Mamba, GDN, KDA, RWKV, short-conv)

**Upstream surface:** `MambaSpec` — per-layer `shapes`/`dtypes` tuples, `page_size_padded`, `mamba_cache_mode` "none"/"align" (APC requires align), `num_speculative_blocks`, `num_prefill_checkpoint_blocks` + alignment, `tp_replicated` [V5]; `MambaManager` — align-mode checkpoints, skip = `num_computed − 1`, ≤ 1 new block/step plus first-allocation speculative reservation, checkpoint reservation at the prefill-end boundary, connector boundary-state offload hooks, DCP/PCP == 1 [V4]; SGLang mamba component — slot allocator, int8 checkpoint pool, CoW on match, checkpoint grid, hybrid branching at the last grid-aligned full-KV position, page_size 1 without the extra buffer, write-through-only eviction of new branching states [S3]; pool shapes: conv `(dim, K−1)` (transposed for Kimi/KDA), SSM `(HV, V, K)`, ReplaySSM power-of-two rings with fp32 gate ring, `ShortConvPool`, PLE n-gram pool [S5]; families mamba1/2, GDN, KDA (Kimi Linear, GLM-5 Next), Lightning RWKV, short-conv (LFM2/ZAYA) [S12]; separate mamba dtype/size knobs [V6]. **InferX delivery:** F08 (slot allocator, checkpoints, CoW, prefix integration), C04 (reset addressing), C05 (preemption restore), A01/A13 (spec state), A02 (branch isolation). **Today:** R — `RecurrentStatePool` and GDN ops exist but are referenced by no runner, model, or scheduler path; pure-recurrent models are rejected (`src/models/model_runner.cc:266`).

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-07.1 | C04 sentinel suite | Exact matrix-and-conv zeroing of first/middle/last slots across heterogeneous layers, incl. `conv_kernel_size == 1`; CPU and CUDA (already C04's Accept — kept here as the T04 gate) (S) |
| P-07.2 | Slot lifecycle | Independent slot pool: allocation, exhaustion (explicit failure, no silent reuse), release at finish; exhaustion of slots does not corrupt KV-pool accounting (F05/F08) (S) |
| P-07.3 | Hybrid boundary | Attention-prefix hit 300, newest recurrent checkpoint at 256 → adopt 256 and recompute 44 tokens; missing-checkpoint-despite-longer-attention-hit adopts the shorter checkpoint boundary (F08) (S) |
| P-07.4 | Checkpoint grid/alignment | Captures at the documented grid (align mode / prefill-end boundaries [V4, S3]); checkpoints evictable separately from active slots; write-back eviction unsupported → R or write-only, mirroring [S3] (S/R) |
| P-07.5 | Preemption | Exact matrix+conv state restored at the same boundary as reused KV; resumed output equals the uninterrupted reference (C05+F08) (S) |
| P-07.6 | Branch/fork (A02) | Branch copies into a fresh slot (CoW on match, SGLang analog [S3]); branches mutate independently; parent/child release in any order without slot leaks (S) |
| P-07.7 | Speculative state (A01/A13) | Draft state saved before lookahead and restored on rejection; accept none/some/all leaves state exactly at the accepted boundary (S with A13; cache-side only before) |
| P-07.8 | Family shapes | Per-layer heterogeneous shapes/dtypes — mamba2 matrix+conv, GDN delta-rule state (ops exist `src/ops/gdn.*`), KDA transposed conv, RWKV/Lightning state, short-conv ring [S5, S12] — each fixture or explicit R per family (S/R) |
| P-07.9 | Pure-recurrent model | Runner constructs no KV pool but a state pool and serves it (opens `src/models/model_runner.cc:266`); prefix caching rejected or checkpoint-backed only (F08) (S) |
| P-07.10 | `tp_replicated` state | Replicated states consume equal bytes on every rank; sharded states divide; A07 agreement validates which (S) |
| P-07.11 | Quantized recurrent checkpoints | Separate decision with quality tests (F09): R by default until then (R) |
| P-07.12 | Mamba × DCP/PCP; APC without align mode | R mirroring upstream [V4] (D/R) |

## P-08 — Cross-attention, encoder–decoder, and non-generation requests

**Upstream surface:** `CrossAttentionSpec` sized via `max_num_encoder_input_tokens`; `CrossAttentionManager` — prefix caching raises (encoder state unique per request) [V4, V5]; `EncoderOnlyAttentionSpec` (zero KV) and Whisper-style `tokens_per_state = Fraction(1, block_pool_size)` pooling [V5]; SGLang: encoder-decoder support varies by backend (triton lacks cross attention [S12]). **InferX delivery:** A08 (+ runner output support outside `cache`). **Today:** R — single decoder-style chain; logprobs requests already rejected at the server (`src/server/api.cc:125`).

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-08.1 | FX-cross: same decoder tokens, different encoder inputs | No cross-state sharing (F04 identity includes encoder input); decoder-only prefix reuse unaffected (S) |
| P-08.2 | Encoder/decoder independence | Separately sized pools and lifetimes; encoder KV freed at request completion and never published to the prefix index [V4] (S) |
| P-08.3 | Beam/multi-sample sharing encoder KV (A02) | Shared encoder blocks reference-counted; branches hold independent decoder tails (S) |
| P-08.4 | Preemption of an encoder–decoder request | Encoder state retained; decoder replay equals the uninterrupted reference (S) |
| P-08.5 | Prompt-logprobs / embedding / hidden-state requests | Reuse bypassed; outputs bit-comparable to cache-off runs (A08); until then R (S/R — today's HTTP 400 is the tested R) |
| P-08.6 | Encoder-only / pooling model (zero-KV) | Served with no KV allocation (upstream `EncoderOnlyAttentionSpec` [V5]) or R at registration (S/R) |
| P-08.7 | Whisper-style pooled state (`tokens_per_state < 1`) | Compression/pooling translation implemented with X01 or R (S/R) |

## P-09 — Sparse, selected-token, and compressed auxiliary families

**Upstream surface:** DSA/QSA/NSA sparse backends, MiniCPM and MiniMax sparse wrappers [S12]; `HiSparseHotSpec` (ephemeral per-request ring, not prefix-cacheable) and `HiSparseResidentSpec` (reclaimable GPU pages for host-backed KV) plus `HISPARSE_SOURCE`/`HISPARSE_INDEXER` group roles [V4, V5]; `tokens_per_state > 1` compression (DSv4) [V5]; `CircularBufferSpec` — one ring block per request whose block_size is the ring capacity, required to exceed the compression ratio by speculative lookahead, not prefix-cacheable [V5]. **InferX delivery:** X01 (roles, translations, quality contracts), F07 (ring retention), A04 (host tier for resident pages). **Today:** R — none of this exists; the one-entry latent pool is not sparse support (per X01's own warning).

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-09.1 | FX-sparse end-to-end | Selected-token attention over an indexer matches an exact-attention reference within a declared approximation tolerance; the tolerance itself is asserted, not open-ended (S) |
| P-09.2 | Selected tokens crossing pages; original positional indices | Token-position translation exact both directions; no aliasing between compressed and source positions (S) |
| P-09.3 | Auxiliary-pool independence | Indexer/hot pools exhaust and evict without affecting source-pool ownership or the T01 oracle invariants (S) |
| P-09.4 | Hot-set change | Source state required to recompute approximate sets is retained or restored; quality re-bounded after hot-set shifts (S) |
| P-09.5 | Ephemeral planes not published | Hot ring and circular buffers never enter the prefix index (upstream: not cacheable [V5]); verified by hash-absence, not just counters (S) |
| P-09.6 | Ring capacity validation | Ring block_size > compression ratio × speculative lookahead enforced at startup [V5] (S/R) |
| P-09.7 | Host-backed resident pages (A04) | Reclaim/restore round trip byte-exact; residency accounting correct under pressure; eviction of resident pages never loses the host copy (S after A04; R before) |
| P-09.8 | `tokens_per_state > 1` compressed states | Compression boundaries at multiples of the state period; decompressed reads match reference; tails/padding handled (S with X01; R before) |

## P-10 — Quantized KV configuration parity

**Upstream surface:** vLLM `cache_dtype` — auto/float16/bfloat16, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla, nvfp4_ds_mla, turboquant family, int4/int8/fp8_per_token_head, nvfp4, nvfp4_4over6; `kv_cache_dtype_skip_layers`; accuracy-drop warnings without proper scales [V6]. SGLang — auto (checkpoint FP8 algorithm → e4m3), fp8_e5m2/e4m3 (e5m2 rejected on CPU), mxfp8, bf16, nvfp4 (needs torch ≥ 2.8 / CUDA 12.8), draft-worker dtype override, fa4-draft `K.dtype == Q.dtype` constraint; FP8 stored as uint8 [S5, S6]. **InferX delivery:** F09 (FP8 pipeline, scales, mixed layers), F06 (format contracts). **Today:** byte-level FP8 allocation works; no quantized read/write path; execution BF16 only.

| Case | Scenario | Expected (verdict) |
| --- | --- | --- |
| P-10.1 | Dtype accept/reject matrix | FP8 e4m3/e5m2 → S via F09; every other upstream dtype string-equivalent (per-token-head int4/int8/fp8, nvfp4*, turboquant*, mxfp8, fp8_inc) → R at startup naming kernel support — no silent BF16 fallback (S/R table, ground rule 4) |
| P-10.2 | Scale contracts | Granularity (tensor/token/head), lifetime, zero/large/NaN validation; startup rejection of invalid scales [V6 warnings analog] (S) |
| P-10.3 | Cached-vs-uncached tolerance | Quantized cache on/off output delta bounded and asserted for every family that permits quantization (S) |
| P-10.4 | Scales through lifecycle | Preserved through block sharing (T01 oracle), branch copies (A02), offload/restore (A04), remote handoff (A06) (S per owning item) |
| P-10.5 | Per-layer skips/overrides (`kv_cache_dtype_skip_layers` analog) | Mixed BF16+FP8 layers in one model; per-group report (F10); unsupported mixes R (S/R) |
| P-10.6 | Quantized × mechanism | FP8 KV × {full, window, sink, CLA, MLA-latent, cross} each S or R explicitly; quantized recurrent checkpoints R by default (P-07.11) (S/R) |
| P-10.7 | Draft-cache dtype override | Different draft/target KV dtypes verified — meaningful only with A13 (S/R) |

## P-11 — Cross-cutting configuration parity

Upstream knobs mapped to InferX state; anything not mirrored must be a recorded non-goal with a reason, not an omission.

| Upstream knob | InferX status → target |
| --- | --- |
| `block_size` (vLLM default 16 [V6]); SGLang `--page-size` {1, 32, 64} | CLI {16,32,64} today → F06 validates the supported set per backend; page 1 via A03; unsupported values R at startup |
| `gpu_memory_utilization` (vLLM always-on, 0 < g ≤ 1) | Opt-in profiling today (`model_runner.cc:270`) → F11 fixes precedence (utilization > explicit bytes > block count) and in-range validation; test all three orderings |
| `num_gpu_blocks_override`; `kv_cache_memory_bytes` | `num_kv_blocks` / `kv_cache_memory_bytes` exist → F11 negative/zero/below-one-block validation and exact-byte override tests |
| `enable_prefix_caching` | Exists → on/off equivalence tests per family; A09 fixes the misleading comment |
| `prefix_caching_hash_algo` (sha256/cbor/xxhash [V6]) | Recorded non-goal: SHA-256 is both sides' default; algorithm choice is a perf knob, revisit with A11 evidence |
| `prefix_match_unit` (divisibility validated [V6]) | A03 delivers configurable match granularity incl. divisibility checks |
| `prefix_cache_retention_interval` (multiple of block size [V6]) | F07 retention-interval support or recorded deferral |
| `sliding_window` config override [V6] | F07: override validated against model layers (per-layer windows remain distinct groups, P-05.3) |
| `mamba_cache_mode`/`mamba_block_size`/`mamba_ssm_cache_dtype`/`page_size_padded`/shared-prefix checkpoint [V6] | F08: align-mode gating (APC requires align [V4]), state dtype separate from KV dtype, checkpoint sharing or R |
| `kv_offloading_size` / `kv_offloading_backend` (native/lmcache) [V6] | A04/A05: size-gated activation (offload only when configured), backend abstraction transport-agnostic |
| `kv_sharing_fast_prefill` (YOCO/Gemma-3n [V6]) | F05 aliasing + F13 fixture execution, or D recorded |
| `swa_bounded_replay` (default true, requires runner V2 [V6]) | P-02.12 verdict propagates here |
| CUDA graph sizes; graphs × TP | A12 (replay-safe tables); TP graphs stay R until explicitly lifted — test the existing error (`src/engine/model_executor.cc:55`) |
| Chunked prefill enable/size | Exists in spirit (always on, `scheduler.cc:78`); `enable_chunked_prefill=false` parsed but unenforced (`scheduler_config.h:25`) → C08 amendment: honor it (admit whole prompts) or reject the configuration |
| `max_model_len` vs capacity; admission headroom | C08: cannot-fit prompts fail explicitly; headroom/full-fit policy selected and tested |
| Eviction policy (lru default) / scheduler policies | F03 tail-first baseline; A10 optional variants compared against it |
| DCP / PCP sharded cache | A07 future → D everywhere upstream asserts == 1 [V4] |

## Gap register and dispositions

| # | Gap found while building the matrix | Disposition |
| --- | --- | --- |
| 1 | MLA execution is absent (storage + assembly op exist; no family, runner path, or TP test) — no existing item delivers it | **New item F12** |
| 2 | No executable fixture/family for window, sink, chunked-local, hybrid, or recurrent mechanisms; InferX's own kernels are unreachable and the config gates reject everything | **New item F13** |
| 3 | Speculative decoding has cache-side contracts (A01) but no engine integration (draft runner, verify loop, masks, draft groups) — `docs/speculative_decode.md` is a proposal only | **New item A13** (cache side stays A01) |
| 4 | RSWA / bounded-replay / `tokens_per_state` compression / ring-buffer rules exist upstream but appear in no Accept list verbatim | Amendments into F07 (RSWA, ring capacity) and X01 (compression/pooling, indexer roles) — the P-02.12/P-04/P-09 cases above are the acceptance tests |
| 5 | Same-type grouping rule (different windows = different types) and stride-aliasing single-owner rule [V5] not explicit in F05 | Amendment into F05 — cases P-05.3/P-05.4 |
| 6 | Pipeline parallelism: upstream sizes pools per PP rank today; A07 only names TP agreement and future CP | Amendment into A07 — per-rank layer-partition sizing validation |
| 7 | `enable_chunked_prefill=false` is parsed but never enforced | Amendment into C08 — honor or reject |
| 8 | `non_causal` (prefix-LM) reuse semantics undefined | Amendment into F04 — define or reject; case P-01.12 |
| 9 | Encoder-only/pooling and hidden-state-cache verdicts undefined | Amendment into A08 — cases P-08.6/P-08.7/P-06.9 |
| 10 | Draft-worker KV dtype override [S6] meaningless without engine spec decode | Attach to A13 — case P-10.7 |
| 11 | Hash-algorithm configurability, backend-name parity, non-LRU eviction | Recorded non-goals with reasons (perf knobs / kernel-agnostic scope / A10 evidence first) |
| 12 | Upstream-restricted combinations (DCP/PCP asserts, Mamba APC align-only, MLA backend allowlists, Blackwell GDN allowlist, e5m2-on-CPU) | Mirrored R/D with citation — never supported by analogy |

## New backlog items

### F12 — Wire MLA execution end to end

- [ ] The cache layer can store a one-entry latent (`KvLayout.entries_per_token = 1`) and `AssembleMlaCaches` exists (`src/ops/mla.cc`), but no model family, runner path, or TP test reaches them; "complete MLA execution" is called absent in this backlog's own review table.
- **Change:** Register an MLA family or the FX-mla fixture; write the latent through `KeyCache()` (one entry per token, no V plane); implement absorbed decode or route through decompressed assembly; handle the RoPE plane as a first-class F06 descriptor; TP replicates the latent (never divides `kv_heads`) while sharding queries; startup-reject unsupported latent geometries and kernels. Quantized latent lands with F09.
- **Accept:** P-06.1–P-06.6, P-06.10 green (FP32-reference tolerance; TP replication with equal per-rank latent bytes; token-granular prefix cache on latents; dense+MLA hybrid boundary; FP8 latent tolerance when F09 lands).
- **Dependencies:** F05, F06; F09 for quantization. **References:** `MLAAttentionSpec` [V5]; MLA backends and allowlists [S12].

### F13 — Ship executable fixture families and open the attention config gates

- [ ] Every mechanism item above (F05–F08) is verifiable only through a model the runner will actually execute. Today window/sink/CLA/hybrid/recurrent checkpoints are rejected at config (`src/models/qwen3/qwen3_config.cc:113`, `src/models/llama/llama_config.cc:52`, unregistered `gpt_oss`), leaving InferX's own window/sink kernels, GDN ops, and recurrent pool unreachable by any test.
- **Change:** Land the fixture table (FX-swa, FX-sink, FX-cla, FX-hyb, FX-mamba, FX-gdn; production anchor `models/gpt-oss-20b` for sink+window) with tiny BF16 checkpoints and FP32 references, wired through the real scheduler/runner; open each config gate as its delivering item lands (F07 for window/sink/CLA, F08 for mamba/GDN) and keep an explicit tested rejection otherwise.
- **Accept:** Each fixture passes prefill+decode golden/FP32 gates and its T02 cache-on/off comparison; previously-rejecting configs now serve or fail with the new explicit error; registry/config tests updated (`tests/unit/models/registry_test.cc:31` superseded by a real entry or a tested rejection).
- **Dependencies:** F07, F08 per family; F12 for FX-mla. Registry/config work sits outside `cache`.

### A13 — End-to-end speculative decoding integration

- [ ] A01 delivers tentative allocation, commit, and rollback, but there is no draft runner, verify/accept loop, or mask plumbing (`docs/speculative_decode.md` is a proposal). Upstream carries draft groups explicitly (`is_eagle_group`, EAGLE hash-drop [V4, V5]), draft-worker KV dtype overrides [S6], and `extra_retained_tokens` for multi-module re-prefill [V5].
- **Change:** Engine-level draft/verify/accept loop (chain first, tree later) over every fixture family; draft-target cache groups with independent layouts/dtypes; replay-safe masks consistent with A12 graphs; recurrent-state save/restore on rejection (F08).
- **Accept:** Accept none/some/all across full, window, sink, CLA, hybrid, and recurrent fixtures with output equal to the non-speculative greedy reference; draft-only hashes never published; no leaks or hash-chain drift across repeated rollback (extends A01's Accept); graphs on; TP on.
- **Dependencies:** A01, F02, A12; F08 for recurrent families.

## Gate G5 (parity) and maintenance

- **G5:** the matrix is fully executed — every P-case above has a recorded S/R/D verdict backed by a merged test or a tested rejection; the fixture families exist; gap-register items 1–3 (F12/F13/A13) are implemented and items 4–10 amendments are folded into their owning items' acceptance lists.
- **Maintenance:** when the pinned upstream snapshots move, re-derive the mechanism list from `kv_cache_interface.py` [V5], the per-type managers [V4], the SGLang pool families [S5, S7], and the backend registry [S12], and extend this matrix before claiming parity for new mechanisms. A mechanism absent from this matrix is unsupported by definition.
