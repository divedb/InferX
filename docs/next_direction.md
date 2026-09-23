# Next exploration: speculative decoding, MoE, or tensor parallelism

Research date: 2026-09-23. Status: recommendation only; no code changed.

**Recommendation: speculative decoding next, in three stages — with one prerequisite
honesty check about the KV plugin (it is not yet in the serving loop, so "finding its
limits" first requires wiring it in or building in-engine prefix caching as the
measurement baseline).** MoE is second (start from tiny random checkpoints; defer the
MXFP4 kernel strategy). Tensor parallelism stays last: this box has one GPU and the
plan already marks TP as requiring a target rig (plan.md §0.2, M6).

The recommendation is consistent with the plan the repo already carries: spec decode
is milestone M5 with the D10 staging (n-gram first, `Proposer` interface later), and
D23 makes MTP a proposer on the same lookahead/verify/rollback machinery — machinery
that two of the three P0 target models (Qwen3.8-Flash-Next, DeepSeek-V4.1-Flash)
require regardless. Building it next serves both the KV-research goal and the P0
model plan.

## What InferX has today (audited for this decision)

| Capability | State | Spec-decode consequence |
|---|---|---|
| Scheduler | Unified token budget + chunked prefill; **no preemption** (pressure skips, not preempts) | Lookahead reservation and rejection rollback are net-new scheduler states |
| KV block pool | Plain LIFO free list; `BlockTable` Append/Clear only, no truncation, no refcount/sharing, **no prefix caching in the engine loop** | Rollback needs tail-block reclaim + computed-watermark reset; alloc/free churn rises ~k× per step |
| kvc provider ABI | SDK 0.1 shipped (host + tiered CUDA providers) but **not called from the serving loop** (kvcache_plan.md §27 use-cases 25–27 "engine-side"; §28 lists the missing identity helper) | Measuring "where the plugin's benefits diminish" requires first wiring it (or an in-engine prefix cache as baseline) |
| CUDA graphs | Captured per exact pure-decode batch size | Verify steps process k tokens/seq with variable accepted length → pad to fixed k, capture spec-shape graphs, or fall back to eager |
| Sampler | Greedy fast path only | Greedy verification is trivial (draft token == target argmax); stochastic rejection sampling comes later |
| Attention | FlashInfer split-KV decode wrapper (page-size coupled) | Verification needs multi-query rows per seq (qo_indptr > 1) or the prefill path with per-chunk causal masks; tree masks are a later step |
| Models | Dense llama + qwen3 builders; local checkpoints: Qwen3-0.6B (bf16), Qwen3-8B-AWQ, gpt-oss-20b (MoE/MXFP4), Qwen3-0.6B-GPTQ | Draft-model pairing is locally feasible (0.6B + 8B-AWQ share a tokenizer, both fit in 16 GB with KV) |

## Upstream state (what the maintained paths are)

- **vLLM v1** supports n-gram, EAGLE/EAGLE-3, MTP, Medusa-class proposers — but
  **deprecated standalone draft-model speculative decoding** in v1 (two-model serving
  complexity; community work to restore it is ongoing). See
  [vLLM speculative decoding docs](https://docs.vllm.ai).
- **EAGLE-3 is the de-facto production standard** (2–4× claims; ~40% observed draft
  acceptance in one production report; top-k 8–12 typical). See
  [Spheron's EAGLE-3 writeup](https://spheron.network),
  [HF production notes](https://huggingface.co), and
  [SpecForge (lmsys)](https://lmsys.org).
- **SGLang** ships EAGLE-2/3, MTP, NGram, and classic draft decoding
  ([docs.sglang.ai](https://docs.sglang.ai)); multi-process "parallel speculative
  decoding" (draft and target in separate processes) is an active 2026 roadmap theme
  ([vLLM roadmap #27462](https://github.com)).
- **gpt-oss on consumer Ada GPUs is painful even for vLLM**: MXFP4 requires
  Triton ≥ 3.4 + `triton_kernels`, otherwise it dequantizes to BF16 (~26 GB+ — does
  not fit 16 GB), and the Marlin MoE fallback has open MXFP4 bugs; official support
  targets Blackwell/Hopper. See the
  [gpt-oss-20b HuggingFace discussion](https://huggingface.co),
  [vLLM gpt-oss recipes](https://docs.vllm.ai), and
  [Modal's guide](https://modal.com).

## The user's hypothesis, examined: does spec decode reveal KV-plugin limits?

Yes — with a sharper statement of the mechanism. Speculative decoding is a
memory-bandwidth optimization, and its benefit tracks whichever memory dominates:

| Regime | Bottleneck | Spec-decode effect | KV-plugin relevance |
|---|---|---|---|
| Small batch, short context | Weight reads per token | Largest wins (1.5–3×) | Low — KV is a small fraction of traffic |
| Large batch, short context | Compute | Wins shrink or invert (verify work is not free) | Low |
| Small/large batch, **long context** | **KV reads per token** | Wins persist | **High — KV placement/bandwidth dominates** |

Sources: [philkrav.com](https://philkrav.com),
[Redis on speculative decoding](https://redis.com),
[Together AI on long-context high-throughput spec decoding](https://www.together.ai),
[Batch Speculative Decoding Done Right (arXiv 2025)](https://arxiv.org).

Spec decoding also stresses KV bookkeeping directly, in ways the current code has
never seen:

1. **Rollback.** Rejected speculative tokens must free/invalidated KV tails every
   step. `BlockTable` cannot truncate; the pool has no per-block valid-token counts.
   Block-size choice becomes measurable: large blocks waste reclaim on partial
   rollback, small blocks raise metadata cost at k× churn.
2. **Allocation churn.** k× faster token production means k× block-append rate and
   earlier capacity pressure — which finally forces the preemption decision the
   scheduler currently skips.
3. **Verify-batch attention shapes.** k queries per sequence per step change the
   read pattern over paged KV and interact with the FlashInfer plan/workspace
   sizing and CUDA-graph shape set.
4. **Prefix-cache coupling.** n-gram/prompt-lookup proposals come from the prompt;
   their acceptance (and thus speedup) correlates with prefix-cache hits — the
   plugin's home turf.

So spec decoding is the best available *measurement vehicle* for KV-plugin
decisions, but the instrument is not plugged in: the honest prerequisite is either
(a) minimal in-engine hash-block prefix caching (fast, gives a baseline), or (b)
wiring the kvc provider path into the serving loop (needed eventually per
kvcache_plan use-cases 25–27). Do (a) as part of stage S1, (b) as part of S2.

## Proposed stages

- **S0 (baseline, inside S1):** in-engine hash-block prefix caching over the
  existing `KvBlockPool` (refcounted blocks; scheduler consults it on admission).
  This is also plan.md D5, currently missing.
- **S1 (greedy n-gram speculation, no new weights):** scheduler lookahead
  reservation, draft-token injection, verify batching (fixed k, padded), greedy
  verification, rollback accounting + KV tail reclaim; CUDA-graph policy for spec
  shapes (pad-or-eager). Success = token-for-token greedy parity plus the M5
  criterion direction (≥1.5× tokens/s on repetitive workloads, default-off until
  measured).
- **S2 (the KV experiments the user wants):** with spec decode driving load,
  measure: alloc/free churn and metadata cost vs block size (16/32/64); rollback
  reclaim efficiency; capacity-pressure onset vs concurrency; ITL/tokens-per-step
  vs batch size at short and long context (expect the crossover above); provider
  on/off (host-tier CUDA provider) in the KV-bound regime. Output: a documented
  "where KV-provider benefits appear/vanish" matrix.
- **S3 (later, model-attached):** EAGLE-3-style or MTP proposers when the P0
  target models arrive (D23 reuses S1 machinery). Optional side experiment while
  hardware allows: 0.6B draft → 8B-AWQ target pairing, noting upstream deprecated
  standalone draft-model serving in v1 for complexity reasons.

## Why not MoE or TP first

- **MoE** is P0-critical per plan.md (both Flash targets) and the local gpt-oss-20b
  is tempting, but on this box it is unrunnable at full size (MXFP4 → BF16 dequant
  exceeds 16 GB; kernel paths for MXFP4 on Ada are immature upstream), so local work
  would be tiny-random-checkpoint development — valuable, but with no local
  end-to-end payoff until the kernel strategy and target rig exist. It also barely
  exercises the KV plugin beyond sliding-window retention (kvc specs already
  "designed" for it).
- **Tensor parallelism** has nothing to measure on one RTX 4080 SUPER without
  NVLink; plan.md M6 explicitly gates it on a target rig. The interfaces are
  already TP-aware (`KvLayout::kv_heads` is documented as per-rank; kvc use-case 21
  covers per-rank providers), so deferral costs nothing.

## Validation checklist (before calling any stage done)

- Greedy token-for-token parity vs non-speculative path across acceptance rates
  (force k=1 and adversarial low-acceptance prompts).
- Rollback correctness under KV pressure (no leaked blocks; watermark consistency).
- CUDA-graph replay equivalence for padded verify shapes.
- Concurrent spec + non-spec requests mixed in one batch.
- The S2 measurement matrix reproduced twice on the same box with bounded variance.
