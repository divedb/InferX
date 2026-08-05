# Serving gpt-oss-20b

A plan for making `inferx-serve` run `openai/gpt-oss-20b`, the MoE checkpoint
already cached on this box, and for what that buys.

**Status: Phases 0–2 done. Phases 3–4 proposed.** The oracle exists, the MXFP4 bit
layout is settled against HuggingFace's own decoder, and every definition this
document originally marked "recalled, verify later" has been read out of the
reference implementation. §8 records what changed. The rest exists to be argued
with before it is written.

The short case for doing it: M9 built MoE and MLA layers that are tested
against their own definitions and nothing else, because no MoE checkpoint fit
the 16 GB envelope. gpt-oss-20b fits — 13.76 GB on disk — *because* its expert
weights are MXFP4. So it is simultaneously the thing that would turn M9's
"self-consistent" into "matches a reference", and the thing that forces a
quantization format we do not implement. Those are the two halves of the
trade, and the second one is bigger than it looks.

The short case against: this is not one milestone. It is a new weight format, a
new attention variant, a new positional encoding, a new activation, a new
tokenizer pre-tokenizer, and a KV cache whose lifetime differs per layer. Each
is individually small. Together they are M11, not "M9 integration", and
pretending otherwise is how a milestone runs three times over.

---

## 1. What the checkpoint actually is

Read from
`~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81…/`,
not from memory.

| | |
|---|---|
| architecture | `GptOssForCausalLM`, 24 layers, hidden 2880 |
| attention | 64 Q heads, 8 KV heads (GQA 8:1), `head_dim` 64 → `q_dim` 4096 ≠ hidden |
| attention extras | per-head learned **sinks** (`self_attn.sinks [64]`), Q/K/V/O **biases** |
| layer types | alternating `sliding_attention` / `full_attention`, `sliding_window: 128` |
| positions | **YaRN**, factor 32, `beta_fast` 32, `beta_slow` 1, orig 4096 → 131072, `rope_theta` 150000 |
| FFN | MoE: 32 experts, top-4, `intermediate_size` 2880, **no shared expert** |
| expert weights | **MXFP4**: `*_blocks` u8 `[32, N, 90, 16]` + `*_scales` u8 `[32, N, 90]`, plus bf16 `*_bias` |
| not quantized | attention, router, `embed_tokens`, `lm_head` (per `modules_to_not_convert`) |
| activation | `hidden_act: silu` with `swiglu_limit: 7.0` |
| vocab | 201088, `tie_word_embeddings: **false**` |
| tokenizer | BPE, o200k-style Split regex, 199998 vocab + 21 added |

### Where the bytes go, and why that matters

| | bf16 | as stored |
|---|---|---|
| expert weights (24 × 32 × [5760 + 2880] × 2880) | ~38 GB | **10.1 GB** MXFP4 |
| attention + norms (24 layers) | 1.27 GB | 1.27 GB |
| `embed_tokens` + `lm_head`, untied | 2.32 GB | 2.32 GB |
| **total** | | **13.76 GB** ✓ matches disk |

**This is the single most important number in the document.** On a 16376 MiB
card, 13.76 GB of weights leaves roughly **2 GB** for the KV cache, activations,
cuBLASLt workspace and logits. Three consequences follow immediately:

- **Dequantizing MXFP4 to bf16 is not an option at any point in the serving
  path.** 38 GB does not fit, is not close to fitting, and no amount of
  streaming makes a resident weight non-resident.
- **KV is cheap here and is not the constraint** — a pleasant surprise. 8 KV
  heads × 64 = 2048 B/token/layer; only 12 layers are full-attention, and the
  12 sliding ones need 128 tokens each forever. That is ~24.6 KB/token, so a
  4096-token context costs ~100 MB. Contrast Qwen2.5-3B's 36 KB/token.
- **The logits buffer is a real problem.** 201088 vocab × fp32 is 804 KB *per
  position*. A 2048-token chunked prefill that materializes all rows wants
  1.6 GB — most of the headroom. The engine already only needs
  `logits_indices` rows; this has to be verified rather than assumed for a
  vocab this wide.

The cheapest lever if it turns out too tight: `lm_head` is **untied** and is
1.16 GB of bf16 on its own. FP8 halves it. That is a knob we already have.

---

## 2. What already fits, and it is more than expected

M9's `MoeFfn` was written against Qwen2-MoE's shape and it lands on gpt-oss
almost exactly:

- Expert weights are **stacked** in the checkpoint — `gate_up_proj_blocks
  [32, 5760, 2880]`, `down_proj_blocks [32, 2880, 2880]` — which is precisely
  the `[E, 2·inter, hidden]` / `[E, hidden, inter]` layout `MoeWeights`
  specifies. No transposition, no per-expert reassembly at load.
- `router.weight [32, 2880]` is exactly `MoeWeights::router`.
- **The routing rule already matches**, now confirmed against
  `GptOssTopKRouter.forward`: it computes `topk(logits, 4)` then `softmax` over
  those four. `MoeRouteTopK` softmaxes over all 32, takes top-4, then
  renormalizes when `norm_topk_prob` — algebraically identical, both being
  `e^{z_i} / Σ_{top4} e^{z_j}`. So the M9 router is gpt-oss's router with the
  flag already on. (Its bias is the one missing piece; see R-E.)
- Top-4 of 32 with no shared expert is the simplest configuration `MoeFfn`
  supports.
- The config parser's decoupled `head_dim` already handles `q_dim` 4096 against
  hidden 2880, which is not a shape the dense Qwen path ever produced.

This is not luck so much as the stacked layout being the obvious one; but it
does mean the MoE *plumbing* is done and what remains is arithmetic.

---

## 3. The work register

Ordered by risk, not by size. Sizes are gut estimates and should be treated as
such.

### R-A. MXFP4 — mandatory, on the critical path

The format: 32 elements share one **E8M0** scale (a bare power-of-two exponent,
bias 127, one byte); each element is **FP4 E2M1** (sign, 2 exponent bits,
1 mantissa bit), representable values `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
Decode is `value = 2^(scale − 127) × lut[nibble]` — no multiply-by-float in the
scale, which makes it *cheaper* to dequantize than our existing int4.

Storage confirmed from disk: `[…, 90, 16]` u8 = 90 blocks × 16 bytes = 90 × 32
nibbles, and 90 × 32 = 2880 = hidden. ✓

How it relates to what we have: `w4a16_gemm.cu` implements per-group-**128**
symmetric int4 with a bf16 scale. MXFP4 is group-**32** with an exponent-only
scale. The group size is a template constant and the scale application becomes a
shift rather than a multiply, so this is an *adaptation* of an existing kernel
rather than a new one — but it is an adaptation of the kernel that commit
`691e7e5` says is **still behind cuBLASLt bf16**. Building the flagship model on
it means either accepting that or fixing it first.

**Settled in Phase 0**, bitwise, against `transformers.integrations.mxfp4` — see
`scripts/gen_mxfp4_golden.py`, which is the executable version of this
paragraph:

- **Low nibble first.** Byte `i` holds element `2i` in its low four bits and
  element `2i+1` in its high four.
- Scale is E8M0 with bias 127; the multiplier is `2^(scale − 127)`, applied by
  `ldexp`. No mantissa, so dequantization is a shift.
- LUT is `[0, ±.5, ±1, ±1.5, ±2, ±3, ±4, ±6]` in sign-magnitude nibble order.
- No additional scaling anywhere.

Three cases — layer 0 `gate_up` expert 0, layer 0 `down` expert 0, layer 23
`gate_up` expert 31 — decode **exactly** (`max |Δ| = 0.0`) against HF, so the
tolerance in the C++ test can be zero rather than a guess.

**And a layout finding worth more than it looks.** HF's decoder returns
`[E, hidden, 2·inter]`, because its module computes `x @ W`. The checkpoint's
own axis order is `[E, 2·inter, hidden]` — which is *exactly*
`MoeWeights::gate_up`, since `LinearBF16` computes `y = x·Wᵀ`. **The loader
needs no transpose.** Anyone reading HF's shapes and matching them would insert
one and be wrong.

### R-B. Attention sinks

**Verified in Phase 0** against `GptOssAttention` / `eager_attention_forward`.
One learned logit per head is *concatenated* to that row's scores, the softmax
runs over `[scores | sink]`, and the sink column is then dropped:

```
combined = cat([scores, sink_h], dim=-1) - max(...)
probs    = softmax(combined)
out      = probs[..., :-1] @ V        # sink contributes to the denominator only
```

So the denominator gains `exp(sink_h − max)` and the numerator gains nothing —
a head can attend "nowhere" and emit near-zero. Mechanically trivial; the
problem is entirely where it has to live.

**Resolved in Phase 2, and it needs no FlashInfer change at all.**

The spike first: `prefill.cuh` *does* have a hook — it calls
`variant.update_m_d(...)`, which an `AttentionVariant` can override. `decode.cuh`
does **not**; it maintains `st.m` / `st.d` inline with no variant call. So the
paged decode kernel we actually run cannot express a sink, and X1 looked real.

It is not, because the sink factors out of the softmax exactly. With
`D = Σ_j e^{s_j}` over the real keys:

    out_sink = (Σ_j e^{s_j} v_j) / (D + e^{sink})
             = out_plain · D / (D + e^{sink})
             = out_plain · σ(lse − sink)

`lse` is the log-sum-exp the kernel already computes and `BatchDecodeParams`
already has a field for — we pass `nullptr` today purely because nothing wanted
it. So a sink is **a scalar rescale per (token, head) after an ordinary
attention**, and the kernel underneath stays untouched, unpatched and fast.

Verified to 2e-16 against the concatenate-and-drop form for sinks from −5 to
+20, then implemented as `ApplyAttentionSinks` and tested against the same
definition on device.

One trap worth recording: `state_t::get_lse()` returns `m + log2(d)` — **base
two**, because the kernels fold `log2(e)` into the softmax scale so they can use
`exp2`. Using it as a natural log puts a factor of `ln 2` inside a sigmoid,
which is wrong by a plausible-looking amount rather than an obvious one. The
kernel takes the convention as an explicit argument for that reason.

### R-C. Sliding-window attention — the sneaky one

`sliding_window: 128` on 12 of 24 layers. This is *not* a kernel flag. It means
**KV lifetime differs per layer**, and `KvBlockPool`, `BlockTable`, the block
manager and the radix prefix cache all assume every layer caches every token
(§6.2). A sliding layer needs 128 tokens of KV and never more.

Done properly it is a large memory win and a genuine capability. Done as an
afterthought it corrupts the prefix cache, because a cached prefix means
something different for a sliding layer than for a full one. **Treat this as a
scheduler design change, not an attention change.**

### R-D. YaRN positional scaling

NTK-by-parts interpolation with `beta_fast`/`beta_slow` ramps, factor 32,
`truncate: false`. Our `RotaryEmbedding` computes `inv_freq` inline in the
kernel from `theta`. YaRN makes `inv_freq` a per-dimension table plus an
attention-temperature term, which is better precomputed at load into a
`[head_dim/2]` buffer than recomputed per launch — a small refactor that also
helps the existing path.

**Answered in Phase 0: yes, mscale applies.** The config sets no
`attention_factor`, `mscale` or `mscale_all_dim`, so
`_compute_yarn_parameters` falls to `attention_factor = 0.1·ln(factor) + 1`,
which at `factor = 32` is **≈ 1.34657**. `cos`/`sin` are scaled by it. Missing
this is not a subtle degradation — it is a constant multiplier on every rotated
component. `truncate: false` also changes `find_correction_range`, so the ramp
bounds are the untruncated ones.

### R-E. The activation is not our SwiGLU

**Verified in Phase 0**, and it is different from `silu(gate) · up` in four
independent ways, every one of which is a silent-wrongness bug if guessed:

```python
gate, up = gate_up[..., ::2], gate_up[..., 1::2]   # INTERLEAVED, not split-half
gate = gate.clamp(max=limit)                       # limit = 7.0
up   = up.clamp(min=-limit, max=limit)
glu  = gate * sigmoid(gate * alpha)                # alpha = 1.702, not plain SiLU
out  = (up + 1) * glu                              # note the +1
```

**The interleaving is the structural one.** Our `SiluMulFused` reads
`[gate_half | up_half]` out of one buffer, and gpt-oss alternates them along the
5760 axis. Two ways out: de-interleave the `2·inter` axis once at load — a
permutation of the stacked weight, free thereafter — or write a kernel that
reads stride-2. The load-time permutation is clearly right, since it makes the
weight layout match every other model and costs nothing at run time. Either way
the *activation function itself* still needs a new kernel, because clamp, alpha
and `(up + 1)` are not parameters `SiluMulFused` has.

The expert projections also carry **biases** (`gate_up_proj_bias`,
`down_proj_bias`), and so does the **router** (`F.linear(x, weight, bias)`).
`MoeFfn` has no bias path in either place. All small, all mandatory.

### R-F. The tokenizer will reject this checkpoint outright

Verified: `src/tokenizer/tokenizer.cc` hardcodes Qwen's Split pattern in
`kExpectedSplitPattern` and returns `Unimplemented` for anything else — which is
good design and exactly what will happen here. gpt-oss uses the o200k pattern,
which differs materially: `\p{N}{1,3}` (digit runs capped at three), explicit
`\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}` case classes, `[\r\n/]*`.

So this is a second pre-tokenizer implementation plus a second conformance
corpus generated from HF, held to the same bar M4 set: exact id equality over a
few hundred strings. Self-contained, testable with no GPU, and easy to
underestimate.

### R-G. Harmony chat template

gpt-oss uses the "harmony" response format with channels (analysis / final) and
a much more elaborate template than Qwen's. `chat_template.jinja` is in the
snapshot. Our `chat_template.cc` handles Qwen's shape. Scope this as "enough of
it to serve `/v1/chat/completions` correctly", and do not attempt full harmony
semantics in the first pass.

---

## 4. What it would buy

Back-of-envelope, batch 1, the same weight-bandwidth argument §14 uses
everywhere else:

| | per token |
|---|---|
| active expert weights (4 of 32, MXFP4) | 4 × 24.9M params × 0.5 B × 24 layers ≈ **1.20 GB** |
| attention weights (bf16, all layers) | 26.6M params × 2 B × 24 ≈ **1.27 GB** |
| total moved per token | **≈ 2.5 GB** |
| floor at 736 GB/s | **≈ 3.4 ms → ~290 tok/s** |

Against Qwen2.5-3B's measured 92.5 tok/s bf16 and 131 tok/s FP8, that is the
MoE payoff **demonstrated rather than asserted** — a 21B-parameter model
decoding faster than a 3B dense one, on the same card, because only 3.6B
parameters are active. That is the strongest reason to do this, and it is a
better headline than anything else reachable on this hardware.

Note what the table also says: attention is **half** the per-token traffic,
because it is bf16 while the experts are 4-bit. FP8 attention weights would take
~2.5 GB → ~1.9 GB and the floor to ~2.6 ms. Worth knowing before optimizing the
MoE path further than it needs.

---

## 5. Phases, with exit criteria

Each phase ends in something checkable. No phase depends on a later one being
right.

### Phase 0 — establish the oracle ✅ *done*

The reference is **transformers 5.14 running gpt-oss natively in MXFP4 on the
GPU**, not vLLM and not CPU. CPU was never viable — dequantized bf16 is 38 GB
against 26 GB of usable host RAM — and transformers' own MXFP4 path fits in
**13.78 GB resident**, which is within 0.02 GB of what §1 predicted from the
tensor shapes. That agreement is itself a small validation of the memory model.

Getting there needed three packages in `.venv-vllm`, all installed
`--no-deps` so the vLLM install used by M10's benchmark is untouched:

```
pip install --no-deps 'kernels>=0.15.2,<0.16.0' kernels_data accelerate
```

The version window is not advisory. transformers 5.14 pins
`0.15.2 <= kernels < 0.16.0`; outside it, `is_kernels_available()` returns false
and transformers **silently falls back to dequantizing to bf16**, printing one
warning that scrolls past inside a progress bar before dying on memory much
later, looking like something else entirely. `gen_gptoss_logits.py` asserts the
window up front for exactly that reason. (`kernels_data` is a real dependency of
`kernels` that `--no-deps` skips; 0.16.0 is broken without it in a way that
looks identical.)

**Artifacts**

| | |
|---|---|
| `scripts/gen_gptoss_logits.py` | HF logits → `testdata/gptoss_20b_logits_bf16.bin`, same `IXRL` format as `gen_reference_logits.py`, so one C++ loader reads both |
| `scripts/gen_mxfp4_golden.py` | numpy MXFP4 decoder, cross-checked bitwise against HF, → `testdata/gptoss_mxfp4_golden.bin` |

```bash
CKPT=~/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee
.venv-vllm/bin/python scripts/gen_mxfp4_golden.py  "$CKPT" testdata/gptoss_mxfp4_golden.bin
.venv-vllm/bin/python scripts/gen_gptoss_logits.py "$CKPT" testdata/gptoss_20b_logits_bf16.bin
```

Both land in `testdata/`, which is gitignored — same arrangement as the Qwen
reference logits, and the same consequence: a fresh checkout regenerates them or
skips the tests that need them.

**Sanity**: the reference continues `"The capital of France is"` with `" Paris"`
(token 12650), from ids `[976, 9029, 328, 10128, 382]`.

**One caveat on the tolerance this oracle can support.** For Qwen2.5 the
reference is fp32 and the implementation bf16, so any disagreement is ours. Here
the reference is bf16 *and reads the same 4-bit weights*, so it is not more
precise than what it checks. Argmax agreement at every position is the property
Phase 2 should gate on; elementwise closeness is weaker evidence here than it is
for the dense path, and the C++ test should say so.

Note vLLM was **not** used, though `GptOssForCausalLM` is in its registry — it
needs `triton_kernels`, which is not importable in that venv despite the HF
cache entry. transformers got there first and one oracle is enough; vLLM
remains available as a second opinion if Phase 2 finds a disagreement it cannot
explain.

### Phase 1 — MXFP4 decode, in isolation ✅ *done*

Phase 0 had done the risky half; what remained was mechanical, and was.

- `kernels/mxfp4.{h,cu}`: `DequantizeMxfp4ToBf16`, plus
  `DequantizeMxfp4GateUpToBf16` which folds R-E's de-interleave into the same
  pass — the kernel picks a destination row regardless, so picking a permuted
  one is free.
- `tests/kernel/mxfp4_test.cc` reads the golden container and asserts **exact**
  equality. All three cases decode bit-for-bit.
- **Loader support needed nothing.** `safetensors.cc` already maps `U8` and is
  rank-agnostic, so the `_blocks` / `_scales` / `_bias` triples read as they
  are. Confirmed by loading them rather than by reading the code.

**Exit met, and then some.** The suite is six tests, and two of them are worth
more than the headline:

- `LoadsAndDecodesStraightFromTheCheckpoint` runs the whole chain — checkpoint
  on disk, our reader, our kernel — and compares against the golden bytes,
  which were produced without our reader ever being involved. A mis-strided
  rank-4 u8 tensor fails here while the isolated decode passes.
- The nibble order was checked for *teeth*: swapping it deliberately fails
  **83,388 of 92,160 values**, so the test cannot pass by accident. A test
  pinning a convention is worth only what it costs to violate.

`EveryDecodedValueSurvivesBf16Exactly` also promotes mxfp4.h's exactness claim
from a comment to an assertion, which is what entitles the other tests to
demand equality instead of a tolerance.

### Phase 2 — a deliberately slow, correct forward ✅ *done*

`model::GptOssModel` runs the whole stack and its logits match HuggingFace's.
**M9's central caveat is retired**: the MoE layer is no longer tested only
against its own definition.

The memory strategy is the load-bearing part. Expert weights stay **packed on
the host** and one layer is uploaded and dequantized into a reusable bf16
scratch as it is reached — peak ~5.5 GB on device, ~10 GB over PCIe per call,
about 24 seconds per forward. (The plan previously said 423 MB per layer; that
is the *packed* size. Dequantized it is **1.59 GB**, which is what the scratch
holds and why the peak is what it is.) That is not a serving path and is not
trying to be — Phase 4's mainloop replaces it, and this class stays as the
thing that says the fast path is right.

Reuses `MoeFfn` unchanged except for two additions gpt-oss forced, both of
which any biased MoE would have needed: expert and router **biases**, and an
**activation selector** for the clamped form.

**Exit met**, on two prompts:

| | |
|---|---|
| 5-token prompt | argmax agrees at **every** position; top-5 sets overlap 5/5 |
| 225-token prompt | 13 of 225 positions differ — see below; exercises the 128-token window on 12 of 24 layers |

The bug it caught is the one worth recording. The first run picked the wrong
token at *every* position, and the cause was that **`gate_up`'s bias must be
de-interleaved exactly as its weight was**. R-E had established that gpt-oss
interleaves gate and up, and the weight de-interleave went into the MXFP4
decode where it belongs — but the bias is indexed by the same axis and was
uploaded in checkpoint order, so every gate's bias landed on an `up` and vice
versa. It produced a model that ran, and generated fluent text, and was wrong.
No unit test would have caught it; only the end-to-end comparison did.

### Phase 3 — the tokenizer and the server path *(medium)*

R-F and R-G. Conformance corpus, exact id equality, chat template.

- **Exit:** `inferx-serve --model <gpt-oss>` answers `/v1/completions` with
  text that matches vLLM's for a greedy prompt.

### Phase 4 — make it fast *(large, and genuinely open)*

- MXFP4 in the GEMM mainloop, which is also the moment to find out why
  `w4a16_gemm` is behind cuBLASLt bf16.
- Sliding-window KV (R-C) as a scheduler change, with the prefix cache made
  aware of per-layer lifetime.
- CUDA graph capture, which the MoE dispatch's host round-trip currently
  prevents (§7.3) — this is where a grouped GEMM stops being optional.
- **Exit:** a `bench_serve.sh` row against vLLM on the same model, and an
  honest comparison against the ~290 tok/s estimate above.

---

## 6. Risks worth stating before starting

| | risk | severity |
|---|---|---|
| X1 | ~~FlashInfer's pinned version may not support attention sinks~~ | **Retired in Phase 2.** It does not, in the decode path — and it does not need to. The sink factors out of the softmax as `out · σ(lse − sink)`, so it is a post-pass rescale using the lse the kernel already computes. No patch, no fallback, no lost speedup |
| X2 | 2 GB of headroom after weights; the 201088-wide logits buffer can eat most of it in one prefill | Medium — mitigable, `lm_head` to FP8 buys 1.16 GB |
| X3 | Phase 4 rests on a fused 4-bit GEMM that today loses to bf16; the MoE win could evaporate into kernel work | **High** — Phase 2 delivers value regardless, which is why it is sequenced first |
| X4 | Sliding-window KV touches the prefix cache and block manager, the two components with the most invariants | Medium — contained if treated as a scheduler change |
| X5 | ~~Several definitions here (activation shape, nibble order, YaRN mscale) are recalled rather than verified~~ | **Retired in Phase 0.** All three read out of the reference; MXFP4 pinned bitwise by a golden file. The activation turned out to be interleaved, which nothing would have caught later except a wrong model |

---

## 7. Recommendation

Phases 0–2 are done. Phase 2's exit criterion is met: gpt-oss-20b runs
end to end and its logits match HuggingFace's. Phases 3 (tokenizer and
serving) and 4 (make it fast) are the remaining work, and §9 revises what they
look like now that the model runs.

That subset is where nearly all the value sits: it retires M9's caveat, it
proves MXFP4, and it produces a checkpoint-validated MoE forward pass — while
being the part *least* exposed to X1 and X3, the two risks that could stall the
effort indefinitely. Phases 3–4 turn it into a serving story, and are worth
deciding on with Phase 2's findings in hand rather than now.

Call it **M11**. M9 stays what it is: layers, tested against their definitions.
Folding this into M9 retroactively would make a finished milestone
open-ended, which is the bookkeeping equivalent of what §14 keeps warning about.

---

## 8. What Phase 0 changed about this document

Recorded because the point of a plan is to be wrong in public and get corrected.

**Everything marked "recalled, verify later" was recalled correctly**, which is
less reassuring than it sounds — three of the four had a detail that would not
have survived first contact:

- The activation is **interleaved**, not split-half. That is a weight-layout
  consequence, not an activation-kernel one, and it would have been found as a
  fluent-but-wrong model rather than as a crash. Biggest single catch.
- YaRN's `mscale` **does** apply here (≈ 1.34657), by falling through a default
  rather than by being configured, which is the kind of thing that is easy to
  read past.
- The MoE weight axis order needs **no transpose** — the opposite of what
  matching HF's module shapes would suggest.

**One estimate was confirmed rather than corrected**: 13.78 GB resident against
13.76 GB predicted from tensor shapes. §1's memory analysis can be trusted, and
with it the ~2 GB headroom problem (X2) and the ~290 tok/s estimate.

**One assumption was wrong**: the plan assumed CPU or vLLM would be the oracle.
CPU cannot be (38 GB dequantized), and vLLM needs `triton_kernels` it does not
have. transformers-on-GPU-in-MXFP4 was the path, and it needed a pinned version
window nobody would guess.

**Nothing was found that changes the phasing or the recommendation.** X1
(FlashInfer and attention sinks) is still the risk most likely to force an
unpleasant choice, and Phase 0 said nothing about it either way — it is a
Phase 2 discovery by construction.

---

## 9. What Phase 2 changed, and the one finding that outlives it

### MoE routing is a discontinuity, and gpt-oss's router is *close*

The 225-token prompt disagrees with HuggingFace at 13 of 225 positions, and
chasing that down produced the most useful thing in this document.

It is not an implementation error. gpt-oss routes each token to 4 of 32
experts, and the router's margin between the **4th and 5th ranked expert** is
tiny: over that prompt the median is **0.07**, and **28 of 225 positions sit
below 0.01**. bf16 carries about three decimal digits. So any difference in
accumulation order — ours against HuggingFace's, or HuggingFace's against
itself on other hardware — can reorder that boundary, and when it does the
token runs through a *different expert* and its output changes by far more than
the perturbation that caused it.

The measurements that establish this, in order:

1. Divergence begins at **layer 0** (5% relative), so it is not accumulation.
2. Within layer 0, the **attention output** agrees to ~2 bf16 ulps — and a
   from-scratch fp32 numpy reimplementation of that attention disagrees with
   HuggingFace by the *same* amount, so the kernel matches the definition and
   the residue is bf16 rounding.
3. The **MoE output** at layer 0 agrees on average (mean |Δ| 0.007 against a
   mean magnitude of 0.30) but is badly wrong at **10 of 225 positions** — the
   signature of different expert selection, not of wrong arithmetic.
4. Router margins, measured: median 0.07, 12% of positions under 0.01.
5. The disagreement count **moves between 6 and 13 for changes as small as
   swapping `__sincosf` for `sincosf`** — which is the clearest possible
   demonstration that these are coin-flips being re-flipped rather than errors
   being fixed.

This is the same class of fact as §6.3's "a cache hit can change a greedy
answer", and it deserves the same treatment: state it, bound it, and do not
pretend the bound is tighter than the arithmetic allows. The test bounds
disagreement at 10% and says why; a structural bug is nowhere near that line,
as the bias bug demonstrated by missing at 100%.

**It has consequences beyond this test, and they are for Phase 3 and 4 to
absorb rather than rediscover.** Any change to accumulation order in the router
path — batching a request differently, a grouped GEMM, chunked prefill,
serving a prefix from cache — can change which experts a token visits and hence
what it generates. §6.3 already accepts that prefix caching costs bitwise
determinism; MoE routing raises the price, because the perturbation needed is
smaller and the consequence larger. A `--deterministic-routing` mode, if it is
ever wanted, would have to fix the router's arithmetic and not merely its
inputs.

### Smaller corrections to this document

- Per-layer dequantized experts are **1.59 GB**, not the 423 MB stated in
  Phase 2's original sketch — that figure was the packed size. The strategy is
  unaffected; the peak is ~5.5 GB rather than ~2 GB.
- `sincosf` replaced `__sincosf` in the YaRN kernel. At position 224 the angle
  reaches 224 radians, where the fast intrinsic's argument reduction is
  visibly worse. It is not the cause of anything above, and it is free here.
- The existing `RotaryEmbedding` in `layers.cu` still uses `__sincosf`, and
  Qwen's reference test only ever runs 5 tokens. That is an untested corner
  rather than a known bug, and it is worth a look independently of this work.
