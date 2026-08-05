#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

/// The three things gpt-oss does differently from every other model in this
/// engine, once MXFP4 is dealt with.
///
/// Each is small. Each is also a silent-wrongness bug if guessed rather than
/// read out of the reference implementation, which is what
/// docs/gpt-oss-20b-llm.md §8 is about — every one of these was verified
/// against `transformers.models.gpt_oss` rather than derived.
namespace inferx::kernels {

/// \brief Folds attention sinks into an already-computed attention output.
///
/// gpt-oss gives every head a **learned sink logit** that participates in the
/// softmax denominator with no value vector behind it — the head can attend
/// "nowhere" and emit near-zero. HuggingFace expresses it by concatenating the
/// sink to that row's scores, softmaxing over `[scores | sink]`, and dropping
/// the sink column.
///
/// Done that way it would need a change inside the attention kernel, and
/// FlashInfer's **decode** kernel has no hook for it — `prefill.cuh` calls
/// `variant.update_m_d`, `decode.cuh` maintains its softmax state directly. That
/// looked like the plan's worst risk (X1): patch a pinned submodule we do not
/// own, or fall back to our own slower kernel.
///
/// It is neither, because the sink factors out exactly. With `D = Σ_j e^{s_j}`
/// over the real keys, the sinked output is
///
///     out_sink = (Σ_j e^{s_j} v_j) / (D + e^{sink})
///              = out_plain · D / (D + e^{sink})
///              = out_plain · σ(lse − sink)
///
/// where `lse = ln D` is the log-sum-exp the attention kernel already computes
/// and already has a field for. So a sink is **a scalar rescale per (token,
/// head)** applied after an ordinary attention, and the kernel underneath needs
/// to know nothing about it. Verified against the concatenate-and-drop form to
/// 2e-16 across sinks from −5 to +20.
///
/// \param out    `[tokens, heads, head_dim]` bf16, rescaled in place.
/// \param lse    `[tokens, heads]` fp32, as written by FlashInfer's `lse` output.
/// \param sinks  `[heads]` bf16, the learned per-head logits.
/// \param lse_is_log2 FlashInfer's `state_t::get_lse()` returns `m + log2(d)` —
///                    base **two**, because the kernels fold `log2(e)` into the
///                    softmax scale so they can use `exp2`. Pass true for that
///                    convention and the kernel converts; pass false if the lse
///                    is a natural log. Getting this wrong is a plausible-looking
///                    error of a factor of `ln 2` inside a sigmoid, which is why
///                    it is an explicit argument rather than a constant.
Status ApplyAttentionSinks(const TensorView& out, const TensorView& lse,
                           const TensorView& sinks, bool lse_is_log2 = true,
                           cudaStream_t stream = nullptr);

/// \brief gpt-oss's gated activation: clamped, alpha-scaled, and offset.
///
/// Four differences from the `silu(gate) · up` every other model here uses, and
/// all four change the numbers:
///
///     gate = clamp(gate, max=limit)          # limit = 7.0
///     up   = clamp(up, -limit, limit)
///     glu  = gate * sigmoid(gate * alpha)    # alpha = 1.702, not plain SiLU
///     out  = (up + 1) * glu                  # note the +1
///
/// `alpha = 1.702` makes `x·σ(αx)` a close approximation of GELU rather than
/// SiLU, and the `(up + 1)` means a zero `up` passes the gate through rather
/// than annihilating it.
///
/// A fifth difference is **not** handled here and deliberately so: gpt-oss
/// interleaves gate and up along the `2·intermediate` axis, where this kernel
/// expects them split, `[gate | up]`. That permutation happens once at load, in
/// `DequantizeMxfp4GateUpToBf16`, so the run-time path never sees the odd
/// layout. \see mxfp4.h.
///
/// \param gate_up `[tokens, 2·intermediate]` bf16, `[gate | up]`.
/// \param out     `[tokens, intermediate]` bf16.
/// \param limit   `swiglu_limit` from the config (7.0).
/// \param alpha   1.702.
Status GptOssSwiGlu(const TensorView& gate_up, const TensorView& out,
                    float limit, float alpha, cudaStream_t stream = nullptr);

/// \brief Rotary embedding from a precomputed inverse-frequency table.
///
/// The existing `RotaryEmbedding` computes `inv_freq[j] = theta^(-2j/d)` inside
/// the kernel, which is right for a model whose frequencies are a closed form.
/// gpt-oss uses **YaRN**, where each dimension's frequency is a blend of
/// interpolated and extrapolated values controlled by a ramp between
/// `beta_fast` and `beta_slow` — perfectly computable, but not from two
/// constants, and recomputing it per launch would be silly when it is fixed for
/// the whole run.
///
/// It also carries an **attention factor**: YaRN scales `cos`/`sin` by
/// `0.1·ln(factor) + 1`, which for gpt-oss's `factor = 32` is ≈ 1.34657. That
/// arrives by falling through a default rather than by being configured, so it
/// is easy to read past and it is a constant multiplier on every rotated
/// component.
///
/// \param q            `[tokens, q_heads, head_dim]` bf16, rotated in place.
/// \param k            `[tokens, kv_heads, head_dim]` bf16, rotated in place.
/// \param positions    `[tokens]` int32.
/// \param inv_freq     `[head_dim/2]` fp32, precomputed on the host.
/// \param attn_factor  Scales cos and sin. 1.0 for a model without YaRN.
Status RotaryEmbeddingFromTable(const TensorView& q, const TensorView& k,
                                const TensorView& positions,
                                const TensorView& inv_freq, float attn_factor,
                                cudaStream_t stream = nullptr);

/// \brief Fills `inv_freq` with YaRN's blended frequencies, on the host.
///
/// Separate from the kernel because it runs once per model, is fiddly, and is
/// far easier to check against the reference on the host than on the device.
/// The blend follows `transformers.modeling_rope_utils._compute_yarn_parameters`.
///
/// \param head_dim      Rotary width.
/// \param base          `rope_theta`.
/// \param factor        `rope_scaling.factor`.
/// \param beta_fast     `rope_scaling.beta_fast`.
/// \param beta_slow     `rope_scaling.beta_slow`.
/// \param original_max  `rope_scaling.original_max_position_embeddings`.
/// \param truncate      `rope_scaling.truncate`.
/// \param out           `[head_dim/2]` floats, filled.
/// \return              The attention factor to pass to
///                      `RotaryEmbeddingFromTable`.
float ComputeYarnInvFreq(int64_t head_dim, double base, double factor,
                         double beta_fast, double beta_slow,
                         int64_t original_max, bool truncate, float* out);

}  // namespace inferx::kernels
