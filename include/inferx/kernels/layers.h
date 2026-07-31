#pragma once

#include <cuda_runtime_api.h>

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// The per-layer kernels a decoder-only transformer needs beyond its GEMMs.
///
/// All of them take and return **bf16**, and accumulate in **fp32**. bf16
/// because that is what the checkpoint stores, so computing in it means no
/// conversion pass over 6 GB of weights; fp32 accumulation because bf16 has 8
/// mantissa bits and a 2048-term sum of them is not worth the memory it would
/// save. Every one of these is small, memory-bound and fusable, which is
/// exactly the category ARCHITECTURE.md §9 says we write ourselves.
///
/// Shapes are stated per call. Nothing here is batched over sequences: M2 is
/// batch 1 with no KV cache, and the shapes are chosen so that adding both
/// later changes the launch config rather than the maths.

/// \brief `out[t,:] = x[t,:] * rsqrt(mean(x[t,:]²) + eps) * weight`.
///
/// RMSNorm rather than LayerNorm: no mean subtraction and no bias, which is
/// what Llama and Qwen2 both use. The mean of squares is accumulated in fp32
/// across the row regardless of the input dtype.
///
/// \param x      `[tokens, hidden]` bf16.
/// \param weight `[hidden]` bf16.
/// \param out    `[tokens, hidden]` bf16. May alias `x`.
/// \param eps    Added inside the sqrt, from the config.
Status RmsNorm(const TensorView& x, const TensorView& weight,
               const TensorView& out, float eps,
               cudaStream_t stream = nullptr);

/// \brief Applies rotary position embeddings to Q and K, in place.
///
/// The half-split ("NeoX") formulation HF uses, *not* the interleaved one from
/// the original RoFormer code. For a head of width `d` and position `p`, with
/// `inv_freq[j] = theta^(-2j/d)` for `j < d/2`:
///
///     out[j]       = x[j]       * cos(p·inv_freq[j]) - x[j + d/2] * sin(p·inv_freq[j])
///     out[j + d/2] = x[j + d/2] * cos(p·inv_freq[j]) + x[j]       * sin(p·inv_freq[j])
///
/// Getting the two conventions confused produces a model that generates fluent
/// text with no long-range coherence, which is why the formula is written out
/// here rather than left to the reader of the kernel.
///
/// Q and K are rotated with the same frequencies; only their head counts
/// differ, which is the GQA asymmetry.
///
/// \param q         `[tokens, q_heads, head_dim]` bf16, rotated in place.
/// \param k         `[tokens, kv_heads, head_dim]` bf16, rotated in place.
/// \param positions `[tokens]` int32, absolute position of each token.
/// \param theta     `rope_theta` from the config (1e6 for Qwen2.5).
Status RotaryEmbedding(const TensorView& q, const TensorView& k,
                       const TensorView& positions, float theta,
                       cudaStream_t stream = nullptr);

/// \brief `out = silu(gate) * up`, elementwise, where `silu(x) = x·sigmoid(x)`.
///
/// The activation half of SwiGLU. The two projections are separate tensors here
/// rather than one fused `[2·intermediate]` buffer, because the checkpoint
/// stores them separately and fusing is a load-time decision we have not made
/// yet.
///
/// \param gate `[tokens, intermediate]` bf16.
/// \param up   `[tokens, intermediate]` bf16.
/// \param out  `[tokens, intermediate]` bf16. May alias either input.
Status SiluMul(const TensorView& gate, const TensorView& up,
               const TensorView& out, cudaStream_t stream = nullptr);

/// \brief Naive causal self-attention with GQA. Batch 1, no KV cache.
///
/// Deliberately the simplest correct thing: one block per (query, head), an
/// fp32 online-softmax pass over the keys, no tiling and no shared-memory
/// staging of K/V. It is the reference the FlashInfer path gets checked against
/// at M3 (R5), so being obviously correct matters more here than being fast.
///
/// GQA is handled by mapping query head `h` to KV head `h / (q_heads/kv_heads)`
/// rather than by materializing repeated K/V, which would cost the memory the
/// grouping exists to save.
///
/// \param q     `[tokens, q_heads, head_dim]` bf16.
/// \param k     `[tokens, kv_heads, head_dim]` bf16.
/// \param v     `[tokens, kv_heads, head_dim]` bf16.
/// \param out   `[tokens, q_heads, head_dim]` bf16. Must not alias q/k/v.
/// \param scale Usually `1/sqrt(head_dim)`; passed in so a model that scales
///              differently does not need a new kernel.
Status Attention(const TensorView& q, const TensorView& k, const TensorView& v,
                 const TensorView& out, float scale,
                 cudaStream_t stream = nullptr);

/// \brief `out[t,:] = table[ids[t],:]`, an embedding lookup.
///
/// \param table `[vocab, hidden]` bf16.
/// \param ids   `[tokens]` int32. Out-of-range ids are an error, not a wrap.
/// \param out   `[tokens, hidden]` bf16.
Status EmbeddingLookup(const TensorView& table, const TensorView& ids,
                       const TensorView& out, cudaStream_t stream = nullptr);

/// \brief `out += residual`, elementwise, in bf16 with fp32 accumulation.
///
/// \param out      `[tokens, hidden]` bf16, updated in place.
/// \param residual `[tokens, hidden]` bf16.
Status AddInPlace(const TensorView& out, const TensorView& residual,
                  cudaStream_t stream = nullptr);

}  // namespace inferx::kernels
