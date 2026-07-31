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

/// \brief Scatters new K/V into their paged cache slots.
///
/// Each token carries the flat slot it belongs in -- `block_index · block_size
/// + offset` -- computed host-side by the scheduler, which is the only thing
/// that knows the block table. Passing slots rather than a block table keeps
/// this kernel indifferent to how blocks were assigned, and lets prefill (many
/// tokens, contiguous slots) and decode (one token per sequence, scattered)
/// use the same launch.
///
/// \param k         `[tokens, kv_heads, head_dim]` bf16, new keys.
/// \param v         Same shape, new values.
/// \param k_cache   `[blocks, block_size, kv_heads, head_dim]` bf16.
/// \param v_cache   Same shape.
/// \param slots     `[tokens]` int32, destination slot per token.
Status AppendToKvCache(const TensorView& k, const TensorView& v,
                       const TensorView& k_cache, const TensorView& v_cache,
                       const TensorView& slots, cudaStream_t stream = nullptr);

/// \brief Causal attention over a paged KV cache.
///
/// The M3 replacement for `Attention`: keys and values are read through a block
/// table rather than from a contiguous buffer, which is what lets sequences
/// grow without reserving their maximum length up front (§6.2).
///
/// One query token per sequence per launch is the decode case; several is
/// chunked prefill. Both go through the same kernel, because the only thing
/// that changes is how many queries share a sequence's keys.
///
/// Correctness is defined by agreement with `Attention` above, which is the
/// naive contiguous reference (R5). The paged kernel is the one that will be
/// swapped for FlashInfer; the reference is the one that stays.
///
/// \param q            `[tokens, q_heads, head_dim]` bf16.
/// \param k_cache      `[blocks, block_size, kv_heads, head_dim]` bf16.
/// \param v_cache      Same shape.
/// \param block_table  `[seqs, max_blocks]` int32. Row `s` lists the blocks
///                     holding sequence `s`, in order.
/// \param seq_of_token `[tokens]` int32, which sequence each query belongs to.
/// \param q_pos        `[tokens]` int32, the query's absolute position, which
///                     is also how many keys precede it.
/// \param out          `[tokens, q_heads, head_dim]` bf16.
/// \param scale        Usually `1/sqrt(head_dim)`.
Status PagedAttention(const TensorView& q, const TensorView& k_cache,
                      const TensorView& v_cache, const TensorView& block_table,
                      const TensorView& seq_of_token, const TensorView& q_pos,
                      const TensorView& out, float scale,
                      cudaStream_t stream = nullptr);

/// \brief Greedy sampling: the argmax of each logits row, on the device.
///
/// §4 step 7 in its simplest form. Sampling on the GPU is what makes the
/// overlap pipeline possible at all: if the host has to read the logits to
/// learn the next token, it has to wait for them, and that wait is the
/// synchronization §5.2 forbids. Writing the token id straight into the buffer
/// the next step's embedding lookup reads means the host never needs the value
/// to keep going -- only positions and slots, which are predictable.
///
/// \param logits `[tokens, vocab]` bf16.
/// \param rows   `[n]` int32, which rows to sample. A prefill produces logits
///               for every prompt token and wants only the last, so sampling
///               all of them would be `tokens` times the work to discard all
///               but one row of it.
/// \param out    `[n]` int32, receiving one token id per requested row.
Status ArgmaxSample(const TensorView& logits, const TensorView& rows,
                    const TensorView& out, cudaStream_t stream = nullptr);

/// \brief Copies `src[i]` into `dst[slot[i]]`, on the device.
///
/// Places each sequence's freshly sampled token where the next step's
/// embedding lookup will read it. A device-to-device scatter rather than a
/// round trip through the host, which is the entire point: the token never
/// leaves the GPU on the critical path.
///
/// \param src   `[n]` int32, sampled ids in batch order.
/// \param dst   `[m]` int32, the next step's token buffer.
/// \param slots `[n]` int32, destination index in `dst` for each entry.
Status ScatterTokens(const TensorView& src, const TensorView& dst,
                     const TensorView& slots, cudaStream_t stream = nullptr);

/// \brief `out[t,:] = table[ids[t],:]`, an embedding lookup.
///
/// \param table `[vocab, hidden]` bf16.
/// \param ids   `[tokens]` int32. Out-of-range ids are an error, not a wrap.
/// \param out   `[tokens, hidden]` bf16.
Status EmbeddingLookup(const TensorView& table, const TensorView& ids,
                       const TensorView& out, cudaStream_t stream = nullptr);

/// \brief Splits a fused QKV projection into contiguous Q, K and V, adding the
///        bias on the way through.
///
/// A fused `[tokens, q_dim + 2·kv_dim]` GEMM is one launch where three were,
/// but its output is interleaved per token, so Q is strided as soon as there is
/// more than one token -- and everything downstream is contiguous-only by
/// design (T16). A split is therefore unavoidable. Folding the bias into it
/// makes the fused path cost *two* launches per layer where the unfused one
/// costs six: three GEMMs and three bias adds.
///
/// \param fused `[tokens, q_dim + 2·kv_dim]` bf16, the GEMM's output.
/// \param bias  `[q_dim + 2·kv_dim]` bf16, or an undefined view for
///              architectures without attention bias (Llama).
/// \param q     `[tokens, q_dim]` bf16.
/// \param k, v  `[tokens, kv_dim]` bf16.
Status SplitQkvWithBias(const TensorView& fused, const TensorView& bias,
                        const TensorView& q, const TensorView& k,
                        const TensorView& v, cudaStream_t stream = nullptr);

/// \brief `out = silu(fused[..., :n]) * fused[..., n:]` for a fused gate/up.
///
/// The FFN's counterpart, and it needs no split at all: SiluMul is elementwise,
/// so it can read both halves out of one buffer directly. Fusing gate and up
/// therefore removes a launch for free.
///
/// \param fused `[tokens, 2·intermediate]` bf16.
/// \param out   `[tokens, intermediate]` bf16.
Status SiluMulFused(const TensorView& fused, const TensorView& out,
                    cudaStream_t stream = nullptr);

/// \brief `out[t, i] += bias[i]`, broadcasting the bias across tokens.
///
/// Qwen2 biases its Q/K/V projections; Llama biases nothing. Kept separate from
/// the GEMM rather than folded into a cuBLASLt epilogue because the epilogue
/// would have to be part of the cached plan, and the same shape is used with
/// and without a bias.
///
/// \param out  `[tokens, width]` bf16, updated in place.
/// \param bias `[width]` bf16.
Status AddBiasInPlace(const TensorView& out, const TensorView& bias,
                      cudaStream_t stream = nullptr);

/// \brief `out += residual`, elementwise, in bf16 with fp32 accumulation.
///
/// \param out      `[tokens, hidden]` bf16, updated in place.
/// \param residual `[tokens, hidden]` bf16.
Status AddInPlace(const TensorView& out, const TensorView& residual,
                  cudaStream_t stream = nullptr);

}  // namespace inferx::kernels
