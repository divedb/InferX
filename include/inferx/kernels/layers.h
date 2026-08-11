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

/// \brief Scatters new bf16 K/V into an **fp8 e4m3** paged cache, quantizing
/// in flight against fixed per-layer scales.
///
/// The fp8-KV counterpart of `AppendToKvCache`: same paged scatter, but each
/// bf16 element is scaled by `1/k_scale` (K) or `1/v_scale` (V) and converted
/// to e4m3 as it is written. Fused into one pass so the write does not pay a
/// second K/V trip through an fp8 scratch -- the decode step is bandwidth-bound
/// and that round trip would double the cost for nothing. The scales are the
/// same dequant factors the attention kernel takes (`RunFp8`/`PrefillFp8`),
/// frozen per layer at warmup so the value baked into this captured kernel is
/// stable.
///
/// \param k       `[tokens, kv_heads, head_dim]` bf16, new keys.
/// \param v       Same shape, new values.
/// \param k_cache `[blocks, block_size, kv_heads, head_dim]` f8e4m3.
/// \param v_cache Same shape.
/// \param slots   `[tokens]` int32, destination slot per token.
/// \param k_scale Per-layer K dequant scale (amax / 448).
/// \param v_scale Per-layer V dequant scale.
Status AppendBf16AsFp8(const TensorView& k, const TensorView& v,
                       const TensorView& k_cache, const TensorView& v_cache,
                       const TensorView& slots, float k_scale, float v_scale,
                       cudaStream_t stream = nullptr);

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
/// \param max_context The longest key sequence any query in this batch attends
///                    over, which sets the shared-memory tile. Zero means
///                    "assume the block table's full width", which is what this
///                    used to do unconditionally -- and that was a bug worth
///                    naming: the table is `max_blocks_per_seq` wide because the
///                    *scheduler* was configured that way, so a ten-token prompt
///                    asked for enough shared memory to hold `max_seq_len` keys
///                    and failed to launch. A server configured for 16k context
///                    could not prefill anything at all.
///
///                    Callers inside a captured CUDA graph must pass the largest
///                    value they will ever replay with, since the tile size is
///                    baked at capture.
Status PagedAttention(const TensorView& q, const TensorView& k_cache,
                      const TensorView& v_cache, const TensorView& block_table,
                      const TensorView& seq_of_token, const TensorView& q_pos,
                      const TensorView& out, float scale,
                      int64_t max_context = 0,
                      cudaStream_t stream = nullptr);

/// \brief Paged attention that also writes the per-(token, head) log-sum-exp
///        and honours a sliding window.
///
/// The same kernel as `PagedAttention` with two additions gpt-oss needs and no
/// other served model here does:
///
///   * **`lse` output.** The softmax denominator's log is the ingredient that
///     makes an attention sink a post-pass rescale (\see kernels/gpt_oss.h,
///     `ApplyAttentionSinks`) rather than a change inside the kernel. The plain
///     `PagedAttention` computes it internally and discards it; this entry
///     writes it. The convention is **natural log** (`max + ln sum`), so callers
///     that pair it with `ApplyAttentionSinks` must pass `lse_is_log2=false`.
///   * **`window`.** When non-zero, a query at position `p` attends to keys
///     `max(0, p - window + 1) .. p` rather than `0 .. p`. gpt-oss alternates
///     full and sliding (128) layers; a wrong layer mask is a model with the
///     wrong receptive field and no error message. `max_context` should be sized
///     from `window` on a sliding layer so the shared-memory tile is not.
///
/// `lse` may be an empty `TensorView` when the caller does not need it, in
/// which case nothing is written; that keeps a graph capture or a path that
/// only wants the window from having to allocate a sink-scratch it never reads.
Status PagedAttentionWithLse(const TensorView& q, const TensorView& k_cache,
                             const TensorView& v_cache,
                             const TensorView& block_table,
                             const TensorView& seq_of_token,
                             const TensorView& q_pos, const TensorView& out,
                             const TensorView& lse, float scale,
                             int64_t window, int64_t max_context = 0,
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

/// \brief Temperature and nucleus (top-p) sampling, on the device.
///
/// The API accepts `temperature` and `top_p` and, until this existed, ignored
/// them: a caller asking for 1.5 got greedy output and only a
/// `system_fingerprint` of "greedy" hinted otherwise. This closes that gap.
///
/// One block per row, and deliberately no sort. Nucleus sampling is usually
/// written as "sort descending, take the prefix summing to p", but a sort over
/// a 151936-wide vocabulary per sequence per step is far more work than the
/// step it decorates. The same *set* is found by binary-searching the
/// probability cutoff: the nucleus is every token whose probability is at least
/// the largest threshold whose tail mass still reaches `top_p`. Thirty-two
/// bisection steps resolve that to well under bf16's precision, and each step
/// is a block reduction over the row rather than a comparison-sort of it.
///
/// Determinism is by seed, not by luck. `seeds[i]` fully determines row i's
/// draw, so a request that pins its seed reproduces exactly -- which is what
/// makes a sampled server testable at all.
///
/// \param logits      `[rows, vocab]` bf16.
/// \param rows        `[n]` int32, which logits rows to sample.
/// \param temperature `[n]` float. Zero or below means greedy, which is not a
///                    special case bolted on: it is the limit of the
///                    distribution and callers pass it constantly.
/// \param top_p       `[n]` float in (0, 1]. Values at or above 1 disable
///                    truncation.
/// \param top_k       `[n]` int32. Keep only the k most probable tokens
///                    (ties keep more). 0 disables.
/// \param min_p       `[n]` float in [0, 1]. Drop tokens whose probability is
///                    below `min_p * max_prob`. 0 disables.
/// \param seeds       `[n]` uint64, one per row.
/// \param out         `[n]` int32, the sampled ids.
Status SampleTokens(const TensorView& logits, const TensorView& rows,
                    const TensorView& temperature, const TensorView& top_p,
                    const TensorView& top_k, const TensorView& min_p,
                    const TensorView& seeds, const TensorView& out,
                    cudaStream_t stream = nullptr);

/// \brief Applies repetition/presence/frequency penalties and stop-token
/// masks to logits rows in place, before sampling.
///
/// Runs unconditionally inside the captured decode body -- a branch would
/// bake whichever mode was live at capture -- and rows without penalties or
/// masks exit in a handful of reads. The host supplies each row's generated
/// history as (unique id, count) pairs, so no atomics are needed: every entry
/// touches a distinct logit.
///
/// \param logits         `[rows, vocab]` bf16, mutated in place.
/// \param rows           `[n]` int32, which rows to touch.
/// \param presence       `[n]` float, additive on any seen token (0 off).
/// \param frequency      `[n]` float, additive scaled by count (0 off).
/// \param repetition     `[n]` float, HF multiplicative form (1 off).
/// \param history_ids    `[n, history_cap]` int32, unique generated ids,
///                       -1 padded.
/// \param history_counts `[n, history_cap]` int32, occurrence counts.
/// \param mask_ids       `[n, mask_cap]` int32, ids to force to -inf
///                       (min-tokens stop suppression), -1 padded.
Status ApplyPenalties(const TensorView& logits, const TensorView& rows,
                      const TensorView& presence, const TensorView& frequency,
                      const TensorView& repetition,
                      const TensorView& history_ids,
                      const TensorView& history_counts,
                      const TensorView& mask_ids,
                      cudaStream_t stream = nullptr);

/// \brief Log-probabilities of the sampled tokens, after the fact.
///
/// Computed over the (post-penalty) logits at temperature 1, matching
/// vLLM's default reporting rather than the sampling distribution. Rows whose
/// `k_wanted` is negative exit immediately, so the kernel is graph-safe to
/// launch unconditionally.
///
/// \param logits    `[rows, vocab]` bf16.
/// \param rows      `[n]` int32.
/// \param chosen    `[n]` int32, the ids `SampleTokens` picked.
/// \param k_wanted  `[n]` int32: -1 off, 0 chosen-token only, k>0 also the
///                  top-k alternatives.
/// \param chosen_lp `[n]` float out, the chosen token's logprob.
/// \param top_ids   `[n, max_k]` int32 out, most probable first, -1 padded.
/// \param top_lps   `[n, max_k]` float out, parallel to `top_ids`.
Status ComputeLogprobs(const TensorView& logits, const TensorView& rows,
                       const TensorView& chosen, const TensorView& k_wanted,
                       const TensorView& chosen_lp, const TensorView& top_ids,
                       const TensorView& top_lps,
                       cudaStream_t stream = nullptr);

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
