#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

/// The kernels multi-head latent attention needs that grouped-query attention
/// does not.
///
/// MLA's difference from GQA is what it stores. GQA caches `K` and `V` per head
/// per token; MLA caches **one compressed latent** per token per layer, plus
/// **one RoPE key shared by every head** — and reconstructs each head's K and V
/// from that latent with a projection. The cache is therefore
/// `kv_lora_rank + qk_rope_head_dim` wide instead of `2 · kv_heads · head_dim`,
/// which for DeepSeek-V2's shape is 576 elements against 8192.
///
/// Two structural consequences, both of which the rest of the engine has to
/// respect and neither of which is negotiable (§7.3, T11):
///
///   * The pool's `[k|v][heads][head_dim]` geometry does not apply. MLA
///     instantiates `KvLayout` with `entries_per_token = 1`, `kv_heads = 1` and
///     `head_dim = kv_lora_rank + qk_rope_head_dim`, and reads the whole thing
///     through `KeyCache()` — `ValueCache()` deliberately fails for this layout
///     rather than returning half of a latent.
///   * The latent is **replicated** across tensor-parallel ranks, not sharded,
///     because every rank needs every head's reconstruction input. KV memory
///     does not fall with TP for an MLA model, which is why
///     `ModelConfig::KvElementsPerTokenPerLayer` is a model property rather
///     than something the scheduler divides by `tp_size`.
///
/// The RoPE convention here is the half-split ("NeoX") one, the same as
/// `RotaryEmbedding` — see the note on `MlaRopeInPlace` for why that choice is
/// stated rather than assumed.
namespace inferx::kernels {

/// \brief Applies RoPE in place to the trailing `rope_dim` columns of each head.
///
/// MLA splits every Q head into a position-free part and a rotated part laid
/// out contiguously as `[nope | rope]`, so the rotated part is a strided
/// sub-vector rather than a tensor of its own. Copying it out, rotating it and
/// copying it back would be three passes over the same bytes to avoid a stride;
/// this rotates in place.
///
/// The rotation is the half-split form: within the `rope_dim` sub-vector,
/// element `j` pairs with element `j + rope_dim/2`. **This is a convention, and
/// it is stated rather than inferred**: HF's DeepSeek implementation reaches
/// the same maths through an interleaving of its own, and without a DeepSeek
/// checkpoint on this box there is no way to assert the two agree. What the
/// tests here do assert is that this kernel is the same rotation
/// `RotaryEmbedding` applies, so the two attention families cannot drift apart.
///
/// \param x         `[tokens, heads, head_dim]` bf16, rotated in place.
/// \param rope_dim  Width of the trailing rotated part. Must be even and no
///                  wider than `head_dim`.
/// \param positions `[tokens]` int32, absolute position of each token.
/// \param theta     `rope_theta` from the config.
Status MlaRopeInPlace(const TensorView& x, int64_t rope_dim,
                      const TensorView& positions, float theta,
                      cudaStream_t stream = nullptr);

/// \brief `MlaRopeInPlace` with the frequencies from a precomputed table.
///
/// The table-driven form YaRN needs, standing to `MlaRopeInPlace` exactly as
/// `RotaryEmbeddingFromTable` stands to `RotaryEmbedding`: the blended
/// frequencies are not a closed form of two constants, and the attention
/// factor scales cos and sin. With `inv_freq[j] = theta^(-2j/rope_dim)` and
/// `attn_factor = 1` this is the same rotation as `MlaRopeInPlace`, which is
/// what the tests pin.
///
/// \param x           `[tokens, heads, head_dim]` bf16, rotated in place.
/// \param rope_dim    Width of the trailing rotated part.
/// \param positions   `[tokens]` int32.
/// \param inv_freq    `[rope_dim/2]` fp32 on the device — for YaRN, from
///                    `ComputeYarnInvFreq` over `rope_dim` (the rope
///                    sub-vector has its own frequency ladder, so the blend is
///                    computed over its width, not the head's).
/// \param attn_factor Scales cos and sin. For DeepSeek this is
///                    `YarnMscale(factor, mscale) / YarnMscale(factor,
///                    mscale_all_dim)` — exactly 1 for V2-Lite, where the two
///                    are equal and the whole mscale effect moves into the
///                    attention softmax scale instead.
Status MlaRopeFromTable(const TensorView& x, int64_t rope_dim,
                        const TensorView& positions, const TensorView& inv_freq,
                        float attn_factor, cudaStream_t stream = nullptr);

/// \brief Splits `[rows, heads, a + b]` into `[rows, heads, a]` and
///        `[rows, heads, b]`.
///
/// MLA concatenates pairs along the head dimension in three places — Q into
/// `[nope | rope]`, the KV down-projection into `[latent | rope_key]`, and the
/// up-projection into `[k_nope | v]` — because each pair comes out of one GEMM
/// and separating them at the source would cost a second one. Everything
/// downstream wants them apart and contiguous, so one kernel does all three.
///
/// \param src  `[rows, heads, a + b]` bf16.
/// \param head `[rows, heads, a]` bf16 — the leading part.
/// \param tail `[rows, heads, b]` bf16 — the trailing part.
Status SplitTrailing(const TensorView& src, const TensorView& head,
                     const TensorView& tail, cudaStream_t stream = nullptr);

/// \brief Writes each token's latent and shared RoPE key into the paged cache.
///
/// One row per token, `[latent | rope_key]`, at the slot the block table
/// assigned. The GQA counterpart is `AppendToKvCache`; this one writes a single
/// entry rather than a K and a V, which is the whole point.
///
/// \param latent       `[tokens, kv_lora_rank]` bf16, already layer-normed.
/// \param rope_key     `[tokens, qk_rope_head_dim]` bf16, already rotated.
/// \param cache        `[num_blocks, block_size, 1, kv_lora_rank +
///                     qk_rope_head_dim]` bf16 — the pool's `KeyCache(layer)`.
/// \param slot_mapping `[tokens]` int32, the flat slot
///                     `block · block_size + offset` each token writes to.
Status MlaAppendLatent(const TensorView& latent, const TensorView& rope_key,
                       const TensorView& cache, const TensorView& slot_mapping,
                       cudaStream_t stream = nullptr);

/// \brief Reads a sequence's cached latents back into a contiguous buffer.
///
/// The up-projection that reconstructs K and V is an ordinary dense GEMM, and
/// a GEMM cannot walk a block table — so the latents a sequence needs are
/// gathered into one `[context, latent_width]` buffer first. That gather is
/// the cost of the *unabsorbed* form of MLA decode, and it is exactly what the
/// absorbed form exists to avoid (see `MlaAttention`).
///
/// \param cache        `KeyCache(layer)` as above.
/// \param block_table  `[num_blocks_for_seq]` int32, the sequence's blocks.
/// \param context_len  Tokens to read, starting from position 0.
/// \param out          `[context_len, kv_lora_rank + qk_rope_head_dim]` bf16.
Status MlaGatherLatents(const TensorView& cache, const TensorView& block_table,
                        int64_t context_len, const TensorView& out,
                        cudaStream_t stream = nullptr);

/// \brief Causal attention over reconstructed K/V, with the decoupled RoPE key.
///
/// The score for query `i`, key `j` and head `h` is
///
///     (q_nope[i,h] · k_nope[j,h] + q_rope[i,h] · k_rope[j]) / sqrt(qk_dim)
///
/// — note that `k_rope` carries no head index. That shared term is the
/// decoupled part of MLA, and folding it into the same softmax as the per-head
/// term is the only structural difference from ordinary attention.
///
/// This is the **unabsorbed** form: `k_nope` and `v` have already been
/// reconstructed for every cached token, so this kernel is a plain attention
/// and the reconstruction is a GEMM the caller ran. The absorbed form — fold
/// `kv_b_proj` into the query so attention runs against the latent directly,
/// which is what makes MLA decode cheap — is not implemented here (§14, M9).
///
/// \param q_nope     `[q_tokens, heads, qk_nope_head_dim]` bf16.
/// \param q_rope     `[q_tokens, heads, qk_rope_head_dim]` bf16, rotated.
/// \param k_nope     `[context, heads, qk_nope_head_dim]` bf16.
/// \param k_rope     `[context, qk_rope_head_dim]` bf16, rotated. One per token.
/// \param v          `[context, heads, v_head_dim]` bf16.
/// \param out        `[q_tokens, heads, v_head_dim]` bf16.
/// \param query_base Absolute position of the first query token, so causality
///                   is `key_pos <= query_base + i`. A decode step passes
///                   `context - 1`; a prefill from empty passes 0.
/// \param scale      `1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`, passed
///                   rather than derived because YaRN-scaled MLA configs
///                   multiply it by a factor the kernel has no way to know.
Status MlaAttention(const TensorView& q_nope, const TensorView& q_rope,
                    const TensorView& k_nope, const TensorView& k_rope,
                    const TensorView& v, const TensorView& out,
                    int64_t query_base, float scale,
                    cudaStream_t stream = nullptr);

// --- The absorbed form (§14, M9's "obvious next move") ----------------------
//
// Fold `kv_b` into the query and the output instead of reconstructing K and V
// for the whole context every step:
//
//     score(i,j,h) = (W_UK[h]ᵀ q_nope[i,h]) · c[j]  +  q_rope[i,h] · k_rope[j]
//     out(i,h)     = W_UV[h] · Σ_j softmax_j · c[j]
//
// which is algebraically the unabsorbed computation with the projections
// hoisted out of the sum. What changes is the shape of the work: a decode
// step touches O(context · (latent + rope)) cached bytes directly through the
// block table — no gather, no O(context · heads · (nope + v)) reconstruction
// GEMM, and no scratch sized by context on the token side.

/// \brief `q_lat[t,h] = W_UK[h]ᵀ · q_nope[t,h]`.
///
/// \param q_nope     `[tokens, heads, qk_nope_head_dim]` bf16 (not rotated —
///                   the nope part never is).
/// \param kv_b       The unabsorbed path's same `[heads·(nope+v), latent]`
///                   weight; head `h`'s W_UK is rows `[h·(nope+v),
///                   h·(nope+v)+nope)`. Passed whole so the two paths cannot
///                   drift apart by loading different tensors.
/// \param q_lat      `[tokens, heads, kv_lora_rank]` bf16, written.
/// \param v_head_dim The v width, needed only to stride between heads.
Status MlaAbsorbQ(const TensorView& q_nope, const TensorView& kv_b,
                  const TensorView& q_lat, int64_t v_head_dim,
                  cudaStream_t stream = nullptr);

/// \brief Causal attention of absorbed queries directly against the paged
///        latent cache.
///
/// Reads `[latent | rope_key]` rows through the block table; the score is
/// `q_lat` against the latent part plus `q_rope` against the tail, and the
/// value accumulated is the latent itself. One sequence per call, like the
/// unabsorbed path. Two-pass score-recomputing reference kernel, same as
/// `MlaAttention`.
///
/// \param q_lat       `[tokens, heads, kv_lora_rank]` bf16 from `MlaAbsorbQ`.
/// \param q_rope      `[tokens, heads, qk_rope_head_dim]` bf16, rotated.
/// \param cache       `KeyCache(layer)` — `[blocks, block_size, 1,
///                    latent + rope]`.
/// \param block_table `[blocks_for_seq]` int32.
/// \param context_len Cache length after this step's append.
/// \param out_lat     `[tokens, heads, kv_lora_rank]` bf16, written.
/// \param query_base  As in `MlaAttention`.
/// \param scale       The same YaRN-corrected softmax scale.
Status MlaLatentAttention(const TensorView& q_lat, const TensorView& q_rope,
                          const TensorView& cache,
                          const TensorView& block_table, int64_t context_len,
                          const TensorView& out_lat, int64_t query_base,
                          float scale, cudaStream_t stream = nullptr);

/// \brief `out[t,h] = W_UV[h] · attn_lat[t,h]`.
///
/// \param attn_lat         `[tokens, heads, kv_lora_rank]` bf16.
/// \param kv_b             As in `MlaAbsorbQ`; head `h`'s W_UV is rows
///                         `[h·(nope+v)+nope, h·(nope+v)+nope+v)`.
/// \param out              `[tokens, heads, v_head_dim]` bf16, written.
/// \param qk_nope_head_dim The nope width, needed only to stride heads.
Status MlaUnabsorbOut(const TensorView& attn_lat, const TensorView& kv_b,
                      const TensorView& out, int64_t qk_nope_head_dim,
                      cudaStream_t stream = nullptr);

}  // namespace inferx::kernels
