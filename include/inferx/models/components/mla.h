/// \file
/// \brief Multi-latent attention (DeepSeek): configuration, weights, and
///        decompressed execution.

#ifndef INFERX_MODELS_COMPONENTS_MLA_H_
#define INFERX_MODELS_COMPONENTS_MLA_H_

#include <cstdint>
#include <utility>

#include "inferx/core/tensor.h"
#include "inferx/models/components/rope.h"
#include "inferx/models/state.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/flash_attention.h"

namespace inferx {

struct AttentionBatch;
class DiagnosticTrace;
class KvBlockPool;

namespace components {

/// \brief Multi-latent attention in its decompressed form.
///
/// The latent projections run per step and per-head K/V enter the same paged
/// pool as GQA (kv_heads = query_heads, head_dim = nope + rope, V
/// zero-padded), so attention kernels, block management, and CUDA graphs are
/// shared with every other family. The compressed latent cache that makes
/// MLA cheap at DeepSeek scale is a future KvLayout mode (entries_per_token
/// already anticipates it); this path is the correct one first.
struct MlaConfig {
  int64_t query_heads = 0;
  int64_t q_lora_rank = 0;       ///< Query down-projection width.
  int64_t kv_lora_rank = 0;      ///< Shared KV latent width.
  int64_t qk_nope_head_dim = 0;  ///< Per-head latent key width.
  int64_t qk_rope_head_dim = 0;  ///< Per-head rotated key width.
  int64_t v_head_dim = 0;        ///< Per-head value width.
  /// Attention logit scale; 0 selects 1/sqrt(nope + rope).
  float scale_override = 0.0f;
  RotaryConfig rotary;

  /// Combined per-head key dimension; also the padded V dimension.
  int64_t head_dim() const { return qk_nope_head_dim + qk_rope_head_dim; }
};

/// \brief MLA projections as loaded.
///
/// `q_b` rows are per head [rope | nope] so RoPE rotates the leading slice;
/// `k_rope` and `kv_latent` are the row split of the checkpoint's fused
/// kv_a_proj_with_mqa; `output` is column-padded to heads * head_dim with
/// zeros so the padded V tail contributes nothing.
struct MlaWeights {
  Tensor q_a;        ///< [q_lora, hidden]
  Tensor q_a_norm;   ///< [q_lora]
  Tensor q_b;        ///< [heads * head_dim, q_lora], per head [rope|nope].
  Tensor k_rope;     ///< [rope, hidden]
  Tensor kv_latent;  ///< [kv_lora, hidden]
  Tensor kv_a_norm;  ///< [kv_lora]
  Tensor kv_b;       ///< [heads * (nope + v), kv_lora]
  Tensor output;     ///< [hidden, heads * head_dim], zero-padded columns.
};

/// One down-projection's weight; kept separate from LinearWeights because
/// these are internal to the mixer (no sharding or bias yet).
/// \brief MLA workspace buffers, sized once by the decoder stack.
struct MlaWorkspace {
  std::optional<Tensor> q_lora;   ///< [max_tokens, q_lora_rank]
  std::optional<Tensor> kv_lora;  ///< [max_tokens, kv_lora_rank]
  std::optional<Tensor> k_rope;   ///< [max_tokens, rope]
  std::optional<Tensor> kv_b;     ///< [max_tokens, heads * (nope + v)]
  std::optional<Tensor> query;    ///< [max_tokens, heads, head_dim]
  std::optional<Tensor> key;      ///< [max_tokens, heads, head_dim]
  std::optional<Tensor> value;    ///< [max_tokens, heads, head_dim]
  std::optional<Tensor> attn_out;  ///< [max_tokens, heads, head_dim]
  ops::AttentionPlanWorkspace plan;  ///< Kernel planning for the shared path.
};

/// \brief Runs one MLA layer into `mixed_out` ([rows, hidden]).
///
/// Down-project, normalize, up-project, rotate the shared rope slice,
/// assemble per-head K/V, then reuse the paged-attention path every other
/// mixer uses.
Status RunMlaAttention(const MlaConfig& config, const MlaWeights& weights, const Tensor& normed,
                       float norm_eps, const AttentionBatch& batch, const PagedKvState& kv_state,
                       const KvBlockPool& pool, MlaWorkspace& ws, ops::ExecutionContext& ctx,
                       DiagnosticTrace* trace, std::string_view prefix, Tensor& mixed_out);

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_MLA_H_
