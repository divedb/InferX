#include "inferx/models/components/mla.h"

#include <cmath>
#include <string_view>

#include "inferx/core/shape.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/op_context.h"
#include "inferx/ops/linear.h"
#include "inferx/ops/mla.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rotary.h"

namespace inferx::components {

Status RunMlaAttention(const MlaConfig& a, const MlaWeights& w, const Tensor& normed,
                       float norm_eps, const AttentionBatch& batch, const PagedKvState& kv_state,
                       const KvBlockPool& pool, MlaWorkspace& ws, ops::OpContext& ctx,
                       Tensor& mixed_out) {
  const int64_t rows = normed.Dim(0);
  const int64_t heads = a.query_heads;
  const int64_t nope = a.qk_nope_head_dim, rope = a.qk_rope_head_dim;
  const int64_t v_dim = a.v_head_dim, head_dim = a.head_dim();

  // Query side: down-project, normalize, up-project per head [rope | nope].
  INFERX_ASSIGN_OR_RETURN(Tensor qa, ws.q_lora->Slice(0, rows * a.q_lora_rank));
  INFERX_ASSIGN_OR_RETURN(Tensor qa2d, qa.Reshape(Shape({rows, a.q_lora_rank})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.q_a, qa2d));
  INFERX_RETURN_IF_ERROR(
      ops::RmsNorm(ctx, qa2d, w.q_a_norm, qa2d, ops::RMSNormConfig{norm_eps, false, true}));
  INFERX_ASSIGN_OR_RETURN(Tensor q_flat, ws.query->Slice(0, rows * heads * head_dim));
  INFERX_ASSIGN_OR_RETURN(Tensor q, q_flat.Reshape(Shape({rows, heads, head_dim})));
  {
    INFERX_ASSIGN_OR_RETURN(Tensor q2d, q_flat.Reshape(Shape({rows, heads * head_dim})));
    INFERX_RETURN_IF_ERROR(ops::Linear(ctx, qa2d, w.q_b, q2d));
  }

  // KV side: the shared rope slice, then the normalized latent up-projection.
  INFERX_ASSIGN_OR_RETURN(Tensor k_rope_flat, ws.k_rope->Slice(0, rows * rope));
  INFERX_ASSIGN_OR_RETURN(Tensor k_rope, k_rope_flat.Reshape(Shape({rows, rope})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.k_rope, k_rope));
  INFERX_ASSIGN_OR_RETURN(Tensor c_flat, ws.kv_lora->Slice(0, rows * a.kv_lora_rank));
  INFERX_ASSIGN_OR_RETURN(Tensor latent, c_flat.Reshape(Shape({rows, a.kv_lora_rank})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.kv_latent, latent));
  INFERX_RETURN_IF_ERROR(
      ops::RmsNorm(ctx, latent, w.kv_a_norm, latent, ops::RMSNormConfig{norm_eps, false, true}));
  INFERX_ASSIGN_OR_RETURN(Tensor kvb_flat, ws.kv_b->Slice(0, rows * heads * (nope + v_dim)));
  INFERX_ASSIGN_OR_RETURN(Tensor up, kvb_flat.Reshape(Shape({rows, heads * (nope + v_dim)})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, latent, w.kv_b, up));

  INFERX_ASSIGN_OR_RETURN(Tensor k_flat, ws.key->Slice(0, rows * heads * head_dim));
  INFERX_ASSIGN_OR_RETURN(Tensor k, k_flat.Reshape(Shape({rows, heads, head_dim})));
  INFERX_ASSIGN_OR_RETURN(Tensor v_flat, ws.value->Slice(0, rows * heads * head_dim));
  INFERX_ASSIGN_OR_RETURN(Tensor v, v_flat.Reshape(Shape({rows, heads, head_dim})));
  INFERX_RETURN_IF_ERROR(ops::AssembleMlaCaches(ctx, k_rope, up, heads, nope, rope, v_dim, k, v));

  // RoPE rotates the leading rope slice of every head; the nope columns and
  // the zero tail pass through untouched.
  INFERX_RETURN_IF_ERROR(ops::ApplyRope(ctx, q, k, batch.positions, a.rotary.Params()));

  INFERX_ASSIGN_OR_RETURN(Tensor key_cache, pool.KeyCache(kv_state.pool_layer));
  INFERX_ASSIGN_OR_RETURN(Tensor value_cache, pool.ValueCache(kv_state.pool_layer));
  const int64_t block_size = pool.BlockSize();
  INFERX_ASSIGN_OR_RETURN(Tensor k2d, k_flat.Reshape(Shape({rows, heads * head_dim})));
  INFERX_ASSIGN_OR_RETURN(Tensor v2d, v_flat.Reshape(Shape({rows, heads * head_dim})));
  INFERX_RETURN_IF_ERROR(ops::WritePagedKv(ctx, k2d, v2d, batch.positions, batch.batch_indices,
                                           batch.kv_indptr, batch.kv_indices, key_cache,
                                           value_cache, block_size));
  INFERX_ASSIGN_OR_RETURN(Tensor out_flat, ws.attn_out->Slice(0, rows * heads * head_dim));
  INFERX_ASSIGN_OR_RETURN(Tensor attn_out, out_flat.Reshape(Shape({rows, heads * head_dim})));
  ops::AttentionParams params;
  params.query_heads = heads;
  params.kv_heads = heads;  // Decompressed: every head has its own K and V.
  params.head_dim = head_dim;
  params.scale = a.scale_override > 0.0f
                     ? a.scale_override
                     : 1.0f / std::sqrt(static_cast<float>(head_dim));
  INFERX_ASSIGN_OR_RETURN(Tensor q2d, q_flat.Reshape(Shape({rows, heads * head_dim})));
  INFERX_RETURN_IF_ERROR(ops::PagedAttention(ctx, q2d, batch.qo_indptr, batch.kv_indptr,
                                             batch.kv_indices, batch.last_page_len,
                                             batch.host_qo_indptr, batch.num_seqs, key_cache,
                                             value_cache, block_size, params, ws.plan,
                                             attn_out));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, attn_out, w.output, mixed_out));
  return OkStatus();
}

}  // namespace inferx::components
