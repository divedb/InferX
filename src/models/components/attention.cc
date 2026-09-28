#include "inferx/models/components/attention.h"

#include <cmath>
#include <string_view>

#include "inferx/core/shape.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/linear.h"
#include "inferx/ops/rotary.h"
#include "inferx/ops/rms_norm.h"
#include "models/diagnostic_trace.h"

namespace inferx::components {

Status RunAttention(const AttentionConfig& a, const AttentionWeights& w,
                    const Tensor& normed, float norm_eps, const AttentionBatch& batch,
                    const PagedKvState& kv_state, const KvBlockPool& pool,
                    AttentionWorkspace& ws, Tensor* packed_buffer,
                    ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                    std::string_view prefix, Tensor& mixed_out) {
  const int64_t rows = normed.Dim(0);
  const bool tracing = trace != nullptr && trace->enabled();
  const auto write = [&](std::string_view stage, const Tensor& t) {
    if (tracing) trace->Write(std::string(prefix) + std::string(stage), t);
  };

  // Project, normalize q/k per head, rotate, cache, attend.
  const int64_t query_width = a.query_heads * a.head_dim;
  const int64_t kv_width = a.kv_heads * a.head_dim;
  INFERX_ASSIGN_OR_RETURN(Tensor q_flat, ws.query->Slice(0, rows * query_width));
  INFERX_ASSIGN_OR_RETURN(Tensor q, q_flat.Reshape(Shape({rows, query_width})));
  INFERX_ASSIGN_OR_RETURN(Tensor k_flat, ws.key->Slice(0, rows * kv_width));
  INFERX_ASSIGN_OR_RETURN(Tensor k, k_flat.Reshape(Shape({rows, kv_width})));
  INFERX_ASSIGN_OR_RETURN(Tensor v_flat, ws.value->Slice(0, rows * kv_width));
  INFERX_ASSIGN_OR_RETURN(Tensor v, v_flat.Reshape(Shape({rows, kv_width})));
  // One fused GEMM over the packed QKV weight, then a column de-interleave.
  const int64_t width = query_width + 2 * kv_width;
  INFERX_ASSIGN_OR_RETURN(auto flat, packed_buffer->Slice(0, rows * width));
  INFERX_ASSIGN_OR_RETURN(auto packed, flat.Reshape(Shape({rows, width})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.packed_qkv, packed));
  INFERX_RETURN_IF_ERROR(ops::SplitQkv(ctx, packed, q, k, v));
  write("q_proj", q);
  write("k_proj", k);
  write("v_proj", v);
  const bool fused_norm_rope = a.qk_norm && a.head_dim == 128 && ctx.device().IsCuda();
  if (a.qk_norm && !fused_norm_rope) {
    INFERX_ASSIGN_OR_RETURN(Tensor q_heads, q.Reshape(Shape({rows * a.query_heads, a.head_dim})));
    INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, q_heads, *w.query_norm, q_heads,
                                        ops::RMSNormConfig{norm_eps, false, true}));
    INFERX_ASSIGN_OR_RETURN(Tensor k_heads, k.Reshape(Shape({rows * a.kv_heads, a.head_dim})));
    INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, k_heads, *w.key_norm, k_heads,
                                        ops::RMSNormConfig{norm_eps, false, true}));
  }
  INFERX_ASSIGN_OR_RETURN(Tensor q3, q.Reshape(Shape({rows, a.query_heads, a.head_dim})));
  INFERX_ASSIGN_OR_RETURN(Tensor k3, k.Reshape(Shape({rows, a.kv_heads, a.head_dim})));
  if (fused_norm_rope) {
    INFERX_RETURN_IF_ERROR(ops::NormalizeAndApplyRope(ctx, q3, k3, *w.query_norm, *w.key_norm,
                                                      batch.positions, norm_eps,
                                                      ops::RotaryParams{a.rotary.dim, a.rotary.theta}));
  } else {
    INFERX_RETURN_IF_ERROR(
        ops::ApplyRope(ctx, q3, k3, batch.positions, ops::RotaryParams{a.rotary.dim, a.rotary.theta}));
  }

  write("q_rope", q);
  write("k_rope", k);
  INFERX_ASSIGN_OR_RETURN(Tensor key_cache, pool.KeyCache(kv_state.pool_layer));
  INFERX_ASSIGN_OR_RETURN(Tensor value_cache, pool.ValueCache(kv_state.pool_layer));
  const int64_t block_size = pool.BlockSize();
  INFERX_RETURN_IF_ERROR(ops::WritePagedKv(ctx, k, v, batch.positions, batch.batch_indices,
                                           batch.kv_indptr, batch.kv_indices, key_cache,
                                           value_cache, block_size));
  INFERX_ASSIGN_OR_RETURN(Tensor attn_flat, ws.attn_out->Slice(0, rows * query_width));
  INFERX_ASSIGN_OR_RETURN(Tensor attn_out, attn_flat.Reshape(Shape({rows, query_width})));
  ops::AttentionParams params;
  params.query_heads = a.query_heads;
  params.kv_heads = a.kv_heads;
  params.head_dim = a.head_dim;
  params.scale = 1.0f / std::sqrt(static_cast<float>(a.head_dim));
  params.sliding_window = a.sliding_window;
  INFERX_RETURN_IF_ERROR(ops::PagedAttention(ctx, q, batch.qo_indptr, batch.kv_indptr,
                                             batch.kv_indices, batch.last_page_len,
                                             batch.host_qo_indptr, batch.num_seqs, key_cache,
                                             value_cache, block_size, params, ws.plan, attn_out));
  write("attention", attn_out);
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, attn_out, w.output.weight, mixed_out));
  write("o_proj", mixed_out);
  return OkStatus();
}

}  // namespace inferx::components
