#include "inferx/models/components/attention.h"

#include <cmath>
#include <optional>
#include <string_view>

#include "inferx/core/shape.h"
#include "inferx/models/model.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/linear.h"
#include "inferx/ops/rms_norm.h"
#include "inferx/ops/rotary.h"

namespace inferx::components {

Status RunAttention(const AttentionConfig& a, const AttentionWeights& w,
                    const Tensor& normed, float norm_eps, const AttentionBatch& batch,
                    const PagedKvState& kv_state, const KvBlockPool& pool,
                    AttentionWorkspace& ws, Tensor* packed_buffer,
                    ops::OpContext& ctx, Tensor& mixed_out) {
  const int64_t rows = normed.Dim(0);

  // Project, optionally bias, normalize q/k per head, rotate, cache, attend.
  const int64_t query_width = a.query_heads * a.head_dim;
  const int64_t kv_width = a.kv_heads * a.head_dim;
  const bool gated = a.output_gate == components::OutputGate::kSigmoid;
  INFERX_ASSIGN_OR_RETURN(Tensor q_flat, ws.query->Slice(0, rows * query_width));
  INFERX_ASSIGN_OR_RETURN(Tensor q, q_flat.Reshape(Shape({rows, query_width})));
  INFERX_ASSIGN_OR_RETURN(Tensor k_flat, ws.key->Slice(0, rows * kv_width));
  INFERX_ASSIGN_OR_RETURN(Tensor k, k_flat.Reshape(Shape({rows, kv_width})));
  INFERX_ASSIGN_OR_RETURN(Tensor v_flat, ws.value->Slice(0, rows * kv_width));
  INFERX_ASSIGN_OR_RETURN(Tensor v, v_flat.Reshape(Shape({rows, kv_width})));
  // One fused GEMM over the packed QKV weight, then a column de-interleave.
  // A gated output doubles the query rows: [query | gate | key | value].
  const int64_t width = (gated ? 2 * query_width : query_width) + 2 * kv_width;
  INFERX_ASSIGN_OR_RETURN(auto flat, packed_buffer->Slice(0, rows * width));
  INFERX_ASSIGN_OR_RETURN(auto packed, flat.Reshape(Shape({rows, width})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, w.packed_qkv, packed));
  if (w.qkv_bias.has_value()) {
    INFERX_RETURN_IF_ERROR(ops::AddBias(ctx, packed, *w.qkv_bias, packed));
  }
  std::optional<Tensor> gate;
  if (gated) {
    INFERX_ASSIGN_OR_RETURN(Tensor gate_flat, ws.gate->Slice(0, rows * query_width));
    INFERX_ASSIGN_OR_RETURN(Tensor gate2d, gate_flat.Reshape(Shape({rows, query_width})));
    gate = std::move(gate2d);
  }
  INFERX_RETURN_IF_ERROR(ops::SplitProjection(ctx, packed, q, gate ? &*gate : nullptr, k, v));
  const bool fused_norm_rope =
      a.qk_norm && !a.qk_norm_plus_one && a.head_dim == 128 && ctx.Device().IsCuda();
  if (a.qk_norm && !fused_norm_rope) {
    const ops::RMSNormConfig head_norm{norm_eps, a.qk_norm_plus_one, !a.qk_norm_plus_one};
    INFERX_ASSIGN_OR_RETURN(Tensor q_heads, q.Reshape(Shape({rows * a.query_heads, a.head_dim})));
    INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, q_heads, *w.query_norm, q_heads, head_norm));
    INFERX_ASSIGN_OR_RETURN(Tensor k_heads, k.Reshape(Shape({rows * a.kv_heads, a.head_dim})));
    INFERX_RETURN_IF_ERROR(ops::RmsNorm(ctx, k_heads, *w.key_norm, k_heads, head_norm));
  }
  INFERX_ASSIGN_OR_RETURN(Tensor q3, q.Reshape(Shape({rows, a.query_heads, a.head_dim})));
  INFERX_ASSIGN_OR_RETURN(Tensor k3, k.Reshape(Shape({rows, a.kv_heads, a.head_dim})));
  const ops::RotaryParams rotary = a.rotary.Params();
  if (fused_norm_rope) {
    INFERX_RETURN_IF_ERROR(ops::NormalizeAndApplyRope(ctx, q3, k3, *w.query_norm, *w.key_norm,
                                                      batch.positions, norm_eps, rotary));
  } else {
    INFERX_RETURN_IF_ERROR(ops::ApplyRope(ctx, q3, k3, batch.positions, rotary));
  }

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
  // Gemma derives the scale from query_pre_attn_scalar rather than head_dim.
  params.scale = a.scale_override > 0.0f
                     ? a.scale_override
                     : 1.0f / std::sqrt(static_cast<float>(a.head_dim));
  params.sliding_window = a.sliding_window;
  if (w.sinks.has_value()) params.sinks = &*w.sinks;
  INFERX_RETURN_IF_ERROR(ops::PagedAttention(ctx, q, batch.qo_indptr, batch.kv_indptr,
                                             batch.kv_indices, batch.last_page_len,
                                             batch.host_qo_indptr, batch.num_seqs, key_cache,
                                             value_cache, block_size, params, ws.plan, attn_out));
  if (gate.has_value()) {
    // Qwen3-Next gates the attention output elementwise before the projection.
    INFERX_RETURN_IF_ERROR(ops::MulSigmoidGate(ctx, attn_out, *gate));
  }
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, attn_out, w.output.weight, mixed_out));
  if (w.output_bias.has_value()) {
    INFERX_RETURN_IF_ERROR(ops::AddBias(ctx, mixed_out, *w.output_bias, mixed_out));
  }
  return OkStatus();
}

}  // namespace inferx::components
