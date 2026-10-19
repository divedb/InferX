#include "inferx/models/llama/llama_attention.h"

#include <cmath>
#include <utility>

#include "inferx/ops/elementwise.h"
#include "inferx/ops/attention/types.h"
#include "inferx/ops/rotary.h"

namespace inferx {
namespace models {
namespace llama {

StatusOr<LlamaAttention> LlamaAttention::Create(const LlamaConfig& config,
                                                const ParallelConfig& parallel,
                                                DataType dtype, Device device) {
  const int64_t tp = parallel.tensor_parallel_size;
  INFERX_ASSIGN_OR_RETURN(const int64_t local_heads, config.LocalHeads(tp));
  INFERX_ASSIGN_OR_RETURN(const int64_t local_kv_heads, config.LocalKvHeads(tp));
  const int64_t head_dim = config.head_dim;

  const int64_t qkv_rows = (local_heads + 2 * local_kv_heads) * head_dim;
  INFERX_ASSIGN_OR_RETURN(Linear qkv_proj,
                          Linear::Create(config.hidden_size, qkv_rows, dtype, device));
  INFERX_ASSIGN_OR_RETURN(
      Linear o_proj, Linear::Create(local_heads * head_dim, config.hidden_size, dtype,
                                    device));

  return LlamaAttention(std::move(qkv_proj), RotaryEmbedding(config.rope),
                        std::move(o_proj), local_heads, local_kv_heads, head_dim,
                        config.sliding_window, parallel.tensor_parallel_rank);
}

attn::AttentionSpec LlamaAttention::Spec() const {
  attn::AttentionSpec spec;
  spec.params = attn::DenseParams{.num_q_heads = static_cast<std::int32_t>(local_heads_),
                                  .num_kv_heads = static_cast<std::int32_t>(local_kv_heads_),
                                  .head_dim = static_cast<std::int32_t>(head_dim_)};
  spec.q_dtype = DataType::kBFloat16;
  spec.kv_dtype = DataType::kBFloat16;
  spec.kv_layout = attn::KvPageLayout::kKvPlanarNhd;  // What WritePagedKv writes.
  if (sliding_window_ > 0) {
    spec.mask = attn::MaskSpec{attn::MaskKind::kSlidingWindow,
                               static_cast<std::int32_t>(sliding_window_), 0};
  }
  return spec;
}

Status LlamaAttention::Forward(ForwardContext& ctx, const Tensor& positions,
                               const Tensor& hidden, Tensor& out) const {
  const int64_t tokens = hidden.Dim(0);
  const DataType dtype = hidden.GetDataType();
  const int64_t q_width = local_heads_ * head_dim_;
  const int64_t kv_width = local_kv_heads_ * head_dim_;

  INFERX_ASSIGN_OR_RETURN(Tensor qkv,
                          Tensor::Empty(dtype, Shape({tokens, q_width + 2 * kv_width}),
                                        &ctx.scratch));
  INFERX_ASSIGN_OR_RETURN(Tensor q,
                          Tensor::Empty(dtype, Shape({tokens, q_width}), &ctx.scratch));
  INFERX_ASSIGN_OR_RETURN(Tensor k,
                          Tensor::Empty(dtype, Shape({tokens, kv_width}), &ctx.scratch));
  INFERX_ASSIGN_OR_RETURN(Tensor v,
                          Tensor::Empty(dtype, Shape({tokens, kv_width}), &ctx.scratch));

  INFERX_RETURN_IF_ERROR(qkv_proj_.Forward(ctx.op, hidden, qkv));
  INFERX_RETURN_IF_ERROR(ops::SplitQkv(ctx.op, qkv, q, k, v));

  // Rope in place over the head views; the rank-2 views handed onward see
  // the rotated bytes.
  INFERX_ASSIGN_OR_RETURN(Tensor q_by_head,
                          q.Reshape(Shape({tokens, local_heads_, head_dim_})));
  INFERX_ASSIGN_OR_RETURN(Tensor k_by_head,
                          k.Reshape(Shape({tokens, local_kv_heads_, head_dim_})));
  INFERX_RETURN_IF_ERROR(
      ops::ApplyRope(ctx.op, q_by_head, k_by_head, positions, rope_.Params()));

  // Attention runs through the bound op: [T, H, D] views over the rank-2
  // projections (no copies), fused KV write plus attention per the kernel.
  INFERX_ASSIGN_OR_RETURN(
      Tensor q_by_dim, q.Reshape(Shape({tokens, local_heads_, head_dim_})));
  INFERX_ASSIGN_OR_RETURN(
      Tensor k_by_dim, k.Reshape(Shape({tokens, local_kv_heads_, head_dim_})));
  INFERX_ASSIGN_OR_RETURN(
      Tensor v_by_dim, v.Reshape(Shape({tokens, local_kv_heads_, head_dim_})));
  INFERX_ASSIGN_OR_RETURN(Tensor attn_out,
                          Tensor::Empty(dtype, Shape({tokens, q_width}), &ctx.scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor out_by_dim, attn_out.Reshape(Shape({tokens, local_heads_, head_dim_})));

  attn::Inputs in;
  in.q = attn::ViewOf(q_by_dim);
  in.k = attn::ViewOf(k_by_dim);
  in.v = attn::ViewOf(v_by_dim);
  INFERX_RETURN_IF_ERROR(
      ctx.Attention(ctx.layer).Forward(ctx.step, in, attn::ViewOf(out_by_dim)));

  INFERX_RETURN_IF_ERROR(o_proj_.Forward(ctx.op, attn_out, out));
  return ctx.comm.AllReduceSum(ctx.op, out);
}

Status LlamaAttention::BindWeights(const std::string& prefix, WeightMap& weights) {
  const int64_t q_rows = local_heads_ * head_dim_;
  const int64_t kv_rows = local_kv_heads_ * head_dim_;
  Tensor* qkv_weight = qkv_proj_.MutableWeight();

  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".q_proj.weight",
      WeightBinding::ShardRows(qkv_weight, q_rows, tp_rank_, 0)));
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".k_proj.weight",
      WeightBinding::ShardRows(qkv_weight, kv_rows, tp_rank_, q_rows)));
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".v_proj.weight",
      WeightBinding::ShardRows(qkv_weight, kv_rows, tp_rank_, q_rows + kv_rows)));
  return weights.Add(prefix + ".o_proj.weight",
                     WeightBinding::ShardCols(o_proj_.MutableWeight(),
                                              local_heads_ * head_dim_, tp_rank_));
}


}  // namespace llama
}  // namespace models
}  // namespace inferx
