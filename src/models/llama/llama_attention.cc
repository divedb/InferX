#include "inferx/models/llama/llama_attention.h"

#include <cmath>
#include <utility>

#include "inferx/ops/elementwise.h"
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

  Attention::Geometry geometry;
  geometry.num_heads = local_heads;
  geometry.num_kv_heads = local_kv_heads;
  geometry.head_dim = head_dim;
  geometry.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  geometry.sliding_window = config.sliding_window;
  INFERX_ASSIGN_OR_RETURN(Attention core, Attention::Create(geometry));

  return LlamaAttention(std::move(qkv_proj), RotaryEmbedding(config.rope),
                        std::move(o_proj), std::move(core), local_heads, local_kv_heads,
                        head_dim, parallel.tensor_parallel_rank);
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

  INFERX_ASSIGN_OR_RETURN(Tensor attn_out,
                          Tensor::Empty(dtype, Shape({tokens, q_width}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(
      core_.Forward(ctx.op, ctx.attn, ctx.kv, ctx.layer, ctx.plan, positions, q, k,
                     v, attn_out));

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
