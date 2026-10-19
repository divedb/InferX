#include "inferx/models/qwen3/qwen3_attention.h"

#include <cmath>
#include <utility>

#include "inferx/core/status.h"
#include "inferx/ops/elementwise.h"
#include "inferx/dist/shard.h"
#include "inferx/ops/attention/types.h"
#include "inferx/ops/rotary.h"

namespace inferx {
namespace models {
namespace qwen3 {

StatusOr<Qwen3Attention> Qwen3Attention::Create(const Qwen3Config& config,
                                                const ParallelConfig& parallel,
                                                DataType dtype, Device device) {
  const int64_t tp = parallel.tensor_parallel_size;
  if (tp <= 0 || config.num_attention_heads % tp != 0) {
    return InvalidArgumentError("attention heads (", config.num_attention_heads,
                                ") must divide tensor-parallel size ", tp);
  }
  if (config.num_key_value_heads % tp != 0) {
    return InvalidArgumentError("kv heads (", config.num_key_value_heads,
                                ") must divide tensor-parallel size ", tp);
  }
  const int64_t local_heads = config.num_attention_heads / tp;
  const int64_t local_kv_heads = config.num_key_value_heads / tp;
  const int64_t hidden = config.hidden_size;
  const int64_t head_dim = config.head_dim;

  const int64_t qkv_rows = (local_heads + 2 * local_kv_heads) * head_dim;
  INFERX_ASSIGN_OR_RETURN(Linear qkv_proj, Linear::Create(hidden, qkv_rows, dtype, device));
  INFERX_ASSIGN_OR_RETURN(Linear o_proj, Linear::Create(local_heads * head_dim, hidden,
                                                        dtype, device));

  INFERX_ASSIGN_OR_RETURN(RMSNorm q_norm,
                          RMSNorm::Create(head_dim, config.rms_norm_eps, dtype, device));
  INFERX_ASSIGN_OR_RETURN(RMSNorm k_norm,
                          RMSNorm::Create(head_dim, config.rms_norm_eps, dtype, device));

  const int kv_replicate = static_cast<int>(
      std::max<int64_t>(1, parallel.tensor_parallel_size / config.num_key_value_heads));
  return Qwen3Attention(std::move(qkv_proj), std::move(q_norm), std::move(k_norm),
                        RotaryEmbedding(config.rotary_params), std::move(o_proj), local_heads,
                        local_kv_heads, head_dim, config.sliding_window.value_or(0),
                        parallel.tensor_parallel_rank, hidden,
                        config.num_attention_heads * head_dim,
                        config.num_key_value_heads * head_dim, kv_replicate);
}

std::vector<ParamDecl> Qwen3Attention::DescribeParams(const std::string& prefix) const {
  // Fused QKV: three checkpoint tensors, one packed destination. The K/V
  // pieces replicate across kv_replicate ranks (KV heads when tp exceeds
  // num_key_value_heads); fused placement stays model-side (BindWeights).
  const auto split_rows = [&](int64_t global_rows, int replicate) {
    ShardSpec shard;
    shard.kind = ShardSpec::Kind::kSplit;
    shard.split = SplitSpec{Axis::kTensorParallel, 0, replicate};
    ParamDecl p;
    p.global_shape = ShardShape{global_rows, hidden_size_};
    p.rank = 2;
    p.shard = shard;
    return p;
  };

  std::vector<ParamDecl> params;
  ParamDecl q = split_rows(global_q_rows_, 1);
  q.name = prefix + ".q_proj.weight";
  params.push_back(std::move(q));
  ParamDecl k = split_rows(global_kv_rows_, kv_replicate_);
  k.name = prefix + ".k_proj.weight";
  params.push_back(std::move(k));
  ParamDecl v = split_rows(global_kv_rows_, kv_replicate_);
  v.name = prefix + ".v_proj.weight";
  params.push_back(std::move(v));

  // q_norm/k_norm normalize over head_dim, not hidden.
  const auto replicated = [&](std::string_view name, Tensor weight) {
    ParamDecl p;
    p.name = name;
    p.global_shape = ShardShape{head_dim_};
    p.rank = 1;
    p.dst = std::move(weight);
    return p;
  };
  params.push_back(replicated(prefix + ".q_norm.weight", q_norm_.Weight()));
  params.push_back(replicated(prefix + ".k_norm.weight", k_norm_.Weight()));

  // o_proj holds this rank's column block of [hidden, qh * hd].
  ParamDecl o_proj;
  o_proj.name = prefix + ".o_proj.weight";
  o_proj.global_shape = ShardShape{hidden_size_, global_q_rows_};
  o_proj.rank = 2;
  ShardSpec cols;
  cols.kind = ShardSpec::Kind::kSplit;
  cols.split = SplitSpec{Axis::kTensorParallel, 1, 1};
  o_proj.shard = cols;
  o_proj.dst = o_proj_.Weight();
  params.push_back(std::move(o_proj));
  return params;
}

attn::AttentionSpec Qwen3Attention::Spec() const {
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

Status Qwen3Attention::Forward(ForwardContext& ctx, const Tensor& positions,
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

  // Per-head norms see [tokens * heads, head_dim]; rope wants the head
  // structure back, so reshape both ways -- views only, no copies. The
  // fused kernel normalizes and rotates in place, so the rank-2 q/k views
  // handed to attention already carry the rotated values.
  INFERX_ASSIGN_OR_RETURN(
      Tensor q_by_head, q.Reshape(Shape({tokens, local_heads_, head_dim_})));
  INFERX_ASSIGN_OR_RETURN(
      Tensor k_by_head, k.Reshape(Shape({tokens, local_kv_heads_, head_dim_})));
  INFERX_RETURN_IF_ERROR(ops::NormalizeAndApplyRope(
      ctx.op, q_by_head, k_by_head, q_norm_.Weight(), k_norm_.Weight(), positions,
      q_norm_.Eps(), rope_.Params()));

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

Status Qwen3Attention::BindWeights(const std::string& prefix, WeightMap& weights) {
  const int64_t q_rows = local_heads_ * head_dim_;
  const int64_t kv_rows = local_kv_heads_ * head_dim_;
  Tensor* qkv_weight = qkv_proj_.MutableWeight();

  // The fused buffer packs [q | k | v]; each projection's rank shard lands
  // at its own row offset of the same destination.
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".q_proj.weight",
      WeightBinding::ShardRows(qkv_weight, q_rows, tp_rank_, /*dest_row=*/0)));
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".k_proj.weight",
      WeightBinding::ShardRows(qkv_weight, kv_rows, tp_rank_, /*dest_row=*/q_rows)));
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".v_proj.weight",
      WeightBinding::ShardRows(qkv_weight, kv_rows, tp_rank_,
                               /*dest_row=*/q_rows + kv_rows)));

  INFERX_RETURN_IF_ERROR(weights.Add(prefix + ".q_norm.weight",
                                     WeightBinding::Replicate(q_norm_.MutableWeight())));
  INFERX_RETURN_IF_ERROR(weights.Add(prefix + ".k_norm.weight",
                                     WeightBinding::Replicate(k_norm_.MutableWeight())));

  // o_proj consumes this rank's contiguous column block of the checkpoint's
  // [hidden, qh * hd] matrix.
  return weights.Add(prefix + ".o_proj.weight",
                     WeightBinding::ShardCols(o_proj_.MutableWeight(),
                                              local_heads_ * head_dim_, tp_rank_));
}


}  // namespace qwen3
}  // namespace models
}  // namespace inferx
