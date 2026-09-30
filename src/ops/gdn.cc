#include "inferx/ops/gdn.h"

#include "inferx/ops/cuda/gdn.h"

namespace inferx::ops {
namespace {

Status CheckBf16OnDevice(OpContext& ctx, std::initializer_list<const Tensor*> tensors) {
  for (const Tensor* t : tensors) {
    if (t->GetDataType() != DataType::kBFloat16 || t->Device() != ctx.Device()) {
      return InvalidArgumentError("GDN op requires bfloat16 tensors on the context device");
    }
  }
  return OkStatus();
}

}  // namespace

Status SplitGdnProjection(OpContext& ctx, const Tensor& packed, Tensor& conv_in,
                          Tensor& z, int64_t key_heads, int64_t key_dim, int64_t value_heads,
                          int64_t value_dim) {
  const int64_t nvg = value_heads / key_heads;
  if (packed.Rank() != 2 || conv_in.Rank() != 2 || z.Rank() != 2 ||
      packed.Dim(1) != key_heads * (2 * key_dim + 2 * nvg * value_dim) ||
      conv_in.Dim(1) != 2 * key_heads * key_dim + value_heads * value_dim ||
      z.Dim(1) != value_heads * value_dim || conv_in.Dim(0) != packed.Dim(0) ||
      z.Dim(0) != packed.Dim(0)) {
    return InvalidArgumentError("GDN projection split shapes disagree");
  }
  INFERX_RETURN_IF_ERROR(
      CheckBf16OnDevice(ctx, {&packed, &conv_in, &z}));
  if (!ctx.Device().IsCuda()) return UnimplementedError("GDN ops require CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::SplitGdnProjection(ctx, packed, conv_in, z, key_heads, key_dim, value_heads,
                                  value_dim);
}

Status GdnCausalConv(OpContext& ctx, Tensor& x, const Tensor& weight,
                     const Tensor& state, const Tensor& batch_indices,
                     const Tensor& qo_indptr, int64_t kernel) {
  if (x.Rank() != 2 || weight.Rank() != 2 || weight.Dim(0) != x.Dim(1) ||
      weight.Dim(1) != kernel || kernel <= 0) {
    return InvalidArgumentError("GDN conv shapes disagree");
  }
  if (state.Dim(1) != x.Dim(1) || state.Dim(2) != kernel - 1 ||
      state.GetDataType() != DataType::kFloat32) {
    return InvalidArgumentError("GDN conv state must be [slots, channels, kernel-1] float32");
  }
  INFERX_RETURN_IF_ERROR(CheckBf16OnDevice(ctx, {&x, &weight}));
  if (!ctx.Device().IsCuda()) return UnimplementedError("GDN ops require CUDA");
  if (x.IsEmpty()) return OkStatus();
  return cuda::GdnCausalConv(ctx, x, weight, state, batch_indices, qo_indptr, kernel);
}

Status GdnGates(OpContext& ctx, const Tensor& ba, const Tensor& a_log,
                const Tensor& dt_bias, Tensor& beta, Tensor& g) {
  if (ba.Rank() != 2 || a_log.Rank() != 1 || dt_bias.Rank() != 1 ||
      a_log.GetDataType() != DataType::kFloat32 ||
      dt_bias.GetDataType() != DataType::kFloat32 ||
      beta.GetDataType() != DataType::kFloat32 || g.GetDataType() != DataType::kFloat32 ||
      beta.Numel() != g.Numel() || ba.Dim(1) != 2 * a_log.Numel() ||
      dt_bias.Numel() != a_log.Numel()) {
    return InvalidArgumentError("GDN gate shapes or dtypes disagree");
  }
  if (!ctx.Device().IsCuda()) return UnimplementedError("GDN ops require CUDA");
  if (ba.IsEmpty()) return OkStatus();
  return cuda::GdnGates(ctx, ba, a_log, dt_bias, beta, g);
}

Status GdnRecurrent(OpContext& ctx, const Tensor& conv, int64_t query_width,
                    const Tensor& beta, const Tensor& g, Tensor& state,
                    const Tensor& slot_indices, const Tensor& qo_indptr,
                    const Tensor& batch_indices, Tensor& y) {
  if (state.Rank() != 4 || state.GetDataType() != DataType::kFloat32) {
    return InvalidArgumentError("GDN state must be [slots, heads, dk, dv] float32");
  }
  const int64_t dv = state.Dim(3), dk = state.Dim(2), heads = state.Dim(1);
  if (conv.Rank() != 2 || query_width <= 0 || query_width % dk != 0 ||
      conv.Dim(1) != 2 * query_width + heads * dv || y.Dim(1) != heads * dv ||
      beta.Numel() != conv.Dim(0) * heads || slot_indices.Rank() != 1 ||
      slot_indices.Dim(0) != qo_indptr.Numel() - 1 ||
      beta.GetDataType() != DataType::kFloat32 || g.GetDataType() != DataType::kFloat32) {
    return InvalidArgumentError("GDN recurrent shapes disagree");
  }
  INFERX_RETURN_IF_ERROR(CheckBf16OnDevice(ctx, {&conv, &y}));
  if (!ctx.Device().IsCuda()) return UnimplementedError("GDN ops require CUDA");
  if (conv.IsEmpty()) return OkStatus();
  return cuda::GdnRecurrent(ctx, conv, query_width, beta, g, state, slot_indices, qo_indptr,
                            batch_indices, y);
}

Status RmsNormGated(OpContext& ctx, const Tensor& y, const Tensor& z,
                    const Tensor& weight, float eps, Tensor& out) {
  if (y.Rank() != 2 || z.Rank() != 2 || weight.Rank() != 1 || out.Rank() != 2 ||
      y.Dim(1) != z.Dim(1) || out.Dim(1) != y.Dim(1) || y.Dim(0) != z.Dim(0) ||
      y.Dim(1) % weight.Dim(0) != 0) {
    return InvalidArgumentError("gated norm shapes disagree");
  }
  INFERX_RETURN_IF_ERROR(CheckBf16OnDevice(ctx, {&y, &z, &weight, &out}));
  if (!ctx.Device().IsCuda()) return UnimplementedError("GDN ops require CUDA");
  if (y.IsEmpty()) return OkStatus();
  return cuda::RmsNormGated(ctx, y, z, weight, eps, out);
}

}  // namespace inferx::ops
