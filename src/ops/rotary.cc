#include "inferx/ops/rotary.h"
#include <cmath>

#include "ops/cuda/rotary.h"

namespace inferx::ops {

namespace {
Status ValidateRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params) {
  if (!q.IsDefined() || !k.IsDefined() || !positions.IsDefined()) {
    return InvalidArgumentError("ApplyRope requires defined q, k, and positions");
  }
  if (q.Rank() != 3 || k.Rank() != 3) {
    return InvalidArgumentError("ApplyRope expects rank-3 q and k of [tokens, heads, head_dim]");
  }
  if (q.GetDataType() != DataType::kBFloat16 || k.GetDataType() != DataType::kBFloat16) {
    return UnimplementedError("ApplyRope supports bfloat16 q and k, got ",
                              DataTypeName(q.GetDataType()));
  }
  if (positions.Rank() != 1 || positions.GetDataType() != DataType::kInt32) {
    return InvalidArgumentError("ApplyRope positions must be rank-1 int32");
  }
  if (q.Dim(2) != k.Dim(2) || q.Dim(0) != k.Dim(0)) {
    return InvalidArgumentError("ApplyRope q and k must share head_dim and token count");
  }
  if (positions.Dim(0) != q.Dim(0)) {
    return InvalidArgumentError("ApplyRope has ", positions.Dim(0), " positions for ",
                                q.Dim(0), " tokens");
  }
  if (k.Dim(1) <= 0 || q.Dim(1) % k.Dim(1) != 0) {
    return InvalidArgumentError("ApplyRope query heads (", q.Dim(1),
                                ") must be a multiple of kv heads (", k.Dim(1), ")");
  }
  if (params.rotary_dim <= 0 || params.rotary_dim > q.Dim(2) || params.rotary_dim % 2 != 0) {
    return InvalidArgumentError("ApplyRope rotary_dim must be even and at most head_dim");
  }
  if (!(params.theta > 0.0f)) {
    return InvalidArgumentError("ApplyRope theta must be positive");
  }
  const DeviceId device = ctx.device();
  if (q.Device() != device || k.Device() != device || positions.Device() != device) {
    return InvalidArgumentError("ApplyRope tensors must live on the context's device ",
                                device.ToString());
  }
  return OkStatus();
}
}  // namespace

Status ApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params) {
  INFERX_RETURN_IF_ERROR(ValidateRope(ctx, q, k, positions, params));
  if (q.IsEmpty()) return OkStatus();
  switch (ctx.device().kind) {
    case DeviceKind::kCuda:
      return cuda::ApplyRope(ctx, q, k, positions, params);
    default:
      return UnimplementedError("ApplyRope has no implementation for device ",
                                ctx.device().ToString());
  }
}

Status NormalizeAndApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                             const Tensor& q_weight, const Tensor& k_weight,
                             const Tensor& positions, float eps, const RotaryParams& params) {
  INFERX_RETURN_IF_ERROR(ValidateRope(ctx, q, k, positions, params));
  for (const auto* weight : {&q_weight, &k_weight}) {
    if (!weight->IsDefined() || weight->Rank() != 1 || weight->Dim(0) != q.Dim(2) ||
        weight->GetDataType() != q.GetDataType() || weight->Device() != ctx.device())
      return InvalidArgumentError("NormalizeAndApplyRope requires per-head BF16 weights");
  }
  if (!std::isfinite(eps) || eps < 0) return InvalidArgumentError("invalid normalization epsilon");
  if (q.Dim(2) != 128 || !ctx.device().IsCuda())
    return UnimplementedError("NormalizeAndApplyRope requires CUDA and head dimension 128");
  if (q.IsEmpty()) return OkStatus();
  return cuda::NormalizeAndApplyRope(ctx, q, k, q_weight, k_weight, positions, eps, params);
}

}  // namespace inferx::ops
