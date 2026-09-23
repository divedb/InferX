#include "inferx/ops/rms_norm.h"
#include "inferx/ops/elementwise.h"

#include <limits>

#include "ops/cpu/rms_norm.h"
#include "ops/cuda/rms_norm.h"

namespace inferx::ops {

namespace {
Status ValidateNorm(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
                    const RMSNormConfig& config) {
  if (!x.IsDefined() || !weight.IsDefined() || !out.IsDefined()) {
    return InvalidArgumentError("RmsNorm requires defined x, weight, and out tensors");
  }
  if (x.Rank() != 2 || weight.Rank() != 1 || out.Rank() != 2) {
    return InvalidArgumentError(
        "RmsNorm expects rank-2 x and out and rank-1 weight, got ranks ", x.Rank(), ", ",
        weight.Rank(), ", and ", out.Rank());
  }
  const DataType dtype = x.GetDataType();
  if (dtype == DataType::kUndefined || weight.GetDataType() == DataType::kUndefined ||
      out.GetDataType() == DataType::kUndefined) {
    return InvalidArgumentError("RmsNorm requires typed x, weight, and out tensors");
  }
  if (weight.GetDataType() != dtype || out.GetDataType() != dtype) {
    return InvalidArgumentError("RmsNorm dtype mismatch: x is ", DataTypeName(dtype),
                                " but weight is ", DataTypeName(weight.GetDataType()),
                                " and out is ", DataTypeName(out.GetDataType()));
  }
  if (x.IsEmpty()) {
    return InvalidArgumentError("RmsNorm requires a non-empty x");
  }
  if (out.Dim(0) != x.Dim(0) || out.Dim(1) != x.Dim(1)) {
    return InvalidArgumentError("RmsNorm out is [", out.Dim(0), ", ", out.Dim(1),
                                "] but x is [", x.Dim(0), ", ", x.Dim(1), "]");
  }
  if (weight.Dim(0) != x.Dim(1)) {
    return InvalidArgumentError("RmsNorm weight has ", weight.Dim(0),
                                " elements but x has hidden size ", x.Dim(1));
  }
  if (config.eps < 0.0f) {
    return InvalidArgumentError("RmsNorm eps must be non-negative, got ", config.eps);
  }
  if (x.Dim(0) > std::numeric_limits<uint32_t>::max() ||
      x.Dim(1) > std::numeric_limits<uint32_t>::max()) {
    return InvalidArgumentError("RmsNorm shape exceeds 32-bit kernel extents");
  }
  const DeviceId device = ctx.device();
  if (x.Device() != device || weight.Device() != device || out.Device() != device) {
    return InvalidArgumentError("RmsNorm tensors must live on the context's device ",
                                device.ToString());
  }
  return OkStatus();
}
}  // namespace

Status RmsNorm(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const RMSNormConfig& config) {
  INFERX_RETURN_IF_ERROR(ValidateNorm(ctx, x, weight, out, config));
  switch (ctx.device().kind) {
    case DeviceKind::kCuda:
      return cuda::RmsNorm(ctx, x, weight, out, config);
    case DeviceKind::kCpu:
      return cpu::RmsNorm(ctx, x, weight, out, config);
    default:
      return UnimplementedError("RmsNorm has no implementation for device ", ctx.device().ToString());
  }
}

Status AddRmsNorm(ExecutionContext& ctx, const Tensor& x, Tensor& residual,
                  const Tensor& weight, Tensor& out, const RMSNormConfig& config) {
  INFERX_RETURN_IF_ERROR(ValidateNorm(ctx, x, weight, out, config));
  INFERX_RETURN_IF_ERROR(ValidateNorm(ctx, residual, weight, out, config));
  if (residual.Data() == out.Data())
    return InvalidArgumentError("AddRmsNorm requires distinct residual and output buffers");
  if (ctx.device().IsCuda() && config.round_before_weight &&
      x.GetDataType() == DataType::kBFloat16) {
    return cuda::AddRmsNorm(ctx, x, residual, weight, out, config);
  }
  INFERX_RETURN_IF_ERROR(Add(ctx, x, residual, residual));
  return RmsNorm(ctx, residual, weight, out, config);
}

}  // namespace inferx::ops
