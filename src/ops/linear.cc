#include "inferx/ops/profile.h"
#include "inferx/ops/linear.h"

#include "ops/cuda/linear.h"

namespace inferx::ops {

Status Linear(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out) {
  if (!x.IsDefined() || !weight.IsDefined() || !out.IsDefined()) {
    return InvalidArgumentError("Linear requires defined x, weight, and out tensors");
  }
  if (x.Rank() != 2 || weight.Rank() != 2 || out.Rank() != 2) {
    return InvalidArgumentError("Linear expects rank-2 x, weight, and out");
  }
  const DataType dtype = x.GetDataType();
  if (dtype != DataType::kBFloat16 && dtype != DataType::kFloat) {
    return UnimplementedError("Linear supports bfloat16 and float32 activations, got ",
                              DataTypeName(dtype));
  }
  if (weight.GetDataType() != dtype || out.GetDataType() != dtype) {
    return InvalidArgumentError("Linear dtype mismatch: x is ", DataTypeName(dtype),
                                " but weight is ", DataTypeName(weight.GetDataType()),
                                " and out is ", DataTypeName(out.GetDataType()));
  }
  if (x.IsEmpty() || weight.IsEmpty()) {
    return InvalidArgumentError("Linear requires non-empty x and weight");
  }
  if (weight.Dim(1) != x.Dim(1)) {
    return InvalidArgumentError("Linear reduces over ", weight.Dim(1),
                                " features but x provides ", x.Dim(1));
  }
  if (out.Dim(0) != x.Dim(0) || out.Dim(1) != weight.Dim(0)) {
    return InvalidArgumentError("Linear out is [", out.Dim(0), ", ", out.Dim(1),
                                "] but should be [", x.Dim(0), ", ", weight.Dim(0), "]");
  }
  const DeviceId device = ctx.device();
  if (x.Device() != device || weight.Device() != device || out.Device() != device) {
    return InvalidArgumentError("Linear tensors must live on the context's device ",
                                device.ToString());
  }
  switch (device.kind) {
    case DeviceKind::kCuda:
      return ProfileCall(ctx, "linear", [&] { return cuda::Linear(ctx, x, weight, out); });
    default:
      return UnimplementedError("Linear has no implementation for device ", device.ToString());
  }
}

}  // namespace inferx::ops
