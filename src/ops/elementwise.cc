#include "inferx/ops/elementwise.h"

#include "ops/cuda/elementwise.h"

namespace inferx::ops {
namespace {

/// Shared validation for the elementwise ops: matching shapes and dtype.
Status CheckElementwise(absl::string_view op, const Tensor& a, const Tensor& b, Tensor& out) {
  if (!a.IsDefined() || !b.IsDefined() || !out.IsDefined()) {
    return InvalidArgumentError(op, " requires defined a, b, and out tensors");
  }
  const DataType dtype = a.GetDataType();
  if (dtype != DataType::kBFloat16 && dtype != DataType::kFloat) {
    return UnimplementedError(op, " supports bfloat16 and float32, got ", DataTypeName(dtype));
  }
  if (b.GetDataType() != dtype || out.GetDataType() != dtype) {
    return InvalidArgumentError(op, " dtype mismatch between a, b, and out");
  }
  if (a.Rank() != b.Rank() || out.Rank() != a.Rank()) {
    return InvalidArgumentError(op, " requires a, b, and out of equal rank");
  }
  for (int i = 0; i < a.Rank(); ++i) {
    if (a.Dim(i) != b.Dim(i) || out.Dim(i) != a.Dim(i)) {
      return InvalidArgumentError(op, " requires a, b, and out of equal shape");
    }
  }
  return OkStatus();
}

}  // namespace

Status SplitQkv(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor& k, Tensor& v) {
  const Tensor* tensors[] = {&packed, &q, &k, &v};
  for (const Tensor* t : tensors) {
    if (!t->IsDefined() || t->Rank() != 2 || t->GetDataType() != DataType::kBFloat16 ||
        t->Device() != ctx.device() || t->Dim(0) != packed.Dim(0))
      return InvalidArgumentError("SplitQkv requires compatible BF16 matrices");
  }
  if (packed.Dim(1) != q.Dim(1) + k.Dim(1) + v.Dim(1))
    return InvalidArgumentError("SplitQkv column counts disagree");
  if (!ctx.device().IsCuda()) return UnimplementedError("SplitQkv requires CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::SplitQkv(ctx, packed, q, k, v);
}

Status PackedSiluAndMul(ExecutionContext& ctx, const Tensor& packed, Tensor& out) {
  if (!packed.IsDefined() || !out.IsDefined() || packed.Rank() != 2 || out.Rank() != 2 ||
      packed.GetDataType() != DataType::kBFloat16 || out.GetDataType() != DataType::kBFloat16 ||
      packed.Device() != ctx.device() || out.Device() != ctx.device() ||
      packed.Dim(0) != out.Dim(0) || packed.Dim(1) != 2 * out.Dim(1))
    return InvalidArgumentError("PackedSiluAndMul requires compatible BF16 matrices");
  if (!ctx.device().IsCuda()) return UnimplementedError("PackedSiluAndMul requires CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::PackedSiluAndMul(ctx, packed, out);
}

Status Add(ExecutionContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  INFERX_RETURN_IF_ERROR(CheckElementwise("Add", a, b, out));
  const DeviceId device = ctx.device();
  if (a.Device() != device || b.Device() != device || out.Device() != device) {
    return InvalidArgumentError("Add tensors must live on the context's device ",
                                device.ToString());
  }
  if (a.IsEmpty()) return OkStatus();
  switch (device.kind) {
    case DeviceKind::kCuda:
      return cuda::Add(ctx, a, b, out);
    default:
      return UnimplementedError("Add has no implementation for device ", device.ToString());
  }
}

Status SiluAndMul(ExecutionContext& ctx, const Tensor& gate, const Tensor& up, Tensor& out) {
  INFERX_RETURN_IF_ERROR(CheckElementwise("SiluAndMul", gate, up, out));
  const DeviceId device = ctx.device();
  if (gate.Device() != device || up.Device() != device || out.Device() != device) {
    return InvalidArgumentError("SiluAndMul tensors must live on the context's device ",
                                device.ToString());
  }
  if (gate.IsEmpty()) return OkStatus();
  switch (device.kind) {
    case DeviceKind::kCuda:
      return cuda::SiluAndMul(ctx, gate, up, out);
    default:
      return UnimplementedError("SiluAndMul has no implementation for device ",
                                device.ToString());
  }
}

}  // namespace inferx::ops
