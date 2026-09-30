#include "inferx/ops/elementwise.h"

#include "inferx/ops/cuda/elementwise.h"

namespace inferx::ops {
namespace {

/// Shared validation for the elementwise ops: matching shapes and dtype.
Status CheckElementwise(absl::string_view op, const Tensor& a, const Tensor& b, Tensor& out) {
  const DataType dtype = a.GetDataType();
  if (dtype != DataType::kBFloat16 && dtype != DataType::kFloat32) {
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

Status SplitQkv(OpContext& ctx, const Tensor& packed, Tensor& q, Tensor& k, Tensor& v) {
  const Tensor* tensors[] = {&packed, &q, &k, &v};
  for (const Tensor* t : tensors) {
    if (t->Rank() != 2 || t->GetDataType() != DataType::kBFloat16 ||
        t->Device() != ctx.Device() || t->Dim(0) != packed.Dim(0))
      return InvalidArgumentError("SplitQkv requires compatible BF16 matrices");
  }
  if (packed.Dim(1) != q.Dim(1) + k.Dim(1) + v.Dim(1))
    return InvalidArgumentError("SplitQkv column counts disagree");
  if (!ctx.Device().IsCuda()) return UnimplementedError("SplitQkv requires CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::SplitQkv(ctx, packed, q, k, v);
}

Status PackedSiluAndMul(OpContext& ctx, const Tensor& packed, Tensor& out) {
  if (packed.Rank() != 2 || out.Rank() != 2 ||
      packed.GetDataType() != DataType::kBFloat16 || out.GetDataType() != DataType::kBFloat16 ||
      packed.Device() != ctx.Device() || out.Device() != ctx.Device() ||
      packed.Dim(0) != out.Dim(0) || packed.Dim(1) != 2 * out.Dim(1))
    return InvalidArgumentError("PackedSiluAndMul requires compatible BF16 matrices");
  if (!ctx.Device().IsCuda()) return UnimplementedError("PackedSiluAndMul requires CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::PackedSiluAndMul(ctx, packed, out);
}

Status SplitProjection(OpContext& ctx, const Tensor& packed, Tensor& q, Tensor* gate,
                       Tensor& k, Tensor& v) {
  const Tensor* parts[] = {&packed, &q, &k, &v};
  for (const Tensor* t : parts) {
    if (t->Rank() != 2 || t->GetDataType() != DataType::kBFloat16 ||
        t->Device() != ctx.Device() || t->Dim(0) != packed.Dim(0))
      return InvalidArgumentError("SplitProjection requires compatible BF16 matrices");
  }
  const int64_t gate_width = gate == nullptr ? 0 : gate->Dim(1);
  if (gate != nullptr &&
      (gate->Rank() != 2 || gate->GetDataType() != DataType::kBFloat16 ||
       gate->Device() != ctx.Device() || gate->Dim(0) != packed.Dim(0)))
    return InvalidArgumentError("SplitProjection gate must match the packed rows");
  if (packed.Dim(1) != q.Dim(1) + gate_width + k.Dim(1) + v.Dim(1))
    return InvalidArgumentError("SplitProjection column counts disagree");
  if (!ctx.Device().IsCuda()) return UnimplementedError("SplitProjection requires CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::SplitProjection(ctx, packed, q, gate, k, v);
}

Status PackedGatedActivation(OpContext& ctx, const Tensor& packed, Tensor& out,
                             Activation act, float oai_alpha, float oai_limit) {
  if (packed.Rank() != 2 || out.Rank() != 2 ||
      packed.GetDataType() != DataType::kBFloat16 || out.GetDataType() != DataType::kBFloat16 ||
      packed.Device() != ctx.Device() || out.Device() != ctx.Device() ||
      packed.Dim(0) != out.Dim(0) || packed.Dim(1) != 2 * out.Dim(1))
    return InvalidArgumentError("PackedGatedActivation requires compatible BF16 matrices");
  if (!ctx.Device().IsCuda()) return UnimplementedError("PackedGatedActivation requires CUDA");
  if (packed.IsEmpty()) return OkStatus();
  return cuda::PackedGatedActivation(ctx, packed, out, act, oai_alpha, oai_limit);
}

Status AddBias(OpContext& ctx, const Tensor& x, const Tensor& bias, Tensor& out) {
  if (x.Rank() != 2 || bias.Rank() != 1 || out.Rank() != 2 ||
      x.GetDataType() != DataType::kBFloat16 || bias.GetDataType() != DataType::kBFloat16 ||
      out.GetDataType() != DataType::kBFloat16 || bias.Dim(0) != x.Dim(1) ||
      out.Dim(0) != x.Dim(0) || out.Dim(1) != x.Dim(1))
    return InvalidArgumentError("AddBias requires a [rows, width] matrix and [width] bias");
  if (x.Device() != ctx.Device() || bias.Device() != ctx.Device() ||
      out.Device() != ctx.Device())
    return InvalidArgumentError("AddBias tensors must live on the context's device ",
                                ctx.Device().ToString());
  if (!ctx.Device().IsCuda()) return UnimplementedError("AddBias requires CUDA");
  if (x.IsEmpty()) return OkStatus();
  return cuda::AddBias(ctx, x, bias, out);
}

Status MulSigmoidGate(OpContext& ctx, Tensor& x, const Tensor& gate) {
  if (x.Rank() != gate.Rank() || x.GetDataType() != DataType::kBFloat16 ||
      gate.GetDataType() != DataType::kBFloat16 || x.Device() != ctx.Device() ||
      gate.Device() != ctx.Device())
    return InvalidArgumentError("MulSigmoidGate requires same-shape BF16 tensors");
  for (int i = 0; i < x.Rank(); ++i) {
    if (x.Dim(i) != gate.Dim(i))
      return InvalidArgumentError("MulSigmoidGate requires same-shape BF16 tensors");
  }
  if (!ctx.Device().IsCuda()) return UnimplementedError("MulSigmoidGate requires CUDA");
  if (x.IsEmpty()) return OkStatus();
  return cuda::MulSigmoidGate(ctx, x, gate);
}

Status MulSigmoidRowGate(OpContext& ctx, Tensor& x, const Tensor& gate) {
  if (x.Rank() != 2 || gate.Rank() != 2 || gate.Dim(1) != 1 || gate.Dim(0) != x.Dim(0) ||
      x.GetDataType() != DataType::kBFloat16 || gate.GetDataType() != DataType::kBFloat16 ||
      x.Device() != ctx.Device() || gate.Device() != ctx.Device())
    return InvalidArgumentError("MulSigmoidRowGate requires [rows, width] and [rows, 1] BF16");
  if (!ctx.Device().IsCuda()) return UnimplementedError("MulSigmoidRowGate requires CUDA");
  if (x.IsEmpty()) return OkStatus();
  return cuda::MulSigmoidRowGate(ctx, x, gate);
}

Status MulScalar(OpContext& ctx, Tensor& x, float scalar) {
  if (x.GetDataType() != DataType::kBFloat16 || x.Device() != ctx.Device())
    return InvalidArgumentError("MulScalar requires a BF16 tensor on the context device");
  if (!ctx.Device().IsCuda()) return UnimplementedError("MulScalar requires CUDA");
  if (x.IsEmpty()) return OkStatus();
  return cuda::MulScalar(ctx, x, scalar);
}

Status Add(OpContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  INFERX_RETURN_IF_ERROR(CheckElementwise("Add", a, b, out));
  const DeviceId device = ctx.Device();
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

Status SiluAndMul(OpContext& ctx, const Tensor& gate, const Tensor& up, Tensor& out) {
  INFERX_RETURN_IF_ERROR(CheckElementwise("SiluAndMul", gate, up, out));
  const DeviceId device = ctx.Device();
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
