#include "inferx/ops/norm.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "inferx/ops/cpu/norm.h"
#include "inferx/ops/cuda/norm.h"
#include "inferx/ops/dispatch.h"

namespace inferx::ops {

namespace {

// Kernels index rows and the hidden dimension with 32-bit extents.
constexpr int64_t kMaxExtent = std::numeric_limits<int32_t>::max();

// A rank-2 [rows, hidden] activation: same dtype and device as the op, dense.
Status ValidateActivation(OpContext& ctx, const char* op, const char* name, const Tensor& t,
                          int64_t rows, int64_t hidden, DataType dtype) {
  if (t.Rank() != 2) {
    return InvalidArgumentError(op, " expects rank-2 ", name, ", got rank ", t.Rank());
  }
  if (t.Dim(0) != rows || t.Dim(1) != hidden) {
    return InvalidArgumentError(op, " ", name, " is [", t.Dim(0), ", ", t.Dim(1),
                                "] but x is [", rows, ", ", hidden, "]");
  }
  if (t.GetDataType() != dtype) {
    return InvalidArgumentError(op, " dtype mismatch: x is ", DataTypeName(dtype), " but ",
                                name, " is ", DataTypeName(t.GetDataType()));
  }
  if (t.GetDevice() != ctx.GetDevice()) {
    return InvalidArgumentError(op, " ", name, " must live on the context's device ",
                                ctx.GetDevice().ToString());
  }
  // Tensors are contiguous and row-major by construction (core/tensor.h).
  return OkStatus();
}

// A rank-1 [hidden] parameter (weight or bias) with the same checks.
Status ValidateParam(OpContext& ctx, const char* op, const char* name, const Tensor& t,
                     int64_t hidden, DataType dtype) {
  if (t.Rank() != 1) {
    return InvalidArgumentError(op, " expects rank-1 ", name, ", got rank ", t.Rank());
  }
  if (t.Dim(0) != hidden) {
    return InvalidArgumentError(op, " ", name, " has ", t.Dim(0),
                                " elements but x has hidden size ", hidden);
  }
  if (t.GetDataType() != dtype) {
    return InvalidArgumentError(op, " dtype mismatch: x is ", DataTypeName(dtype), " but ",
                                name, " is ", DataTypeName(t.GetDataType()));
  }
  if (t.GetDevice() != ctx.GetDevice()) {
    return InvalidArgumentError(op, " ", name, " must live on the context's device ",
                                ctx.GetDevice().ToString());
  }
  // Tensors are contiguous and row-major by construction (core/tensor.h).
  return OkStatus();
}

// Checks shared by every norm op. Zero rows is valid (callers return early
// via x.IsEmpty()); a zero hidden size is not.
Status ValidateRowwise(OpContext& ctx, const char* op, const Tensor& x, const Tensor& weight,
                       const Tensor& out, const NormConfig& config) {
  if (x.Rank() != 2) {
    return InvalidArgumentError(op, " expects rank-2 x, got rank ", x.Rank());
  }
  const int64_t rows = x.Dim(0);
  const int64_t hidden = x.Dim(1);
  const DataType dtype = x.GetDataType();

  if (hidden <= 0) {
    return InvalidArgumentError(op, " requires a positive hidden size, got ", hidden);
  }
  if (rows > kMaxExtent || hidden > kMaxExtent) {
    return InvalidArgumentError(op, " shape [", rows, ", ", hidden,
                                "] exceeds 32-bit kernel extents");
  }
  if (!std::isfinite(config.eps) || config.eps < 0.0f) {
    return InvalidArgumentError(op, " eps must be finite and non-negative, got ", config.eps);
  }

  INFERX_RETURN_IF_ERROR(ValidateActivation(ctx, op, "x", x, rows, hidden, dtype));
  INFERX_RETURN_IF_ERROR(ValidateActivation(ctx, op, "out", out, rows, hidden, dtype));
  return ValidateParam(ctx, op, "weight", weight, hidden, dtype);
}

using RmsNormFn = Status (*)(OpContext&, const Tensor&, const Tensor&, Tensor&,
                             const NormConfig&);
using AddRmsNormFn = Status (*)(OpContext&, const Tensor&, Tensor&, const Tensor&, Tensor&,
                                const NormConfig&);
using LayerNormFn = Status (*)(OpContext&, const Tensor&, const Tensor&,
                               const std::optional<Tensor>&, Tensor&, const NormConfig&);

const KernelTable<RmsNormFn>& RmsNormBackends() {
  static const KernelTable<RmsNormFn> table{
      {DeviceType::kCuda, &cuda::RmsNorm},
      {DeviceType::kCpu, &cpu::RmsNorm},
  };

  return table;
}

const KernelTable<AddRmsNormFn>& AddRmsNormBackends() {
  static const KernelTable<AddRmsNormFn> table{
      {DeviceType::kCuda, &cuda::AddRmsNorm},
      {DeviceType::kCpu, &cpu::AddRmsNorm},
  };

  return table;
}

const KernelTable<LayerNormFn>& LayerNormBackends() {
  static const KernelTable<LayerNormFn> table{
      {DeviceType::kCuda, &cuda::LayerNorm},
      {DeviceType::kCpu, &cpu::LayerNorm},
  };

  return table;
}

}  // namespace

Status RmsNorm(OpContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const NormConfig& config) {
  constexpr const char* kOp = "RmsNorm";

  if (config.kind != NormKind::kRms) {
    return InvalidArgumentError(kOp, ": config.kind is not kRms");
  }

  INFERX_RETURN_IF_ERROR(ValidateRowwise(ctx, kOp, x, weight, out, config));

  if (x.IsEmpty()) return OkStatus();

  return Dispatch(ctx, kOp, RmsNormBackends(), x, weight, out, config);
}

Status AddRmsNorm(OpContext& ctx, const Tensor& x, Tensor& residual, const Tensor& weight,
                  Tensor& out, const NormConfig& config) {
  constexpr const char* kOp = "AddRmsNorm";
  if (config.kind != NormKind::kRms) {
    return InvalidArgumentError(kOp, ": config.kind is not kRms");
  }
  INFERX_RETURN_IF_ERROR(ValidateRowwise(ctx, kOp, x, weight, out, config));
  INFERX_RETURN_IF_ERROR(
      ValidateActivation(ctx, kOp, "residual", residual, x.Dim(0), x.Dim(1), x.GetDataType()));

  // `residual` is updated in place and `out` is written separately. This only
  // catches identical base pointers, not partial overlap.
  if (residual.Data() == out.Data()) {
    return InvalidArgumentError(kOp, " requires distinct residual and output buffers");
  }
  if (x.IsEmpty()) return OkStatus();

  return Dispatch(ctx, kOp, AddRmsNormBackends(), x, residual, weight, out, config);
}

Status LayerNorm(OpContext& ctx, const Tensor& x, const Tensor& weight,
                 const std::optional<Tensor>& bias, Tensor& out, const NormConfig& config) {
  constexpr const char* kOp = "LayerNorm";
  if (config.kind != NormKind::kLayer) {
    return InvalidArgumentError(kOp, ": config.kind is not kLayer");
  }
  if (config.bias != bias.has_value()) {
    return InvalidArgumentError(kOp, ": config.bias and the bias tensor disagree");
  }
  INFERX_RETURN_IF_ERROR(ValidateRowwise(ctx, kOp, x, weight, out, config));
  if (bias) {
    INFERX_RETURN_IF_ERROR(ValidateParam(ctx, kOp, "bias", *bias, x.Dim(1), x.GetDataType()));
  }
  if (x.IsEmpty()) return OkStatus();

  return Dispatch(ctx, kOp, LayerNormBackends(), x, weight, bias, out, config);
}

}  // namespace inferx::ops