#include "inferx/ops/gather.h"

#include <limits>

#include "inferx/ops/cuda/gather.h"

namespace inferx::ops {

Status GatherRows(ExecutionContext& ctx, const Tensor& src, const Tensor& indices,
                  Tensor& out) {
  if (src.Rank() != 2 || indices.Rank() != 1 || out.Rank() != 2) {
    return InvalidArgumentError("GatherRows expects rank-2 src/out and rank-1 indices");
  }
  if (out.GetDataType() != src.GetDataType()) {
    return InvalidArgumentError("GatherRows out dtype must match src dtype");
  }
  if (indices.GetDataType() != DataType::kInt32) {
    return InvalidArgumentError("GatherRows indices must be int32");
  }
  if (out.IsEmpty()) return OkStatus();
  if (out.Dim(1) != src.Dim(1)) {
    return InvalidArgumentError("GatherRows out width ", out.Dim(1),
                                " does not match src width ", src.Dim(1));
  }
  if (indices.Dim(0) != out.Dim(0)) {
    return InvalidArgumentError("GatherRows has ", indices.Dim(0), " indices but ", out.Dim(0),
                                " output rows");
  }
  const DeviceId device = ctx.device();
  if (src.Device() != device || indices.Device() != device || out.Device() != device) {
    return InvalidArgumentError("GatherRows tensors must live on the context's device ",
                                device.ToString());
  }
  if (src.Dim(0) > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("GatherRows src exceeds 32-bit kernel extents");
  }
  switch (device.kind) {
    case DeviceKind::kCuda:
      return cuda::GatherRows(ctx, src, indices, out);
    default:
      return UnimplementedError("GatherRows has no implementation for device ",
                                device.ToString());
  }
}

Status GatherRowsRange(ExecutionContext& ctx, const Tensor& src, const Tensor& indices,
                       Tensor& out, int64_t row_begin) {
  if (row_begin < 0) {
    return InvalidArgumentError("GatherRowsRange row offset must be non-negative");
  }
  // The plain gather already covers a shard starting at row 0 only when the
  // caller guarantees every index is inside; range semantics always apply.
  if (src.Rank() != 2 || indices.Rank() != 1 || out.Rank() != 2 ||
      out.GetDataType() != src.GetDataType() ||
      indices.GetDataType() != DataType::kInt32 ||
      out.Dim(1) != src.Dim(1) || indices.Dim(0) != out.Dim(0)) {
    return InvalidArgumentError("GatherRowsRange shapes disagree");
  }
  if (out.IsEmpty()) return OkStatus();
  const DeviceId device = ctx.device();
  if (src.Device() != device || indices.Device() != device || out.Device() != device) {
    return InvalidArgumentError("GatherRowsRange tensors must live on the context's device ",
                                device.ToString());
  }
  switch (device.kind) {
    case DeviceKind::kCuda:
      return cuda::GatherRowsRange(ctx, src, indices, out, row_begin);
    default:
      return UnimplementedError("GatherRowsRange has no implementation for device ",
                                device.ToString());
  }
}

Status CopyColumnBlock(ExecutionContext& ctx, const Tensor& src, Tensor& dst,
                       int64_t col_begin) {
  if (src.Rank() != 2 || dst.Rank() != 2 || src.Dim(0) != dst.Dim(0) ||
      src.GetDataType() != dst.GetDataType() || col_begin < 0 ||
      col_begin + src.Dim(1) > dst.Dim(1)) {
    return InvalidArgumentError("CopyColumnBlock shapes disagree");
  }
  if (src.IsEmpty()) return OkStatus();
  const DeviceId device = ctx.device();
  if (src.Device() != device || dst.Device() != device) {
    return InvalidArgumentError("CopyColumnBlock tensors must live on the context's device ",
                                device.ToString());
  }
  switch (device.kind) {
    case DeviceKind::kCuda:
      return cuda::CopyColumnBlock(ctx, src, dst, col_begin);
    default:
      return UnimplementedError("CopyColumnBlock has no implementation for device ",
                                device.ToString());
  }
}

}  // namespace inferx::ops
