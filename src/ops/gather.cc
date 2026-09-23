#include "inferx/ops/gather.h"

#include <limits>

#include "ops/cuda/gather.h"

namespace inferx::ops {

Status GatherRows(ExecutionContext& ctx, const Tensor& src, const Tensor& indices,
                  Tensor& out) {
  if (!src.IsDefined() || !indices.IsDefined() || !out.IsDefined()) {
    return InvalidArgumentError("GatherRows requires defined src, indices, and out");
  }
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

}  // namespace inferx::ops
