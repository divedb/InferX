#include "inferx/core/tensor_io.h"

#include <cstring>

#include "inferx/core/device_runtime.h"

namespace inferx {

Status CopyTensor(const Tensor& src, const Tensor& dst) {
  if (!src.IsDefined() || !dst.IsDefined() || src.GetDataType() != dst.GetDataType() ||
      src.GetShape() != dst.GetShape()) {
    return InvalidArgumentError("copy requires matching shapes and dtypes");
  }
  if (src.Numel() == 0) return OkStatus();
  if (src.IsCpu() && dst.IsCpu()) {
    std::memmove(dst.Data(), src.Data(), src.NBytes());
    return OkStatus();
  }
  if (!src.IsCpu() && !dst.IsCpu() && src.Device() != dst.Device()) {
    return UnimplementedError("cross-device copy requires an explicit transfer");
  }
  const auto device = src.IsCpu() ? dst.Device() : src.Device();
  INFERX_ASSIGN_OR_RETURN(auto* runtime, RuntimeFor(device));
  const auto kind = src.IsCpu()   ? CopyKind::kHostToDevice
                    : dst.IsCpu() ? CopyKind::kDeviceToHost
                                  : CopyKind::kDeviceToDevice;
  return runtime->Copy(dst.Data(), src.Data(), src.NBytes(), kind);
}

StatusOr<Tensor> TensorFromHost(absl::Span<const float> values, const Shape& shape,
                                DeviceId device) {
  if (shape.Numel() != static_cast<int64_t>(values.size())) {
    return InvalidArgumentError("host data size does not match shape");
  }
  INFERX_ASSIGN_OR_RETURN(auto out, Tensor::Empty(DataType::kFloat, shape, device));
  INFERX_ASSIGN_OR_RETURN(auto src, Tensor::FromBlob(const_cast<float*>(values.data()),
                                                     DataType::kFloat, shape, DeviceId::Cpu()));
  INFERX_RETURN_IF_ERROR(CopyTensor(src, out));
  return out;
}

Status TensorToHost(const Tensor& src, absl::Span<float> values) {
  if (src.GetDataType() != DataType::kFloat ||
      src.Numel() != static_cast<int64_t>(values.size())) {
    return InvalidArgumentError("host output must match f32 tensor size");
  }
  INFERX_ASSIGN_OR_RETURN(auto dst, Tensor::FromBlob(values.data(), DataType::kFloat,
                                                     src.GetShape(), DeviceId::Cpu()));
  return CopyTensor(src, dst);
}

}  // namespace inferx
