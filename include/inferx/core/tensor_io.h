#pragma once

#include "inferx/core/tensor.h"

namespace inferx {

// Synchronous copies. Callers order these against any outstanding stream work.
Status CopyTensor(const Tensor& src, const Tensor& dst);
StatusOr<Tensor> TensorFromHost(absl::Span<const float> values, const Shape& shape,
                                DeviceId device);
Status TensorToHost(const Tensor& src, absl::Span<float> values);

}  // namespace inferx
