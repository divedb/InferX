#pragma once

#include <cuda_runtime_api.h>

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// The largest finite magnitude representable in e4m3 (S.EEEE.MMM, no infinity,
/// one NaN encoding). Every scale below is chosen to land a tensor's peak
/// magnitude on this value.
inline constexpr float kFloat8E4M3Max = 448.0f;

/// \brief Computes the per-tensor dequantization scale for `src`.
///
/// Writes `amax(src) / kFloat8E4M3Max` to `scale_dev`, which is the factor that
/// maps the tensor's largest magnitude onto e4m3's largest representable one.
/// Quantization divides by it; the GEMM multiplies by it again through
/// cuBLASLt's scale pointers, so the two cancel and the result is in the
/// original units.
///
/// Per-tensor rather than per-channel, deliberately, and this is the accuracy
/// limit of the M1 prototype rather than of the format: one outlier channel
/// drags the scale and costs every other channel mantissa. Per-channel scales
/// are the obvious next step and do not change this interface -- `scale_dev`
/// becomes a vector and the descriptor attribute changes.
///
/// An all-zero tensor yields a scale of 1 rather than 0, so that quantizing it
/// divides by something.
///
/// \param src       f16 tensor on a CUDA device.
/// \param scale_dev Device pointer to one float. Overwritten.
/// \param stream    Stream to launch on. Does not synchronize.
/// \return          OK, InvalidArgument, or the CUDA error.
Status ComputeF8Scale(const TensorView& src, float* scale_dev,
                      cudaStream_t stream = nullptr);

/// \brief Quantizes an f16 tensor to e4m3, dividing by `*scale_dev`.
///
/// Saturating: values beyond e4m3's range clamp to ±448 rather than becoming
/// inf or NaN, which is what makes a badly-chosen scale degrade accuracy
/// instead of poisoning the GEMM.
///
/// \param src       f16 tensor on a CUDA device.
/// \param dst       f8e4m3 tensor of the same shape and device.
/// \param scale_dev Device pointer to the scale from `ComputeF8Scale`.
/// \param stream    Stream to launch on. Does not synchronize.
/// \return          OK, InvalidArgument, or the CUDA error.
Status QuantizeF16ToF8E4M3(const TensorView& src, const TensorView& dst,
                           const float* scale_dev,
                           cudaStream_t stream = nullptr);

}  // namespace inferx::kernels
