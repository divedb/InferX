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

/// \brief Computes a scale and quantizes in a single launch.
///
/// `ComputeF8Scale` followed by `QuantizeF16ToF8E4M3` costs four launches --
/// a memset, a reduction, a finalize and the quantize. On the decode path that
/// happens for four tensors in each of 36 layers, and at ~4.7 us a launch it
/// would eat most of what FP8 weights save. This does it in one block: a
/// grid-stride pass to find the maximum magnitude, a block reduction, then a
/// second pass to write the quantized output.
///
/// One block means one scale for the whole tensor, which is what cuBLASLt's
/// scalar A/B scale pointers want anyway. It also means large tensors are
/// slower than a multi-block reduction would be -- fine for activations, which
/// is all this is for; weights are quantized once at load where the four-launch
/// path is free.
///
/// \param src       f16 or bf16 tensor on a CUDA device.
/// \param dst       f8e4m3 tensor of the same shape.
/// \param scale_dev Device pointer to one float, receiving the dequant scale.
Status QuantizeToF8E4M3Dynamic(const TensorView& src, const TensorView& dst,
                               float* scale_dev,
                               cudaStream_t stream = nullptr);

// ---------------------------------------------------------------------------
// W4A16: per-group symmetric int4 weights.
//
// This is M8's weight format, and the reason it exists is bandwidth, not
// arithmetic: int4 weights are a quarter the bytes of fp16, which on a
// decode step (m small, weight-read-bound) is the only thing that moves the
// needle. The format is symmetric per-group: each weight w[n, k] becomes an
// int4 in [-7, 7] scaled by a per-group fp16 factor, with the group running
// along k -- the contraction dim, matching how linear-layer weights are laid
// out [n, k]. Group size 128 is the AWQ/GPTQ default and what `scales.Dim(1)
// == k/128` implies; the kernels derive the group from the shapes rather than
// taking it as a parameter so the two cannot disagree.
//
// Two int4 are packed per byte, low nibble first (element 2*i is the low
// nibble of byte i), in two's complement. The packing is the project's own --
// `DataType::kInt4` already addresses sub-byte rows this way -- so a
// checkpoint loader that produces this layout feeds these kernels directly.
//
// What is here is the dequant-then-GEMM path, deliberately: CUTLASS 4.6.1 has
// no turnkey fused W4A16 kernel for sm_89 (mixed-input is Hopper-only; the
// SM80 framework's int4 warp-fragment shuffle is unimplemented -- see
// ARCHITECTURE.md R3/M8). Dequantizing to fp16 and running the stock fp16
// GEMM is correct and is the lower bound a future fused kernel has to beat;
// it pays double bandwidth (read int4, write fp16, read fp16 again), which is
// exactly the cost fusing removes and the reason to measure it.
// ---------------------------------------------------------------------------

/// The symmetric int4 range. Scales are chosen to map a group's peak magnitude
/// onto 7, so values land in [-7, 7] and q and -q are both representable. The
/// -8 code point is unused.
inline constexpr float kInt4SymmetricMax = 7.0f;

/// \brief Quantizes an f16 weight tensor to per-group symmetric int4.
///
/// Computes one fp16 scale per group (group amax / 7) and writes the packed
/// int4 weights. `src` is `[n, k]` f16; `dst` is `[n, k]` int4 (packed
/// 2-per-byte, so its buffer is `n*k/2` bytes and `k` must be even); `scales`
/// is `[n, k/group]` f16. The group size is `k / scales.Dim(1)` and is
/// required to divide `k` evenly.
///
/// Done in two launches -- a per-group amax reduction, then the quantize+pack
/// -- because weights are quantized once at load, not per step, so the launch
/// count is free and the two passes stay simple and independently testable.
///
/// \param src     `[n, k]` f16 on a CUDA device.
/// \param dst     `[n, k]` int4, same device.
/// \param scales  `[n, k/group]` f16, same device. Overwritten.
/// \param stream  Stream to launch on. Does not synchronize.
/// \return        OK, InvalidArgument for shapes this cannot serve, or the
///                CUDA error.
Status QuantizeF16ToInt4(const TensorView& src, const TensorView& dst,
                         const TensorView& scales,
                         cudaStream_t stream = nullptr);

/// \brief Dequantizes per-group int4 weights back to f16.
///
/// The inverse of `QuantizeF16ToInt4`: unpacks each int4, multiplies by its
/// group's scale, and writes fp16. `src` is `[n, k]` int4, `scales` is
/// `[n, k/group]` f16, `dst` is `[n, k]` f16. The materialized f16 is then
/// fed to a stock `LinearF16` -- this is the unfused baseline, not a W4A16
/// kernel.
///
/// \param src     `[n, k]` int4 on a CUDA device.
/// \param scales  `[n, k/group]` f16, same device.
/// \param dst     `[n, k]` f16, same device. Overwritten.
/// \param stream  Stream to launch on. Does not synchronize.
/// \return        OK, InvalidArgument, or the CUDA error.
Status DequantizeInt4ToF16(const TensorView& src, const TensorView& scales,
                           const TensorView& dst,
                           cudaStream_t stream = nullptr);

}  // namespace inferx::kernels
