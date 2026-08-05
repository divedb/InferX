#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// \brief Fused MXFP4-weight GEMM: bf16 activations x MXFP4 weights -> bf16.
///
/// Reads the 4-bit MXFP4 expert weights directly in the mainloop and dequantizes
/// in registers, avoiding the 1.59 GB/layer bf16 scratch the unfused path needs.
/// That scratch was the VRAM killer (13.8 GB resident experts + 1.59 GB scratch
/// left no room for a real KV pool); fusing removes it.
///
/// The kernel is a GEMV-style warp-per-column design, adapted from
/// `W4A16GemmVectorKernel`. It uses scalar fp32 FMAs rather than tensor cores:
/// sm_89 (Ada) has no mixed-input TCs (Hopper does), so a 4-bit GEMM must
/// dequantize to 16-bit before any TC could help, and the dequant-to-SMEM-then-
/// TC path is what the unfused baseline already does. The reason this can still
/// beat cuBLASLt bf16 is that MXFP4's dequant is cheap -- a 16-entry LUT lookup
/// plus an exponent bit-shift, ~3 ops/element, versus int4's sign-extend and
/// bf16 scale multiply at ~5-6 -- and the weight read is 4x narrower. On
/// weight-bandwidth-bound decode that trade pays.
///
/// Like W4A16Gemm, this serves the small-M decode/Prefill-expert case (the
/// activation rows must fit in shared memory). The scalar fallback handles
/// larger M.
///
/// \param x       `[m, k]` bf16 activations.
/// \param blocks  `[n, k/2]` u8 -- the packed MXFP4 nibbles, flattened from the
///                checkpoint's rank-3 `*_blocks` tensors.
/// \param scales  `[n, k/32]` u8 -- the E8M0 per-block-of-32 scales.
/// \param y       `[m, n]` bf16 output.
/// \param deinterleave  True for gate_up weights, whose `n = 2*inter` axis
///                      alternates gate/up rows in the checkpoint. The kernel
///                      remaps output columns so the result lands in `[gate | up]`
///                      split order, which the activation expects.
Status Mxfp4Gemm(const TensorView& x, const TensorView& blocks,
                 const TensorView& scales, const TensorView& y,
                 bool deinterleave = false, cudaStream_t stream = nullptr);

}  // namespace inferx::kernels
