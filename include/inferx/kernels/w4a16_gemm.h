#pragma once

#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// \brief Fused W4A16 GEMM: `y = x · dequant(w_int4, scales)^T`.
///
/// `x` is `[m, k]` bf16, `w_int4` is `[n, k]` packed int4 (two-per-byte, low
/// nibble first, two's complement -- the format `QuantizeBf16ToInt4` writes),
/// `scales` is `[n, k/group]` bf16, `y` is `[m, n]` bf16. fp32 accumulate.
///
/// This is the fused dequant-then-GEMM: each int4 weight element is read from
/// global memory exactly once and dequantized in registers, rather than
/// materializing the whole `[n, k]` bf16 weight and reading it back. On the
/// weight-bandwidth-bound decode step (small `m`) that is the difference
/// between reading 1.54 GB (int4) and 12.3 GB (int4 dequanted to bf16 then read
/// by a bf16 GEMM) per token -- which is the whole point of W4A16.
///
/// CUTLASS 4.6.1's mixed-input collectives are SM90/SM100 only and cuBLASLt has
/// no int4 x bf16 GEMM, so on sm_89 this is hand-rolled. The kernel is aimed at
/// the bandwidth-bound regime; large-`m` (compute-bound) prefill is better
/// served by the unfused dequant + a tensor-core GEMM, which the caller can
/// route to by shape.
Status W4A16Gemm(const TensorView& x, const TensorView& w_int4,
                 const TensorView& scales, const TensorView& y, int64_t group,
                 Stream stream = {});

}  // namespace inferx::kernels
