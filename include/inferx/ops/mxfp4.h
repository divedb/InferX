#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor_view.h"

/// MXFP4 — the OCP microscaling 4-bit format gpt-oss-20b stores its expert
/// weights in.
///
/// The format, pinned bitwise against HuggingFace's own decoder by
/// `scripts/gen_mxfp4_golden.py` and not to be re-derived from a specification:
///
///   * **32 values share one block.** `blocks` is `[rows, G, 16]` uint8 — 16
///     bytes holding 32 nibbles — and `scales` is `[rows, G]` uint8, one per
///     block.
///   * **The scale is E8M0**: a bare exponent, bias 127, so the block
///     multiplier is `2^(scale − 127)`. No mantissa, so applying it is an
///     exponent add rather than a multiply.
///   * **A value is FP4 E2M1**, looked up in
///     `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
///   * **The low nibble comes first.** Byte `i` holds element `2i` in its low
///     four bits and element `2i+1` in its high four. This is the detail with
///     no answer from first principles, and getting it backwards produces
///     weights that are wrong and still look like a language model.
///
/// **Dequantization to bf16 is exact.** Every FP4 value needs at most one
/// explicit mantissa bit and bf16 has seven, and scaling by a power of two
/// moves the exponent without touching the mantissa — so the bf16 output equals
/// the fp32 arithmetic value bit for bit, and the tests assert equality rather
/// than a tolerance. The one way to leave that range is an absurd scale:
/// `2^(255−127)` overflows bf16's exponent exactly as it overflows fp32's. Real
/// checkpoints are nowhere near it.
///
/// What this header does *not* do is make MXFP4 fast. These kernels dequantize
/// to bf16 and hand the result to an ordinary GEMM, which is the
/// double-touch pattern R3 already measured as a lower bound for W4A16: read
/// 4-bit, write bf16, read bf16 again. It is correct, it is what the golden
/// test checks, and a mainloop that reads MXFP4 directly is the thing that
/// eventually replaces it.
namespace inferx::ops {

/// \brief `out[r, :] = decode(blocks[r], scales[r])`.
///
/// \param blocks `[rows, G, 16]` uint8.
/// \param scales `[rows, G]` uint8.
/// \param out    `[rows, G · 32]` bf16.
Status DequantizeMxfp4ToBf16(const TensorView& blocks, const TensorView& scales,
                             const TensorView& out, Stream stream = {});

/// \brief Dequantizes and de-interleaves a gpt-oss `gate_up` weight in one
/// pass.
///
/// gpt-oss stores gate and up **interleaved** along the `2·intermediate` axis —
/// its MLP reads `gate_up[..., ::2]` and `gate_up[..., 1::2]` — where every
/// other model in this engine concatenates them, `[gate | up]`, which is what
/// `SiluMulFused` and the fused `gate_up` GEMM assume.
///
/// Rather than teach the run-time path a second layout, the permutation happens
/// once here, at load: source row `2i` becomes destination row `i`, source row
/// `2i + 1` becomes destination row `intermediate + i`. It is free — the
/// dequantize kernel has to choose a destination row anyway — and it means
/// nothing downstream of the loader ever knows gpt-oss was different.
///
/// \param blocks `[2·intermediate, G, 16]` uint8. The row count must be even.
/// \param scales `[2·intermediate, G]` uint8.
/// \param out    `[2·intermediate, G · 32]` bf16, `[gate | up]`.
Status DequantizeMxfp4GateUpToBf16(const TensorView& blocks,
                                   const TensorView& scales,
                                   const TensorView& out, Stream stream = {});

}  // namespace inferx::ops
