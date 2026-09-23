#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Elementwise sum: out <- a + b.
///
/// All three tensors must agree element-for-element. `out` may alias `a` or
/// `b`; the per-element read happens before the write.
Status Add(ExecutionContext& ctx, const Tensor& a, const Tensor& b, Tensor& out);

/// \brief Gated activation: out <- silu(gate) * up.
///
/// silu(x) = x * sigmoid(x). `out` may alias `gate` (not `up` unless they
/// alias each other); shapes must agree element-for-element.
Status SiluAndMul(ExecutionContext& ctx, const Tensor& gate, const Tensor& up, Tensor& out);

/// Split contiguous columns of a BF16 [tokens, q+k+v] projection.
Status SplitQkv(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor& k, Tensor& v);
/// silu(gate) * up from a BF16 [tokens, 2*width] packed projection.
Status PackedSiluAndMul(ExecutionContext& ctx, const Tensor& packed, Tensor& out);

}  // namespace inferx::ops
