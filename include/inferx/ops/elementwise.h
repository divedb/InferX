#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Gated-activation kernels shared by every feed-forward flavor.
enum class Activation : uint8_t {
  kSilu,      ///< silu(gate) * up; Llama, Qwen, Mistral, DeepSeek, Mixtral.
  kGeluTanh,  ///< gelu_tanh(gate) * up; Gemma GeGLU.
  kSiluOai,   ///< (clamp(up) + 1) * clamp(gate) * sigmoid(alpha * gate); gpt-oss.
};

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

/// \brief Gated activation from a BF16 [tokens, 2*width] packed projection
///        whose rows are [gate | up], for any activation flavor.
///
/// One kernel serves every flavor; `act` selects the gate nonlinearity.
/// gpt-oss clamps both halves at `oai_limit` and biases `up` by +1
/// (alpha 1.702), matching the reference SwiGLU-oai.
Status PackedGatedActivation(ExecutionContext& ctx, const Tensor& packed, Tensor& out,
                             Activation act, float oai_alpha = 1.702f,
                             float oai_limit = 7.0f);

/// \brief Split an attention projection with up to four column groups.
///
/// Column groups are block-contiguous in `packed` in the order q, gate, k, v;
/// `gate` may be null when the projection is ungated. Rows are preserved.
Status SplitProjection(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor* gate,
                       Tensor& k, Tensor& v);

/// \brief Row-wise bias: out[t, :] <- x[t, :] + bias.
///
/// `bias` is [width]; `out` may alias `x`.
Status AddBias(ExecutionContext& ctx, const Tensor& x, const Tensor& bias, Tensor& out);

/// \brief In-place sigmoid gating: x <- x * sigmoid(gate).
///
/// Element-for-element shapes (Qwen3-Next's gated attention output).
Status MulSigmoidGate(ExecutionContext& ctx, Tensor& x, const Tensor& gate);

/// \brief In-place per-row sigmoid gating: x[t, :] <- x[t, :] * sigmoid(gate[t]).
///
/// `gate` is [rows, 1] (a shared-expert gate); broadcasts over columns.
Status MulSigmoidRowGate(ExecutionContext& ctx, Tensor& x, const Tensor& gate);

/// \brief In-place scalar multiply: x <- x * scalar (embedding scaling).
Status MulScalar(ExecutionContext& ctx, Tensor& x, float scalar);

}  // namespace inferx::ops
