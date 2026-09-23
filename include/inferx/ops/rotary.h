#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Parameters of one rotary position encoding.
struct RotaryParams {
  int64_t rotary_dim = 0;  ///< Rotated columns per head; even, <= head_dim.
  float theta = 10000.0f;  ///< Base frequency.
};

/// \brief Applies neox-style (rotate-half) RoPE to `q` and `k`, in place.
///
/// For each head, the first and second halves of the rotary window form
/// complex pairs: x[i] and x[i + rotary_dim/2] rotate by
/// position * theta^(-2i/rotary_dim). Columns past `rotary_dim` pass through
/// untouched. Frequencies and trigonometry run in fp32; cosine/sine values
/// and each multiply/add are rounded to the activation dtype, matching the
/// reference BF16 cache and rotation. Work is enqueued on the context's stream.
///
/// \param ctx        Execution context; all tensors must live on ctx.device().
/// \param q          [tokens, query_heads, head_dim] queries; updated in place.
/// \param k          [tokens, kv_heads, head_dim] keys; updated in place.
/// \param positions  [tokens] int32 position of each token.
/// \param params     Rotary parameters.
/// \return           OK, or InvalidArgument/Unimplemented for bad inputs.
Status ApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params);

/// Rounded per-head RMSNorm followed by RoPE, fused for BF16 heads of width
/// 128. Preserves the intermediate BF16 normalization and rotation roundings.
Status NormalizeAndApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                             const Tensor& q_weight, const Tensor& k_weight,
                             const Tensor& positions, float eps, const RotaryParams& params);

}  // namespace inferx::ops
