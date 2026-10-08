/// \file
/// \brief Gated feed-forward layer (SwiGLU family): fused gate/up
/// projection, a gated activation, and the down projection. The activation
/// flavor and its constants are configuration; execution is shared.

#ifndef INFERX_MODELS_LAYERS_GATED_MLP_H_
#define INFERX_MODELS_LAYERS_GATED_MLP_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/linear.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/op_context.h"

namespace inferx::models::layers {

/// \brief Options of one gated feed-forward layer.
struct GatedMlpConfig {
  int64_t intermediate_size = 0;    ///< Gate/up width.
  ops::Activation activation = ops::Activation::kSilu;  ///< Gate nonlinearity.
  float oai_alpha = 1.702f;  ///< gpt-oss SwiGLU-oai constants; unused by other flavors.
  float oai_limit = 7.0f;    ///< gpt-oss clamp limit.
};

/// \brief Gated projections for one expert or a dense layer.
///
/// `packed_gate_up` is the fused loader weight: rank-local rows in
/// block-contiguous order [gate | up]; per-projection weights are views
/// sharing its storage. `down` is the RowParallelLinear counterpart
/// [hidden, rank_intermediate], whose partial sums need cross-rank
/// reduction under tensor parallelism. Biases (gpt-oss) are packed the same
/// way; `down_bias` is [hidden].
struct GatedMlpWeights {
  Tensor packed_gate_up;  ///< [2 * intermediate, hidden] fused rows.
  LinearWeights gate;     ///< View of packed_gate_up's gate rows.
  LinearWeights up;       ///< View of packed_gate_up's up rows.
  LinearWeights down;     ///< [hidden, intermediate]
  std::optional<Tensor> packed_bias;  ///< [2 * intermediate] fused gate|up bias.
  std::optional<Tensor> down_bias;    ///< [hidden]; present only when biased.
};

/// \brief Reusable gated-MLP workspace, sized once by the decoder stack.
///        One buffer: the fused gate/up GEMM de-interleaves straight into
///        `gate` (the packed gated activation writes [rows, intermediate]).
struct GatedMlpWorkspace {
  std::optional<Tensor> gate;  ///< [max_tokens * max_intermediate] flat.
};

/// \brief One fused GEMM over the packed gate/up weight into
///        `packed_buffer` (the shared packed-projection workspace the stack
///        provides), optional fused-row bias, the configured gated
///        activation, then the down projection and its optional bias.
Status RunGatedMlp(const GatedMlpConfig& config, const GatedMlpWeights& weights,
                   const Tensor& normed, GatedMlpWorkspace& ws, Tensor* packed_buffer,
                   ops::OpContext& ctx, Tensor& mixed_out);

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_GATED_MLP_H_
