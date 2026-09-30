/// \file
/// \brief Gated dense feed-forward: configuration, weights, and execution.

#ifndef INFERX_MODELS_COMPONENTS_MLP_H_
#define INFERX_MODELS_COMPONENTS_MLP_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "inferx/core/tensor.h"
#include "inferx/models/components/linear.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/op_context.h"

namespace inferx {

namespace components {

/// \brief Gated feed-forward; the activation flavor selects the family.
struct SwiGluConfig {
  int64_t intermediate_size = 0;      ///< Gate/up width.
  ops::Activation activation = ops::Activation::kSilu;  ///< Gate nonlinearity.
  /// gpt-oss SwiGLU-oai constants; unused by the other flavors.
  float oai_alpha = 1.702f;
  float oai_limit = 7.0f;
};

/// \brief Gated projections for one expert or a dense layer.
///
/// `packed_gate_up` is the fused MergedColumnParallelLinear weight:
/// rank-local rows in block-contiguous order [gate | up]. The per-projection
/// weights are views into that allocation and share its storage. `down` is
/// the RowParallelLinear counterpart: `[hidden, rank_intermediate]`, whose
/// partial sums need cross-rank reduction once tensor parallelism runs.
/// Biases, when the checkpoint carries them (gpt-oss), are packed the same
/// way; `down_bias` is [hidden].
struct SwiGluWeights {
  Tensor packed_gate_up;               ///< [2 * intermediate, hidden] fused rows.
  LinearWeights gate;                  ///< View of packed_gate_up's gate rows.
  LinearWeights up;                    ///< View of packed_gate_up's up rows.
  LinearWeights down;                  ///< [hidden, intermediate]
  std::optional<Tensor> packed_bias;   ///< [2 * intermediate] fused gate|up bias.
  std::optional<Tensor> down_bias;     ///< [hidden]; present only when biased.
};

/// \brief Reusable SwiGLU workspace, sized once by the decoder stack.
///
/// One buffer: the fused gate/up GEMM de-interleaves straight into `gate`
/// (the packed gated activation writes [rows, intermediate]).
struct MlpWorkspace {
  std::optional<Tensor> gate;  ///< [max_tokens * max_intermediate] flat.
};

/// \brief Runs the gated feed-forward and projects back into `mixed_out`
///        ([rows, hidden]).
///
/// One fused GEMM over the packed gate/up weight into `packed_buffer` (the
/// shared packed-projection workspace the stack always provides), optional
/// fused-row bias, the configured gated activation, then the down projection
/// and its optional bias.
Status RunSwiGlu(const SwiGluConfig& config, const SwiGluWeights& weights, const Tensor& normed,
                 MlpWorkspace& ws, Tensor* packed_buffer, ops::OpContext& ctx,
                 Tensor& mixed_out);

/// A concrete gated-MLP component selected by model traits. Activation
/// flavor, clamps, and biases are runtime configuration; execution is shared.
template <ops::Activation Act>
class GatedMlp {
 public:
  using Config = SwiGluConfig;
  using Weights = SwiGluWeights;
  static constexpr ops::Activation kActivation = Act;

  GatedMlp(Config config, Weights weights)
      : config_(std::move(config)), weights_(std::move(weights)) {}

  Status Forward(const Tensor& input, MlpWorkspace& workspace, Tensor* packed,
                 ops::OpContext& ctx, Tensor& output) const {
    return RunSwiGlu(config_, weights_, input, workspace, packed, ctx, output);
  }

 private:
  Config config_;
  Weights weights_;
};

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_MLP_H_
