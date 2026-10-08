#ifndef INFERX_MODELS_LAYERS_NORMALIZATION_H_
#define INFERX_MODELS_LAYERS_NORMALIZATION_H_

#include <utility>

#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"
#include "inferx/ops/rms_norm.h"

namespace inferx::models::layers {

/// \brief Options of one RMS normalization.
struct NormConfig {
  float eps = 1e-5f;      ///< RMS epsilon.
  bool plus_one = false;  ///< Scale by (1 + w) instead of w (Gemma, Qwen3-Next).
};

/// \brief Root-mean-square normalization over the last dimension. Epsilon
///        and the scale convention are configuration; execution is shared.
class RmsNorm {
 public:
  RmsNorm(NormConfig config, Tensor weight)
      : config_(config.eps, config.plus_one, !config.plus_one),
        weight_(std::move(weight)) {}

  Status Forward(ops::OpContext& ctx, const Tensor& input, Tensor& output) const {
    return ops::RmsNorm(ctx, input, weight_, output, config_);
  }

  /// \brief Residual add fused with the following normalization; the
  ///        sublayer output joins the residual stream here.
  Status AddForward(ops::OpContext& ctx, const Tensor& input, Tensor& residual,
                    Tensor& output) const {
    return ops::AddRmsNorm(ctx, input, residual, weight_, output, config_);
  }

  /// \brief The held weight tensor (memory accounting).
  const Tensor& weight() const { return weight_; }

 private:
  ops::RMSNormConfig config_;
  Tensor weight_;
};

}  // namespace inferx::models::layers

#endif  // INFERX_MODELS_LAYERS_NORMALIZATION_H_
