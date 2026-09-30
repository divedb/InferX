/// \file
/// \brief Normalization parameters shared by every norm site in a model.

#ifndef INFERX_MODELS_COMPONENTS_NORM_H_
#define INFERX_MODELS_COMPONENTS_NORM_H_

#include <utility>

#include "inferx/ops/rms_norm.h"

namespace inferx::components {

/// \brief RMS normalization parameters.
struct NormConfig {
  float eps = 1e-5f;
  bool plus_one = false;  ///< Scale by (1 + w) instead of w (Qwen3-Next).
};

/// Normalization kind is selected by traits; epsilon and scale convention
/// are supplied by the checkpoint translator.
class RmsNorm {
 public:
  RmsNorm(NormConfig config, Tensor weight)
      : config_{config.eps, config.plus_one, !config.plus_one}, weight_(std::move(weight)) {}

  Status Forward(ops::OpContext& ctx, const Tensor& input, Tensor& output) const {
    return ops::RmsNorm(ctx, input, weight_, output, config_);
  }

  Status AddForward(ops::OpContext& ctx, const Tensor& input, Tensor& residual,
                    Tensor& output) const {
    return ops::AddRmsNorm(ctx, input, residual, weight_, output, config_);
  }

 private:
  ops::RMSNormConfig config_;
  Tensor weight_;
};

}  // namespace inferx::components

#endif  // INFERX_MODELS_COMPONENTS_NORM_H_
