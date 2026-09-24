/// \file
/// \brief Normalization parameters shared by every norm site in a model.

#ifndef INFERX_MODELS_COMPONENTS_NORM_H_
#define INFERX_MODELS_COMPONENTS_NORM_H_

namespace inferx::components {

/// \brief RMS normalization parameters.
struct NormConfig {
  float eps = 1e-5f;
  bool plus_one = false;  ///< Scale by (1 + w) instead of w (Qwen3-Next).
};

}  // namespace inferx::components

#endif  // INFERX_MODELS_COMPONENTS_NORM_H_
