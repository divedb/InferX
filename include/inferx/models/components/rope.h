/// \file
/// \brief Rotary position embedding parameters.

#ifndef INFERX_MODELS_COMPONENTS_ROPE_H_
#define INFERX_MODELS_COMPONENTS_ROPE_H_

#include <cmath>
#include <cstdint>
#include <string>

#include "inferx/ops/rotary.h"

namespace inferx {
namespace components {

enum class RopeStyle { kNeox };

/// \brief Raw rotary parameters as the checkpoint states them.
///
/// Finalization (YaRN's effective factor, ramp bounds, attention temperature)
/// happens once in Params(), so every consumer shares one formula.
struct RotaryConfig {
  int64_t dim = 0;               ///< Rotated columns per head; even, <= head_dim.
  float theta = 10000.0f;        ///< Base frequency (rope_scaling may override).
  float factor = 1.0f;           ///< Scaling factor applied by the flavor.
  std::string type = "default";  ///< Flavor: default, linear, llama3, yarn.
  /// Llama3 wavelength split, in multiples of original_max_position.
  float low_freq_factor = 1.0f;
  float high_freq_factor = 4.0f;
  /// Training-time context; 0 falls back to factor-only flavors.
  int64_t original_max_position = 0;
  /// YaRN correction-range rotation counts; ramp in frequency indices.
  float beta_fast = 32.0f;
  float beta_slow = 1.0f;
  bool truncate = true;  ///< YaRN: round the ramp bounds to whole indices.
  /// Raw scaling JSON kept for diagnostics and flavors beyond shared fields.
  std::string parameters_json = "{}";

  /// \brief Finalizes the checkpoint fields into kernel parameters.
  ops::RotaryParams Params() const {
    ops::RotaryParams params;
    params.rotary_dim = dim;
    params.theta = theta;
    ops::RopeScaling& s = params.scaling;
    if (type == "linear" || type == "pi") {
      s.flavor = ops::RopeFlavor::kLinear;
      s.factor = factor;
    } else if (type == "llama3") {
      s.flavor = ops::RopeFlavor::kLlama3;
      s.factor = factor;
      s.low_freq_factor = low_freq_factor;
      s.high_freq_factor = high_freq_factor;
      s.original_max_position = static_cast<float>(original_max_position);
    } else if (type == "yarn") {
      s.flavor = ops::RopeFlavor::kYarn;
      s.factor = factor;
      s.ramp_low = YarnCorrectionDim(beta_fast);
      s.ramp_high = YarnCorrectionDim(beta_slow);
      if (truncate) {
        s.ramp_low = std::floor(s.ramp_low);
        s.ramp_high = std::ceil(s.ramp_high);
      }
      s.ramp_low = std::max(s.ramp_low, 0.0f);
      s.ramp_high = std::min(s.ramp_high, static_cast<float>(dim - 1));
      // Softmax temperature of the YaRN paper: 0.1 * ln(factor) + 1.
      s.attention_scale =
          s.factor > 1.0f ? 0.1f * std::log(s.factor) + 1.0f : 1.0f;
    } else {
      s.flavor = ops::RopeFlavor::kDefault;
    }
    return params;
  }

 private:
  /// Inverse of the rotation-count formula: the frequency index whose
  /// wavelength rotates `rotations` times over the original context.
  float YarnCorrectionDim(float rotations) const {
    if (rotations <= 0.0f || theta <= 1.0f || original_max_position <= 0) return 0.0f;
    const float rope_dim = static_cast<float>(dim);
    const float max_pos = static_cast<float>(original_max_position);
    return rope_dim * std::log(max_pos / (rotations * 6.28318530717958647692f)) /
           (2.0f * std::log(theta));
  }
};

}  // namespace inferx::components
}  // namespace inferx

#endif  // INFERX_MODELS_COMPONENTS_ROPE_H_
