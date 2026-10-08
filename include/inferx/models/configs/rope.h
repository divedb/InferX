/// \file
/// \brief RopeConfig: the shared value type architectures embed for rotary
/// position embeddings. Each architecture parser decides which rope options
/// and defaults are valid for it; finalization into kernel parameters
/// happens once in Params().

#ifndef INFERX_MODELS_CONFIGS_ROPE_H_
#define INFERX_MODELS_CONFIGS_ROPE_H_

#include <cmath>
#include <cstdint>
#include <string>

#include "inferx/core/status.h"
#include "inferx/ops/rotary.h"
#include "inferx/util/config_view.h"

namespace inferx::models {

/// \brief Rotary parameters as the checkpoint states them, parsed and
///        validated. Owns its values; retains no JSON view.
struct RopeConfig {
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

  /// \brief Parses rope settings from one architecture config view.
  ///
  /// Reads `rope_theta` (default 10000), the `rope_scaling` or
  /// `rope_parameters` object when present, and `partial_rotary_factor`
  /// (top-level or inside the scaling object). YaRN's effective factor
  /// becomes the context ratio once the original training length is known.
  ///
  /// \param view      The architecture's configuration object.
  /// \param head_dim  Per-head dimension; the rotary window derives from it.
  /// \param max_position_embeddings  This checkpoint's context length.
  static StatusOr<RopeConfig> Parse(const ConfigView& view, int64_t head_dim,
                                    int64_t max_position_embeddings);

  /// \brief Checks geometry and flavor values, relative to `head_dim`.
  Status Validate(int64_t head_dim) const;

  /// \brief Finalizes the parsed fields into kernel parameters, computing
  ///        YaRN's ramp bounds and attention temperature once.
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
      s.attention_scale = s.factor > 1.0f ? 0.1f * std::log(s.factor) + 1.0f : 1.0f;
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

}  // namespace inferx::models

#endif  // INFERX_MODELS_CONFIGS_ROPE_H_
