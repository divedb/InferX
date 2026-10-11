#include "inferx/models/config_utils.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace inferx {
namespace models {

StatusOr<ops::RopeScaling> ParseRopeScaling(const std::optional<ConfigView>& scaling,
                                            int64_t rotary_dim, double theta) {
  ops::RopeScaling result;
  if (!scaling.has_value()) return result;  // Unscaled rope.

  const ConfigView& view = *scaling;
  // Newer checkpoints say `rope_type`; the original field name was `type`.
  INFERX_ASSIGN_OR_RETURN(const std::string rope_type, view.Str("rope_type", ""));
  INFERX_ASSIGN_OR_RETURN(const std::string type, view.Str("type", "default"));
  const std::string effective = rope_type.empty() ? type : rope_type;
  if (effective == "default") return result;

  ops::RopeFlavor flavor;
  if (effective == "linear") {
    flavor = ops::RopeFlavor::kLinear;
  } else if (effective == "llama3") {
    flavor = ops::RopeFlavor::kLlama3;
  } else if (effective == "yarn") {
    flavor = ops::RopeFlavor::kYarn;
  } else {
    return UnimplementedError("rope scaling type '", effective, "' is not supported");
  }
  result.flavor = flavor;

  INFERX_ASSIGN_OR_RETURN(result.factor, view.Float("factor", 1.0));
  if (result.factor <= 0.0) {
    return InvalidArgumentError("rope scaling factor must be positive, got ",
                                result.factor);
  }
  if (flavor == ops::RopeFlavor::kLinear) return result;

  INFERX_ASSIGN_OR_RETURN(result.low_freq_factor, view.Float("low_freq_factor", 1.0));
  INFERX_ASSIGN_OR_RETURN(result.high_freq_factor, view.Float("high_freq_factor", 4.0));
  INFERX_ASSIGN_OR_RETURN(
      result.original_max_position,
      view.Float("original_max_position_embeddings", 0.0));
  if (result.original_max_position <= 0.0 || result.high_freq_factor <= 0.0 ||
      result.low_freq_factor <= 0.0) {
    return InvalidArgumentError("llama3/yarn scaling needs a positive original context "
                                "and positive frequency factors");
  }
  if (flavor == ops::RopeFlavor::kLlama3) return result;

  // YaRN: wavelength bounds become ramp indices -- below `ramp_low` every
  // frequency extrapolates, above `ramp_high` every frequency interpolates,
  // and between them the kernel blends linearly. wavelen(i) grows with i,
  // so the short-wavelength bound starts the ramp. `attention_scale`
  // carries the 1/mscale temperature into cos/sin.
  constexpr double kTwoPi = 6.28318530717958647692;
  const auto index_of_wavelen = [&](double wavelen) {
    return (static_cast<double>(rotary_dim) / 2.0) * std::log(wavelen / kTwoPi) /
           std::log(theta);
  };
  const double high_wavelen = result.original_max_position / result.high_freq_factor;
  const double low_wavelen = result.original_max_position / result.low_freq_factor;
  result.ramp_low = std::max(0.0, index_of_wavelen(high_wavelen));
  result.ramp_high = std::max(0.0, index_of_wavelen(low_wavelen));
  if (result.ramp_high < result.ramp_low) std::swap(result.ramp_low, result.ramp_high);
  const double mscale = 0.1 * std::log(result.factor) + 1.0;
  result.attention_scale = 1.0 / mscale;
  return result;
}

StatusOr<ops::RotaryParams> ParseRope(const ConfigView& text, int64_t rotary_dim,
                                      double default_theta) {
  if (rotary_dim <= 0 || rotary_dim % 2 != 0) {
    return InvalidArgumentError("rotary dimension must be positive and even, got ",
                                rotary_dim);
  }
  ops::RotaryParams params;
  params.rotary_dim = rotary_dim;
  INFERX_ASSIGN_OR_RETURN(params.theta, text.Float("rope_theta", default_theta));
  INFERX_ASSIGN_OR_RETURN(auto scaling, text.Sub("rope_scaling"));
  INFERX_ASSIGN_OR_RETURN(params.scaling, ParseRopeScaling(scaling, rotary_dim, params.theta));
  return params;
}

}  // namespace models
}  // namespace inferx
