#include "inferx/models/configs/rope.h"

#include <algorithm>
#include <optional>

namespace inferx::models {

StatusOr<RopeConfig> RopeConfig::Parse(const ConfigView& view, int64_t head_dim,
                                       int64_t max_position_embeddings) {
  RopeConfig rope;
  double theta = 0.0;
  INFERX_ASSIGN_OR_RETURN(theta, view.Float("rope_theta", static_cast<double>(rope.theta)));
  rope.theta = static_cast<float>(theta);

  std::optional<ConfigView> scaling;
  for (const char* key : {"rope_scaling", "rope_parameters"}) {
    INFERX_ASSIGN_OR_RETURN(auto found, view.Sub(key));
    if (found.has_value()) {
      scaling = std::move(*found);
      break;
    }
  }
  double partial = 1.0;
  if (!scaling.has_value()) {
    INFERX_ASSIGN_OR_RETURN(partial, view.Float("partial_rotary_factor", 1.0));
  } else {
    const ConfigView& s = *scaling;
    INFERX_ASSIGN_OR_RETURN(rope.type, s.Str("rope_type", std::string(rope.type)));
    INFERX_ASSIGN_OR_RETURN(rope.type, s.Str("type", rope.type));
    INFERX_ASSIGN_OR_RETURN(rope.theta, s.Float("rope_theta", static_cast<double>(rope.theta)));
    INFERX_ASSIGN_OR_RETURN(rope.factor, s.Float("factor", static_cast<double>(rope.factor)));
    INFERX_ASSIGN_OR_RETURN(partial, s.Float("partial_rotary_factor", 1.0));
    if (rope.type == "llama3") {
      INFERX_ASSIGN_OR_RETURN(rope.low_freq_factor,
                              s.Float("low_freq_factor", static_cast<double>(rope.low_freq_factor)));
      INFERX_ASSIGN_OR_RETURN(
          rope.high_freq_factor,
          s.Float("high_freq_factor", static_cast<double>(rope.high_freq_factor)));
      INFERX_ASSIGN_OR_RETURN(
          rope.original_max_position,
          s.Int("original_max_position_embeddings", max_position_embeddings));
    } else if (rope.type == "yarn") {
      INFERX_ASSIGN_OR_RETURN(rope.beta_fast, s.Float("beta_fast", static_cast<double>(rope.beta_fast)));
      INFERX_ASSIGN_OR_RETURN(rope.beta_slow, s.Float("beta_slow", static_cast<double>(rope.beta_slow)));
      INFERX_ASSIGN_OR_RETURN(rope.truncate, s.Bool("truncate", rope.truncate));
      INFERX_ASSIGN_OR_RETURN(const int64_t original,
                              s.Int("original_max_position_embeddings", int64_t{0}));
      if (original > 0) {
        rope.original_max_position = original;
        // The effective factor is the context ratio once the original
        // training length is known (reference rule; matches the config's
        // own factor).
        rope.factor = static_cast<float>(static_cast<double>(max_position_embeddings) /
                                         static_cast<double>(original));
      }
    }
  }
  if (!std::isfinite(partial) || partial <= 0 || partial > 1) {
    return InvalidArgumentError("partial_rotary_factor must be in (0, 1]");
  }
  rope.dim = static_cast<int64_t>(static_cast<double>(head_dim) * partial);
  INFERX_RETURN_IF_ERROR(rope.Validate(head_dim));
  return rope;
}

Status RopeConfig::Validate(int64_t head_dim) const {
  if (head_dim <= 0) return InvalidArgumentError("rope requires a positive head dimension");
  if (dim <= 0 || dim > head_dim || dim % 2 != 0) {
    return InvalidArgumentError("rotary window ", dim,
                                " must be even and no larger than head_dim ", head_dim);
  }
  if (!std::isfinite(theta) || theta <= 0) {
    return InvalidArgumentError("rope theta must be finite and positive");
  }
  if (!std::isfinite(factor) || factor <= 0) {
    return InvalidArgumentError("rope scaling factor must be finite and positive");
  }
  return OkStatus();
}

}  // namespace inferx::models
