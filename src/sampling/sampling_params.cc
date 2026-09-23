#include "inferx/sampling/sampling_params.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace inferx::sampling {

namespace {

/// NaN-safe half-open/closed range checks: any NaN operand fails both
/// comparisons, so NaN is rejected everywhere without a separate isnan.
bool InRange(float value, float low_inclusive, float high_inclusive) {
  return value >= low_inclusive && value <= high_inclusive;
}

}  // namespace

Status SamplingParams::Validate() const {
  if (!(temperature >= 0.0f))
    return InvalidArgumentError("temperature must be >= 0");
  if (!InRange(top_p, 0.0f, 1.0f) || top_p <= 0.0f)
    return InvalidArgumentError("top_p must be in (0, 1]");
  if (!InRange(min_p, 0.0f, 1.0f))
    return InvalidArgumentError("min_p must be in [0, 1]");
  if (!InRange(presence_penalty, -2.0f, 2.0f))
    return InvalidArgumentError("presence_penalty must be in [-2, 2]");
  if (!InRange(frequency_penalty, -2.0f, 2.0f))
    return InvalidArgumentError("frequency_penalty must be in [-2, 2]");
  if (!(repetition_penalty > 0.0f))
    return InvalidArgumentError("repetition_penalty must be > 0");
  if (max_tokens == 0) return InvalidArgumentError("max_tokens must be >= 1");
  for (const auto& [token, bias] : logit_bias) {
    if (token < 0) return InvalidArgumentError("logit_bias token ids must be >= 0");
    if (!std::isfinite(bias) && bias != -std::numeric_limits<float>::infinity())
      return InvalidArgumentError("logit_bias values must be finite or -inf");
  }
  for (std::int32_t token : allowed_token_ids)
    if (token < 0) return InvalidArgumentError("allowed_token_ids must be >= 0");
  for (std::int32_t token : stop_token_ids)
    if (token < 0) return InvalidArgumentError("stop_token_ids must be >= 0");
  return OkStatus();
}

}  // namespace inferx::sampling
