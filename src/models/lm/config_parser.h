#pragma once

#include "inferx/models/lm/stack.h"
#include "nlohmann/json.hpp"

namespace inferx::lm {

StatusOr<std::string> ReadConfig(const std::string& directory);
// Common GQA + SwiGLU defaults. Family builders select the actual block types.
StatusOr<DecoderConfig> AttentionDecoderConfig(const nlohmann::json& json,
                                               bool qk_norm, bool plus_one_norm);

}  // namespace inferx::lm
