#pragma once

#include <string_view>

#include "inferx/models/causal/decoder_config.h"
#include "nlohmann/json.hpp"

namespace inferx::causal {

/// \brief Parses a `config.json` text, wrapping parse failures in a Status.
///
/// The registry owns parsing once; family translators and builders receive
/// the parsed json and never repeat the try/catch boilerplate.
StatusOr<nlohmann::json> ParseConfigJson(std::string_view text);

// Common GQA + SwiGLU defaults. Families translate their config.json on top
// of this and select the actual per-layer block types.
StatusOr<DecoderConfig> AttentionDecoderConfig(const nlohmann::json& json, bool qk_norm,
                                               bool plus_one_norm);

}  // namespace inferx::causal
