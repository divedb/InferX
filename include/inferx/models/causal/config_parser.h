#pragma once

#include <string_view>

#include "inferx/models/causal/decoder_config.h"
#include "inferx/ops/elementwise.h"
#include "nlohmann/json.hpp"

namespace inferx::causal {

/// \brief Parses a `config.json` text, wrapping parse failures in a Status.
///
/// The registry owns parsing once; family translators and builders receive
/// the parsed json and never repeat the try/catch boilerplate.
StatusOr<nlohmann::json> ParseConfigJson(std::string_view text);

/// \brief Family identity traits the checkpoint does not state.
///
/// Everything here is declared by the family, not parsed: checkpoints agree
/// with their reference implementation about Q/K normalization presence and
/// the (1 + w) norm convention, so the translator asserts them once.
struct DecoderDefaults {
  bool qk_norm = false;
  bool plus_one_norm = false;      ///< Layer and final norms (Gemma, Qwen3-Next).
  bool qk_norm_plus_one = false;   ///< Per-head Q/K norms (Gemma, Qwen3-Next).
};

/// \brief Maps a `hidden_act` name (and gpt-oss's swiglu_limit) to a flavor.
StatusOr<ops::Activation> ParseActivation(const nlohmann::json& json);

// Common GQA + gated-MLP defaults. Families translate their config.json on
// top of this and select the actual per-layer block types.
StatusOr<DecoderConfig> AttentionDecoderConfig(const nlohmann::json& json,
                                               const DecoderDefaults& defaults = {});

}  // namespace inferx::causal
