#ifndef INFERX_CONFIG_GENERATION_CONFIG_H_
#define INFERX_CONFIG_GENERATION_CONFIG_H_

#include <cstdint>
#include <string>

#include "inferx/core/status.h"

namespace inferx {

class ModelConfig;

/// \brief Generation defaults a checkpoint ships for itself.
///
/// Every field carries a plain default instead of an optional: the defaults
/// mirror `sampling::SamplingParams`, so a generation_config.json that omits
/// a field merges into sampling parameters as a no-op, and the 0/1.0
/// sentinels mean "disabled" or "not constrained" (documented per field).
struct GenerationConfig {
  /// Sampling temperature; 0 is greedy.
  /// EXAMPLE: "temperature": 0.7
  float temperature = 1.0f;

  /// Nucleus sampling probability mass; 1.0 keeps the full distribution.
  /// EXAMPLE: "top_p": 0.95
  float top_p = 1.0f;

  /// Top-k limit; 0 disables it. HuggingFace writes -1 for "disabled",
  /// normalized to 0 at load time.
  /// EXAMPLE: "top_k": 20
  std::int64_t top_k = 0;

  /// Minimum probability mass (relative to the top token) a candidate must
  /// reach to be sampled; 0.0 disables it.
  /// EXAMPLE: "min_p": 0.05
  float min_p = 0.0f;

  /// Repetition penalty; 1.0 disables it.
  /// EXAMPLE: "repetition_penalty": 1.05
  float repetition_penalty = 1.0f;

  /// Maximum number of tokens the checkpoint asks the server to generate.
  /// 0 means "the checkpoint does not constrain it": merging never
  /// overrides the engine's max_tokens default.
  /// EXAMPLE: "max_new_tokens": 512
  std::int64_t max_new_tokens = 0;
};

/// \brief Loads the effective generation defaults for a model.
///
/// Resolution order, later wins:
///   1. built-in defaults (the struct above);
///   2. the generation config file: ModelConfig::generation_config "auto"
///      reads `<model_dir>/generation_config.json` (a missing file is
///      normal), any other value is the file path itself and must exist;
///   3. ModelConfig::override_generation_config, merged at the JSON level
///      so an override only replaces the keys it names.
///
/// Attributes are read through ConfigView, so a malformed value fails with
/// the offending key path instead of silently becoming a default. Explicit
/// command-line sampling flags still win over everything here; see
/// MergeGenerationConfig in inferx/server/serve.h.
///
/// \param model_config The model configuration to resolve against.
/// \return             The effective generation defaults, or
///                     kNotFound (configured file missing) or
///                     kInvalidArgument (malformed JSON, bad types).
[[nodiscard]] StatusOr<GenerationConfig> LoadGenerationConfig(const ModelConfig& model_config);

}  // namespace inferx

#endif  // INFERX_CONFIG_GENERATION_CONFIG_H_
