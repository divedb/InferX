/// \file
/// \brief Parsed HuggingFace model configuration.

#ifndef INFERX_CHECKPOINT_CONFIG_H_
#define INFERX_CHECKPOINT_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"

namespace inferx {

/// \brief Sampling-relevant defaults parsed from a HuggingFace
///        `generation_config.json`.
///
/// Mirrors the fields vLLM adopts from the checkpoint: absent fields stay
/// nullopt so the server defaults pass through unchanged. This type carries
/// no policy; front ends decide precedence against their own defaults.
struct GenerationConfig {
  std::optional<float> temperature;
  std::optional<float> top_p;
  std::optional<float> min_p;
  /// \brief HF writes -1 for "disabled"; callers map <= 0 to "off".
  std::optional<std::int64_t> top_k;
  std::optional<float> repetition_penalty;
  std::optional<std::int64_t> max_new_tokens;

  /// \brief Parses a `generation_config.json` string; absent or null fields
  ///        are left unset.
  static absl::StatusOr<GenerationConfig> FromJson(const std::string& json_text);

  /// \brief Parses a `generation_config.json` file.
  static absl::StatusOr<GenerationConfig> FromFile(const std::string& path);
};

/// \brief Model dimensions and hyperparameters parsed from a HuggingFace
///        `config.json`.
struct CheckpointConfig {
  std::string architectures;  ///< e.g. "Qwen3ForCausalLM".
  std::string text_model_type;  ///< Nested text architecture, if present.
  std::string model_type;     ///< e.g. "qwen3".
  int64_t hidden_size = 0;
  int64_t intermediate_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;
  float rms_norm_eps = 1e-5f;
  float rope_theta = 10000.0f;
  bool tie_word_embeddings = false;
  int64_t bos_token_id = -1;
  int64_t eos_token_id = -1;
  std::string hidden_act = "silu";

  /// \brief Returns the number of query heads times head_dim.
  int64_t QueryDim() const { return num_attention_heads * head_dim; }
  /// \brief Returns the number of key/value heads times head_dim.
  int64_t KvDim() const { return num_key_value_heads * head_dim; }
  /// \brief Returns the number of rotated columns per attention head; even
  ///        and at most `head_dim`.
  int64_t RotaryDim() const;

  /// \brief Parses a HuggingFace `config.json` file.
  ///
  /// \param path Path to the `config.json` file.
  /// \return The parsed configuration, or an error status.
  static absl::StatusOr<CheckpointConfig> FromFile(const std::string& path);

  /// \brief Parses a `config.json` string.
  ///
  /// \param json_text Raw JSON text of a `config.json` file.
  /// \return The parsed configuration, or an error status.
  static absl::StatusOr<CheckpointConfig> FromJson(const std::string& json_text);
};

}  // namespace inferx

#endif  // INFERX_CHECKPOINT_CONFIG_H_
