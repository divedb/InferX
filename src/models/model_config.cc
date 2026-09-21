#include "inferx/models/model_config.h"

#include <fstream>
#include <sstream>

#include "absl/status/status.h"
#include "nlohmann/json.hpp"

namespace inferx {
namespace {

/// \brief Reads an integral JSON field if present.
///
/// \param j JSON object to read from.
/// \param key Field name.
/// \param fallback Value returned when the field is absent or null.
/// \return The field value, or `fallback`.
int64_t GetInt(const nlohmann::json& j, const char* key, int64_t fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return fallback;
  }
  return it->get<int64_t>();
}

/// \brief Reads a floating-point JSON field if present.
///
/// \param j JSON object to read from.
/// \param key Field name.
/// \param fallback Value returned when the field is absent or null.
/// \return The field value, or `fallback`.
float GetFloat(const nlohmann::json& j, const char* key, float fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return fallback;
  }
  return it->get<float>();
}

/// \brief Reads a boolean JSON field if present.
///
/// \param j JSON object to read from.
/// \param key Field name.
/// \param fallback Value returned when the field is absent or null.
/// \return The field value, or `fallback`.
bool GetBool(const nlohmann::json& j, const char* key, bool fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return fallback;
  }
  return it->get<bool>();
}

/// \brief Reads a string JSON field if present.
///
/// \param j JSON object to read from.
/// \param key Field name.
/// \param fallback Value returned when the field is absent or null.
/// \return The field value, or `fallback`.
std::string GetString(const nlohmann::json& j, const char* key,
                      const std::string& fallback) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return fallback;
  }
  return it->get<std::string>();
}

}  // namespace

absl::StatusOr<ModelConfig> ModelConfig::FromJson(const std::string& text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    return absl::InvalidArgumentError(
        std::string("failed to parse config.json: ") + e.what());
  }

  // Some checkpoints wrap the text config in a "text_config" object.
  if (j.contains("text_config") && j.at("text_config").is_object()) {
    j = j.at("text_config");
  }

  ModelConfig config;
  if (j.contains("architectures")) {
    const nlohmann::json& architectures = j.at("architectures");
    if (architectures.is_string()) {
      config.architectures = architectures.get<std::string>();
    } else if (architectures.is_array() && !architectures.empty() &&
               architectures.front().is_string()) {
      config.architectures = architectures.front().get<std::string>();
    }
  }
  config.hidden_size = GetInt(j, "hidden_size", 0);
  config.intermediate_size = GetInt(j, "intermediate_size", 0);
  config.num_hidden_layers = GetInt(j, "num_hidden_layers", 0);
  config.num_attention_heads = GetInt(j, "num_attention_heads", 0);
  config.num_key_value_heads =
      GetInt(j, "num_key_value_heads", config.num_attention_heads);
  config.vocab_size = GetInt(j, "vocab_size", 0);
  config.max_position_embeddings = GetInt(j, "max_position_embeddings", 2048);
  config.rms_norm_eps = GetFloat(j, "rms_norm_eps", 1e-5f);
  config.hidden_act = GetString(j, "hidden_act", "silu");
  config.tie_word_embeddings = GetBool(j, "tie_word_embeddings", false);
  config.bos_token_id = GetInt(j, "bos_token_id", -1);
  config.eos_token_id = GetInt(j, "eos_token_id", -1);
  config.model_type = GetString(j, "model_type", "");

  // `rope_theta` may live at the top level or under `rope_scaling`/`rope`.
  config.rope_theta = GetFloat(j, "rope_theta", 10000.0f);
  if (j.contains("rope_scaling") && j.at("rope_scaling").is_object()) {
    config.rope_theta =
        GetFloat(j.at("rope_scaling"), "rope_theta", config.rope_theta);
  }

  // `head_dim` is explicit in some architectures (e.g. Qwen2) and derived
  // otherwise.
  config.head_dim = GetInt(j, "head_dim", 0);
  if (config.head_dim == 0 && config.num_attention_heads > 0) {
    config.head_dim = config.hidden_size / config.num_attention_heads;
  }

  if (config.hidden_size <= 0 || config.num_hidden_layers <= 0 ||
      config.num_attention_heads <= 0 || config.vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "config.json is missing required model dimensions");
  }
  if (config.num_key_value_heads <= 0) {
    config.num_key_value_heads = config.num_attention_heads;
  }
  if (config.num_attention_heads % config.num_key_value_heads != 0) {
    return absl::InvalidArgumentError(
        "num_attention_heads must be divisible by num_key_value_heads");
  }
  return config;
}

int64_t ModelConfig::RotaryDim() const {
  return head_dim - head_dim % 2;
}

absl::StatusOr<ModelConfig> ModelConfig::FromFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);

  if (!input) {
    return absl::NotFoundError("could not open config file: " + path);
  }

  std::ostringstream buffer;
  buffer << input.rdbuf();
  return FromJson(buffer.str());
}

}  // namespace inferx
