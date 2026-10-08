#include "inferx/server/generation_config.h"

#include <fstream>
#include <sstream>

#include "nlohmann/json.hpp"

namespace inferx::server {
namespace {

/// \brief Reads an optional numeric JSON field; absent or null stays unset.
template <class T>
std::optional<T> GetOptional(const nlohmann::json& j, const char* key) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return std::nullopt;
  return it->get<T>();
}

}  // namespace

StatusOr<GenerationConfig> GenerationConfig::FromJson(const std::string& text) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError(std::string("failed to parse generation_config.json: ") +
                                e.what());
  }
  if (!j.is_object()) {
    return InvalidArgumentError("generation_config.json must be a JSON object");
  }

  try {
    GenerationConfig config;
    config.temperature = GetOptional<float>(j, "temperature");
    config.top_p = GetOptional<float>(j, "top_p");
    config.min_p = GetOptional<float>(j, "min_p");
    config.top_k = GetOptional<std::int64_t>(j, "top_k");
    config.repetition_penalty = GetOptional<float>(j, "repetition_penalty");
    config.max_new_tokens = GetOptional<std::int64_t>(j, "max_new_tokens");
    return config;
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError(std::string("invalid generation_config.json field: ") +
                                e.what());
  }
}

StatusOr<GenerationConfig> GenerationConfig::FromFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return NotFoundError("could not open generation config: " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return FromJson(buffer.str());
}

StatusOr<std::optional<GenerationConfig>> GenerationConfig::Resolve(
    const std::string& model_dir, const std::string& generation_config) {
  if (generation_config == "vllm") return std::nullopt;
  const std::string& dir =
      generation_config == "auto" ? model_dir : generation_config;
  auto loaded = GenerationConfig::FromFile(dir + "/generation_config.json");
  if (loaded.ok()) return std::optional<GenerationConfig>(*loaded);
  if (generation_config == "auto" && absl::IsNotFound(loaded.status())) {
    return std::nullopt;
  }
  return loaded.status();
}

}  // namespace inferx::server
