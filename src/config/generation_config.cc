#include "inferx/config/generation_config.h"

#include "inferx/config/model_config.h"
#include "inferx/util/config_view.h"
#include "inferx/util/file_util.h"
#include "nlohmann/json.hpp"

namespace inferx {
namespace {

/// Decodes JSON text without exceptions (a discarded value marks failure).
StatusOr<nlohmann::json> ParseJsonText(const std::string& text, const std::string& source) {
  const nlohmann::json parsed =
      nlohmann::json::parse(text, /*callback=*/nullptr, /*allow_exceptions=*/false);
  if (parsed.is_discarded()) {
    return InvalidArgumentError(source, ": not valid JSON");
  }

  return parsed;
}

/// Reads generation settings through ConfigView: absent keys keep the
/// struct defaults, and a present-but-malformed value fails naming the key.
StatusOr<GenerationConfig> ReadSettings(const nlohmann::json& object) {
  assert(object.is_object() && "ReadSettings must be called with a JSON object");

  const ConfigView view(&object, "generation config");

  GenerationConfig config;
  // Floats round-trip through double (ConfigView's numeric type).
  INFERX_ASSIGN_OR_RETURN(const double temperature,
                          view.Float("temperature", config.temperature));
  INFERX_ASSIGN_OR_RETURN(const double top_p, view.Float("top_p", config.top_p));
  INFERX_ASSIGN_OR_RETURN(const double min_p, view.Float("min_p", config.min_p));
  INFERX_ASSIGN_OR_RETURN(const double repetition_penalty,
                          view.Float("repetition_penalty", config.repetition_penalty));
  INFERX_ASSIGN_OR_RETURN(config.top_k, view.Int("top_k", config.top_k));
  INFERX_ASSIGN_OR_RETURN(config.max_new_tokens,
                          view.Int("max_new_tokens", config.max_new_tokens));
  config.temperature = static_cast<float>(temperature);
  config.top_p = static_cast<float>(top_p);
  config.min_p = static_cast<float>(min_p);
  config.repetition_penalty = static_cast<float>(repetition_penalty);

  // HuggingFace writes top_k = -1 for "disabled"; normalize to 0. Any other
  // negative value is a checkpoint bug, not a mode.
  if (config.top_k < 0) {
    if (config.top_k != -1) {
      return InvalidArgumentError(view.PathOf("top_k"), ": must be -1 (disabled) or >= 0");
    }

    config.top_k = 0;
  }

  if (config.max_new_tokens < 0) {
    return InvalidArgumentError(view.PathOf("max_new_tokens"), ": must be >= 0");
  }

  return config;
}

}  // namespace

StatusOr<GenerationConfig> LoadGenerationConfig(const ModelConfig& model_config) {
  const bool is_auto = model_config.generation_config == "auto";
  const std::filesystem::path path =
      is_auto ? std::filesystem::path(model_config.model_dir) / "generation_config.json"
              : std::filesystem::path(model_config.generation_config);
  nlohmann::json effective = nlohmann::json::object();
  std::error_code ec;

  if (std::filesystem::exists(path, ec)) {
    auto text = ReadFileToString(path);

    if (!text.ok()) {
      return text.status();
    }

    const auto parsed = ParseJsonText(*text, path.string());

    if (!parsed.ok()) {
      return parsed.status();
    }

    if (!parsed->is_object()) {
      return InvalidArgumentError(path.string(), ": must be a JSON object");
    }

    effective = std::move(*parsed);
  } else if (ec) {
    return InternalError("cannot access generation config: ", path.string(), ": ",
                         ec.message());
  } else if (!is_auto) {
    // "auto" without a file is normal; an explicitly configured file must exist.
    return NotFoundError("could not open generation config: ", path.string());
  }

  if (!model_config.override_generation_config.empty()) {
    auto override_json =
        ParseJsonText(model_config.override_generation_config, "--override-generation-config");

    if (!override_json.ok()) {
      return override_json.status();
    }

    if (!override_json->is_object()) {
      return InvalidArgumentError("--override-generation-config: must be a JSON object");
    }

    // RFC 7386 merge: only named keys are replaced; an explicit null removes the key.
    effective.merge_patch(*override_json);
  }

  return ReadSettings(effective);
}

}  // namespace inferx
