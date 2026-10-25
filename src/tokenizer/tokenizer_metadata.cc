#include "tokenizer_metadata.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace inferx {
namespace {

/// transformers writes special tokens either as plain strings or as
/// AddedToken objects {"content": ..., "special": ...}; either form
/// resolves to the content string. Returns "" when unset or malformed.
std::string TokenContent(const nlohmann::json& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_object() && value.contains("content") && value["content"].is_string()) {
    return value["content"].get<std::string>();
  }
  return "";
}

/// Collects the contents of an `additional_special_tokens` array.
std::vector<std::string> AdditionalTokens(const nlohmann::json& value) {
  std::vector<std::string> tokens;
  if (!value.is_array()) return tokens;
  for (const auto& entry : value) {
    std::string content = TokenContent(entry);
    if (!content.empty()) tokens.push_back(std::move(content));
  }
  return tokens;
}

/// Applies the special-token entries of one config document onto `metadata`.
void ApplyConfig(TokenizerMetadata& metadata, const nlohmann::json& config) {
  if (config.contains("bos_token")) metadata.bos_token = TokenContent(config["bos_token"]);
  if (config.contains("eos_token")) metadata.eos_token = TokenContent(config["eos_token"]);
  if (config.contains("pad_token")) metadata.pad_token = TokenContent(config["pad_token"]);
  if (config.contains("unk_token")) metadata.unk_token = TokenContent(config["unk_token"]);
  if (config.contains("additional_special_tokens")) {
    metadata.additional_special_tokens =
        AdditionalTokens(config["additional_special_tokens"]);
  }
}

/// Extracts the chat template source from a config document.
///
/// transformers serializes the template as a string, but newer checkpoints
/// may store a {name: source} map or a [{name, template}] list; the named
/// "default" (or first) entry is selected, as in transformers.
std::string ChatTemplateSource(const nlohmann::json& value) {
  if (value.is_string()) return value.get<std::string>();
  if (value.is_object()) {
    if (value.contains("default") && value["default"].is_string()) {
      return value["default"].get<std::string>();
    }
    for (const auto& entry : value.items()) {
      if (entry.value().is_string()) return entry.value().get<std::string>();
    }
    return "";
  }
  if (value.is_array()) {
    for (const auto& entry : value) {
      if (entry.is_object() && entry.contains("name") && entry["name"] == "default" &&
          entry.contains("template") && entry["template"].is_string()) {
        return entry["template"].get<std::string>();
      }
    }
    for (const auto& entry : value) {
      if (entry.is_object() && entry.contains("template") && entry["template"].is_string()) {
        return entry["template"].get<std::string>();
      }
    }
  }
  return "";
}

/// model_max_length, dropping transformers' 1e30 "unbounded" sentinel.
std::optional<int64_t> ModelMaxLength(const nlohmann::json& config) {
  if (!config.contains("model_max_length") || !config["model_max_length"].is_number()) {
    return std::nullopt;
  }
  const double value = config["model_max_length"].get<double>();
  if (value < 1.0 || value >= 1e30) return std::nullopt;
  return static_cast<int64_t>(value);
}

}  // namespace

StatusOr<TokenizerJsonInfo> ParseTokenizerJson(const nlohmann::json& tokenizer_json) {
  if (!tokenizer_json.is_object() || !tokenizer_json.contains("model") ||
      !tokenizer_json["model"].is_object()) {
    return InvalidArgumentError(
        "tokenizer.json must be a fast-tokenizer document with a 'model' object");
  }
  const auto& model = tokenizer_json["model"];

  TokenizerJsonInfo info;
  if (model.contains("vocab") && model["vocab"].is_object()) {
    // BPE / WordLevel / WordPiece: token -> id map.
    info.base_vocab.reserve(model["vocab"].size());
    for (const auto& entry : model["vocab"].items()) {
      if (!entry.value().is_number_integer()) continue;
      const TokenId id = entry.value().get<TokenId>();
      info.base_vocab.emplace_back(entry.key(), id);
      info.max_token_id = std::max(info.max_token_id, id);
    }
  } else if (model.contains("vocab") && model["vocab"].is_array()) {
    // Unigram: [[token, score], ...] with implicit ids.
    TokenId id = 0;
    for (const auto& entry : model["vocab"]) {
      if (entry.is_array() && !entry.empty() && entry[0].is_string()) {
        info.base_vocab.emplace_back(entry[0].get<std::string>(), id);
        info.max_token_id = std::max(info.max_token_id, id);
      }
      ++id;
    }
  } else {
    return InvalidArgumentError("tokenizer.json model has no 'vocab' mapping");
  }
  info.base_vocab_size = info.base_vocab.size();

  if (tokenizer_json.contains("added_tokens") && tokenizer_json["added_tokens"].is_array()) {
    for (const auto& entry : tokenizer_json["added_tokens"]) {
      if (!entry.is_object() || !entry.contains("content") || !entry["content"].is_string() ||
          !entry.contains("id") || !entry["id"].is_number_integer()) {
        continue;
      }
      TokenizerJsonInfo::AddedToken token;
      token.content = entry["content"].get<std::string>();
      token.id = entry["id"].get<TokenId>();
      token.special = entry.value("special", false);
      info.added_tokens.push_back(std::move(token));
      info.max_token_id = std::max(info.max_token_id, info.added_tokens.back().id);
    }
  }
  return info;
}

StatusOr<std::unique_ptr<nlohmann::json>> ParseConfigBlob(std::string_view blob,
                                                          std::string_view what) {
  if (blob.empty()) return std::unique_ptr<nlohmann::json>();
  auto parsed = std::make_unique<nlohmann::json>();
  try {
    *parsed = nlohmann::json::parse(blob);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError(what, " is not valid JSON: ", e.what());
  }
  if (!parsed->is_object()) {
    return InvalidArgumentError(what, " must be a JSON object");
  }
  return parsed;
}

TokenizerMetadata MergeMetadata(const nlohmann::json* tokenizer_config,
                                const nlohmann::json* special_tokens_map) {
  TokenizerMetadata metadata;
  if (tokenizer_config != nullptr) {
    ApplyConfig(metadata, *tokenizer_config);
    metadata.model_max_length = ModelMaxLength(*tokenizer_config);
    if (tokenizer_config->contains("truncation_side") &&
        (*tokenizer_config)["truncation_side"] == "left") {
      metadata.truncation_side = TruncationSide::kLeft;
    }
    if (tokenizer_config->contains("chat_template")) {
      metadata.chat_template =
          ChatTemplateSource((*tokenizer_config)["chat_template"]);
    }
  }
  // special_tokens_map.json overrides the tokenizer_config entries it lists.
  if (special_tokens_map != nullptr) {
    ApplyConfig(metadata, *special_tokens_map);
  }
  return metadata;
}

}  // namespace inferx
