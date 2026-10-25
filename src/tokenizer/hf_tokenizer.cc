/// \brief HFTokenizer implementation over tokenizers-cpp and minja.
///
/// Layout: HFTokenizer is a pimpl façade; Impl owns the Rust backend
/// handle, the metadata extracted from the asset files, the resolved
/// special-token tables, and the parsed chat template. All backend calls
/// funnel through tokenizers::RustTokenizer, which enforces the input
/// preconditions the Rust side assumes (valid UTF-8, in-vocabulary ids)
/// because the vendored wrapper panics — aborting the process — otherwise.
#include "inferx/tokenizer/hf_tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "chat_template_renderer.h"
#include "rust_tokenizer.h"
#include "tokenizer_metadata.h"

namespace inferx {
namespace {

using ordered_json = nlohmann::ordered_json;

/// Reads a whole file; nullopt when it cannot be opened.
std::optional<std::string> ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return std::nullopt;
  std::string blob((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
  return blob;
}

/// Strict UTF-8 validation, matching what the Rust side enforces by unwrap:
/// rejects overlong encodings, surrogates, and code points above U+10FFFF.
bool IsValidUtf8(std::string_view text) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
  const size_t size = text.size();
  for (size_t i = 0; i < size;) {
    const unsigned char lead = bytes[i];
    if (lead < 0x80) {
      ++i;
      continue;
    }
    size_t length = 0;
    uint32_t code_point = 0;
    if ((lead & 0xE0) == 0xC0) {
      length = 2;
      code_point = lead & 0x1F;
    } else if ((lead & 0xF0) == 0xE0) {
      length = 3;
      code_point = lead & 0x0F;
    } else if ((lead & 0xF8) == 0xF0) {
      length = 4;
      code_point = lead & 0x07;
    } else {
      return false;
    }
    if (i + length > size) return false;
    for (size_t j = 1; j < length; ++j) {
      if ((bytes[i + j] & 0xC0) != 0x80) return false;
      code_point = (code_point << 6) | (bytes[i + j] & 0x3F);
    }
    if (length == 2 && code_point < 0x80) return false;
    if (length == 3 && code_point < 0x800) return false;
    if (length == 4 && code_point < 0x10000) return false;
    if (code_point >= 0xD800 && code_point <= 0xDFFF) return false;
    if (code_point > 0x10FFFF) return false;
    i += length;
  }
  return true;
}

/// trims ASCII whitespace at both ends, like Python str.strip().
std::string_view StripWhitespace(std::string_view text) {
  const auto is_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };
  size_t begin = 0;
  size_t end = text.size();
  while (begin < end && is_space(text[begin])) ++begin;
  while (end > begin && is_space(text[end - 1])) --end;
  return text.substr(begin, end - begin);
}

/// Parses a JSON blob that may be empty (absent), returning null on absence.
StatusOr<ordered_json> ParseOptionalJson(std::string_view blob, std::string_view what) {
  if (blob.empty()) return ordered_json();
  ordered_json parsed;
  try {
    parsed = ordered_json::parse(blob);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError(what, " is not valid JSON: ", e.what());
  }
  return parsed;
}

/// Collapses an OpenAI-style content value to plain text.
///
/// A string passes through; an array of parts joins the {"type": "text"}
/// parts. Non-text parts (images, audio) are rejected: multimodal input is
/// out of scope for this tokenizer.
StatusOr<ordered_json> NormalizeContent(const ordered_json& content) {
  if (!content.is_array()) return content;
  std::string joined;
  for (const auto& part : content) {
    if (!part.is_object() || !part.contains("type")) {
      return InvalidArgumentError("content parts must be objects with a 'type' field");
    }
    if (part["type"] != "text") {
      return UnimplementedError("multimodal content parts are not supported by this tokenizer");
    }
    if (!part.contains("text") || !part["text"].is_string()) {
      return InvalidArgumentError("text content parts need a string 'text' field");
    }
    if (!joined.empty()) joined += '\n';
    joined += part["text"].get<std::string>();
  }
  return ordered_json(std::move(joined));
}

/// Validates and normalizes the messages array for template rendering:
/// each message needs a string role; content is collapsed to text; other
/// fields (name, tool_calls, tool_call_id) pass through untouched.
StatusOr<ordered_json> NormalizeMessages(const ordered_json& messages) {
  if (!messages.is_array()) {
    return InvalidArgumentError("messages must be a JSON array");
  }
  ordered_json normalized = ordered_json::array();
  for (const auto& raw : messages) {
    if (!raw.is_object() || !raw.contains("role") || !raw["role"].is_string()) {
      return InvalidArgumentError("each message must be an object with a string 'role'");
    }
    ordered_json message;
    message["role"] = raw["role"];
    if (raw.contains("content")) {
      INFERX_ASSIGN_OR_RETURN(auto content, NormalizeContent(raw["content"]));
      message["content"] = std::move(content);
    }
    for (const auto& field : raw.items()) {
      if (field.key() != "role" && field.key() != "content") {
        message[field.key()] = field.value();
      }
    }
    normalized.push_back(std::move(message));
  }
  return normalized;
}

}  // namespace

/// \brief Backend state and cached properties behind HFTokenizer.
class HFTokenizer::Impl {
 public:
  static StatusOr<std::unique_ptr<Impl>> Create(const HFTokenizerAssets& assets);

  const std::vector<std::string>& AllSpecialTokens() const noexcept {
    return all_special_tokens_;
  }
  const std::vector<TokenId>& AllSpecialIds() const noexcept { return all_special_ids_; }
  std::optional<TokenId> BosTokenId() const noexcept { return bos_id_; }
  std::optional<TokenId> EosTokenId() const noexcept { return eos_id_; }
  std::optional<TokenId> PadTokenId() const noexcept { return pad_id_; }
  size_t VocabSize() const noexcept { return json_info_.base_vocab_size; }
  size_t Size() const noexcept { return size_; }
  TokenId MaxTokenId() const noexcept { return json_info_.max_token_id; }
  TruncationSide GetTruncationSide() const noexcept { return metadata_.truncation_side; }
  std::optional<int64_t> ModelMaxLength() const noexcept { return metadata_.model_max_length; }
  bool IsSpecialTokenId(int32_t id) const noexcept {
    return special_ids_.contains(id);
  }
  const std::unordered_map<std::string, TokenId>& GetAddedVocab() const noexcept {
    return added_vocab_;
  }

  const std::unordered_map<std::string, TokenId>& GetVocab() const {
    if (!vocab_.has_value()) {
      std::unordered_map<std::string, TokenId> vocab;
      vocab.reserve(json_info_.base_vocab.size() + json_info_.added_tokens.size());
      for (const auto& [token, id] : json_info_.base_vocab) {
        vocab.emplace(token, id);
      }
      // Added tokens win over same-string base entries, as in HF get_vocab().
      for (const auto& [token, id] : added_vocab_) {
        vocab.insert_or_assign(token, id);
      }
      vocab_ = std::move(vocab);
    }
    return *vocab_;
  }

  StatusOr<std::vector<TokenId>> Encode(std::string_view text,
                                        const EncodeOptions& options) const {
    if (!IsValidUtf8(text)) {
      return InvalidArgumentError("prompt text is not valid UTF-8");
    }
    std::vector<TokenId> ids = rust_.Encode(text, options.add_special_tokens);
    Truncate(ids, options);
    return ids;
  }

  StatusOr<std::vector<std::vector<TokenId>>> EncodeBatch(std::span<const std::string> texts,
                                                          const EncodeOptions& options) const {
    for (const std::string& text : texts) {
      if (!IsValidUtf8(text)) {
        return InvalidArgumentError("prompt text is not valid UTF-8");
      }
    }
    std::vector<std::string> copies(texts.begin(), texts.end());
    std::vector<std::vector<TokenId>> batch = rust_.EncodeBatch(copies, options.add_special_tokens);
    for (auto& ids : batch) Truncate(ids, options);
    return batch;
  }

  std::string Decode(std::span<const TokenId> ids, bool skip_special_tokens) const {
    // Out-of-vocabulary ids are ignored, per the Tokenizer contract.
    std::vector<TokenId> in_range;
    in_range.reserve(ids.size());
    for (TokenId id : ids) {
      if (id >= 0 && static_cast<size_t>(id) < size_) in_range.push_back(id);
    }
    return rust_.Decode(in_range, skip_special_tokens);
  }

  std::vector<std::string> ConvertIdsToTokens(std::span<const TokenId> ids,
                                              bool skip_special_tokens) const {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (TokenId id : ids) {
      if (skip_special_tokens && IsSpecialTokenId(id)) continue;
      if (id < 0 || static_cast<size_t>(id) >= size_) {
        tokens.emplace_back();
        continue;
      }
      tokens.push_back(rust_.IdToToken(id));
    }
    return tokens;
  }

  StatusOr<std::vector<TokenId>> ConvertTokensToIds(std::span<const std::string> tokens) const {
    std::vector<TokenId> ids;
    ids.reserve(tokens.size());
    for (const std::string& token : tokens) {
      const TokenId id = rust_.TokenToId(token);
      if (id < 0) {
        return InvalidArgumentError("token '", token, "' is not in the vocabulary");
      }
      ids.push_back(id);
    }
    return ids;
  }

  StatusOr<std::string> ConvertTokensToString(std::span<const std::string> tokens) const {
    INFERX_ASSIGN_OR_RETURN(auto ids, ConvertTokensToIds(tokens));
    return Decode(ids, /*skip_special_tokens=*/false);
  }

  StatusOr<std::string> ApplyChatTemplate(std::string_view messages_json,
                                          const ChatTemplateOptions& options) const;

 private:
  Status CreateInternal(const HFTokenizerAssets& assets);

  /// Applies EncodeOptions truncation in place, mirroring the HF sides.
  void Truncate(std::vector<TokenId>& ids, const EncodeOptions& options) const {
    if (!options.truncation) return;
    const std::optional<size_t> max_length = options.max_length.has_value()
                                                 ? options.max_length
                                                 : EffectiveModelMaxLength();
    if (!max_length.has_value() || ids.size() <= *max_length) return;
    if (metadata_.truncation_side == TruncationSide::kLeft) {
      ids.erase(ids.begin(), ids.end() - static_cast<std::ptrdiff_t>(*max_length));
    } else {
      ids.resize(*max_length);
    }
  }

  std::optional<size_t> EffectiveModelMaxLength() const {
    if (!metadata_.model_max_length.has_value()) return std::nullopt;
    const int64_t length = *metadata_.model_max_length;
    if (length <= 0) return std::nullopt;
    return static_cast<size_t>(length);
  }

  /// The renderer for a per-request template override, cached by source.
  StatusOr<const ChatTemplateRenderer*> OverrideRenderer(const std::string& source) const;

  tokenizers::RustTokenizer rust_;
  TokenizerJsonInfo json_info_;
  TokenizerMetadata metadata_;
  size_t size_ = 0;

  std::vector<std::string> all_special_tokens_;
  std::vector<TokenId> all_special_ids_;
  std::unordered_set<TokenId> special_ids_;
  std::optional<TokenId> bos_id_;
  std::optional<TokenId> eos_id_;
  std::optional<TokenId> pad_id_;
  std::unordered_map<std::string, TokenId> added_vocab_;
  mutable std::optional<std::unordered_map<std::string, TokenId>> vocab_;

  std::string chat_template_error_;
  std::optional<ChatTemplateRenderer> chat_template_;
  mutable std::optional<std::pair<std::string, ChatTemplateRenderer>> override_cache_;
};

StatusOr<std::unique_ptr<HFTokenizer::Impl>> HFTokenizer::Impl::Create(
    const HFTokenizerAssets& assets) {
  auto impl = std::make_unique<Impl>();
  INFERX_RETURN_IF_ERROR(impl->CreateInternal(assets));
  return impl;
}

Status HFTokenizer::Impl::CreateInternal(const HFTokenizerAssets& assets) {
  const std::string& where =
      assets.source_dir.empty() ? std::string("tokenizer assets") : assets.source_dir;

  nlohmann::json tokenizer_json;
  try {
    tokenizer_json = nlohmann::json::parse(assets.tokenizer_json);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError(where, "/tokenizer.json is not valid JSON: ", e.what());
  }
  INFERX_ASSIGN_OR_RETURN(json_info_, ParseTokenizerJson(tokenizer_json));

  INFERX_ASSIGN_OR_RETURN(auto tokenizer_config,
                          ParseConfigBlob(assets.tokenizer_config_json,
                                          where + "/tokenizer_config.json"));
  INFERX_ASSIGN_OR_RETURN(auto special_tokens_map,
                          ParseConfigBlob(assets.special_tokens_map_json,
                                          where + "/special_tokens_map.json"));
  metadata_ = MergeMetadata(tokenizer_config.get(), special_tokens_map.get());

  // Create() surfaces Unimplemented when the tokenizers-cpp backend was
  // compiled out, which is how backend-less builds fail construction.
  INFERX_ASSIGN_OR_RETURN(rust_, tokenizers::RustTokenizer::Create(assets.tokenizer_json));
  size_ = rust_.VocabSize();

  for (const auto& token : json_info_.added_tokens) {
    added_vocab_.emplace(token.content, token.id);
  }

  // Special-token tables follow transformers' all_special_tokens order:
  // the named tokens, then additional, then special added tokens.
  auto resolve = [this](const std::string& content) -> std::optional<TokenId> {
    if (content.empty()) return std::nullopt;
    const TokenId id = rust_.TokenToId(content);
    if (id < 0) return std::nullopt;
    return id;
  };
  auto register_token = [&](const std::string& content, std::optional<TokenId> id) {
    if (content.empty()) return;
    if (std::find(all_special_tokens_.begin(), all_special_tokens_.end(), content) !=
        all_special_tokens_.end()) {
      return;
    }
    all_special_tokens_.push_back(content);
    if (id.has_value()) {
      all_special_ids_.push_back(*id);
      special_ids_.insert(*id);
    }
  };

  bos_id_ = resolve(metadata_.bos_token);
  eos_id_ = resolve(metadata_.eos_token);
  pad_id_ = resolve(metadata_.pad_token);
  register_token(metadata_.bos_token, bos_id_);
  register_token(metadata_.eos_token, eos_id_);
  register_token(metadata_.unk_token, resolve(metadata_.unk_token));
  register_token(metadata_.pad_token, pad_id_);
  for (const std::string& token : metadata_.additional_special_tokens) {
    register_token(token, resolve(token));
  }
  for (const auto& token : json_info_.added_tokens) {
    if (token.special) register_token(token.content, token.id);
  }

  if (!metadata_.chat_template.empty()) {
    auto renderer = ChatTemplateRenderer::Create(metadata_.chat_template, metadata_.bos_token,
                                                 metadata_.eos_token);
    if (renderer.ok()) {
      chat_template_ = std::move(*renderer);
    } else {
      chat_template_error_ = std::string(renderer.status().message());
    }
  }
  return OkStatus();
}

StatusOr<std::string> HFTokenizer::Impl::ApplyChatTemplate(std::string_view messages_json,
                                                           const ChatTemplateOptions& options) const {
  if (options.continue_final_message && options.add_generation_prompt) {
    return InvalidArgumentError(
        "continue_final_message and add_generation_prompt are mutually exclusive");
  }

  // Pick the renderer: a per-request override, else the checkpoint template.
  const ChatTemplateRenderer* renderer = nullptr;
  if (options.chat_template.has_value()) {
    INFERX_ASSIGN_OR_RETURN(renderer, OverrideRenderer(*options.chat_template));
  } else if (chat_template_.has_value()) {
    renderer = &*chat_template_;
  } else if (!chat_template_error_.empty()) {
    return InternalError("the checkpoint chat template failed to load: ", chat_template_error_);
  } else {
    // Unimplemented keeps the serving layer's 501 "not supported" mapping
    // for checkpoints that ship no template.
    return UnimplementedError(
        "chat_template_not_supported: the checkpoint defines no chat template");
  }

  INFERX_ASSIGN_OR_RETURN(ordered_json messages, ParseOptionalJson(messages_json, "messages"));
  INFERX_ASSIGN_OR_RETURN(messages, NormalizeMessages(messages));

  // transformers >= 5 continue_final_message: a sentinel tag is appended to
  // the final message's content before rendering, then the rendered prompt
  // is cut at the tag, preserving the pre-filled text's exact spacing.
  constexpr std::string_view kContinueTag = "CONTINUE_FINAL_MESSAGE_TAG ";
  if (options.continue_final_message) {
    if (messages.empty() || !messages.back().contains("content") ||
        !messages.back()["content"].is_string()) {
      return InvalidArgumentError(
          "continue_final_message is set but the final message has no content to continue");
    }
    messages.back()["content"] =
        messages.back()["content"].get<std::string>() + std::string(kContinueTag);
  }

  ordered_json tools;
  if (!options.tools_json.empty()) {
    INFERX_ASSIGN_OR_RETURN(tools, ParseOptionalJson(options.tools_json, "tools"));
    if (!tools.is_array()) {
      return InvalidArgumentError("tools must be a JSON array");
    }
  }
  ordered_json extra_context;
  if (!options.template_kwargs_json.empty()) {
    INFERX_ASSIGN_OR_RETURN(extra_context,
                            ParseOptionalJson(options.template_kwargs_json, "template kwargs"));
    if (!extra_context.is_object()) {
      return InvalidArgumentError("template kwargs must be a JSON object");
    }
  }

  const bool add_generation_prompt =
      options.continue_final_message ? false : options.add_generation_prompt;
  INFERX_ASSIGN_OR_RETURN(std::string rendered,
                          renderer->Render(messages, tools, add_generation_prompt, extra_context));

  if (options.continue_final_message) {
    const std::string_view stripped_tag = kContinueTag.substr(0, kContinueTag.size() - 1);
    const size_t tag_position = rendered.rfind(stripped_tag);
    if (tag_position == std::string::npos) {
      return InvalidArgumentError(
          "continue_final_message is set but the final message does not appear in the rendered "
          "chat; the template may drop the final message content");
    }
    if (rendered.compare(tag_position, kContinueTag.size(), kContinueTag) == 0) {
      // The template preserved the content's trailing spacing.
      rendered.resize(tag_position);
    } else {
      // The template trimmed it; trim the trailing whitespace too.
      rendered.resize(tag_position);
      const std::string_view trimmed = StripWhitespace(rendered);
      return std::string(trimmed);
    }
  }
  return rendered;
}

StatusOr<const ChatTemplateRenderer*> HFTokenizer::Impl::OverrideRenderer(
    const std::string& source) const {
  if (override_cache_.has_value() && override_cache_->first == source) {
    return &override_cache_->second;
  }
  auto renderer = ChatTemplateRenderer::Create(source, metadata_.bos_token, metadata_.eos_token);
  if (!renderer.ok()) return renderer.status();
  override_cache_ = std::make_pair(source, std::move(*renderer));
  return &override_cache_->second;
}

StatusOr<std::shared_ptr<HFTokenizer>> HFTokenizer::Load(const std::string& tokenizer_dir) {
  HFTokenizerAssets assets;
  assets.source_dir = tokenizer_dir;
  auto tokenizer_blob = ReadFile(tokenizer_dir + "/tokenizer.json");
  if (!tokenizer_blob.has_value()) {
    return NotFoundError("no tokenizer.json found under ", tokenizer_dir);
  }
  assets.tokenizer_json = std::move(*tokenizer_blob);
  if (auto blob = ReadFile(tokenizer_dir + "/tokenizer_config.json")) {
    assets.tokenizer_config_json = std::move(*blob);
  }
  if (auto blob = ReadFile(tokenizer_dir + "/special_tokens_map.json")) {
    assets.special_tokens_map_json = std::move(*blob);
  }
  return FromAssets(assets);
}

StatusOr<std::shared_ptr<HFTokenizer>> HFTokenizer::FromAssets(const HFTokenizerAssets& assets) {
  auto impl = Impl::Create(assets);
  if (!impl.ok()) return impl.status();
  auto tokenizer = std::shared_ptr<HFTokenizer>(new HFTokenizer());
  tokenizer->impl_ = std::move(*impl);
  return tokenizer;
}

const std::vector<std::string>& HFTokenizer::AllSpecialTokens() const noexcept {
  return impl_->AllSpecialTokens();
}
const std::vector<TokenId>& HFTokenizer::AllSpecialIds() const noexcept {
  return impl_->AllSpecialIds();
}
std::optional<TokenId> HFTokenizer::BosTokenId() const noexcept { return impl_->BosTokenId(); }
std::optional<TokenId> HFTokenizer::EosTokenId() const noexcept { return impl_->EosTokenId(); }
std::optional<TokenId> HFTokenizer::PadTokenId() const noexcept { return impl_->PadTokenId(); }
size_t HFTokenizer::VocabSize() const noexcept { return impl_->VocabSize(); }
size_t HFTokenizer::Size() const noexcept { return impl_->Size(); }
TokenId HFTokenizer::MaxTokenId() const noexcept { return impl_->MaxTokenId(); }
TruncationSide HFTokenizer::GetTruncationSide() const noexcept { return impl_->GetTruncationSide(); }
std::optional<int64_t> HFTokenizer::ModelMaxLength() const noexcept {
  return impl_->ModelMaxLength();
}
bool HFTokenizer::IsSpecialTokenId(int32_t id) const noexcept {
  return impl_->IsSpecialTokenId(id);
}
const std::unordered_map<std::string, TokenId>& HFTokenizer::GetVocab() const {
  return impl_->GetVocab();
}
const std::unordered_map<std::string, TokenId>& HFTokenizer::GetAddedVocab() const {
  return impl_->GetAddedVocab();
}
StatusOr<std::vector<TokenId>> HFTokenizer::Encode(std::string_view text,
                                                   const EncodeOptions& options) const {
  return impl_->Encode(text, options);
}
StatusOr<std::vector<std::vector<TokenId>>> HFTokenizer::EncodeBatch(
    std::span<const std::string> texts, const EncodeOptions& options) const {
  return impl_->EncodeBatch(texts, options);
}
std::string HFTokenizer::Decode(std::span<const TokenId> ids, bool skip_special_tokens) const {
  return impl_->Decode(ids, skip_special_tokens);
}
std::vector<std::string> HFTokenizer::ConvertIdsToTokens(std::span<const TokenId> ids,
                                                         bool skip_special_tokens) const {
  return impl_->ConvertIdsToTokens(ids, skip_special_tokens);
}
StatusOr<std::vector<TokenId>> HFTokenizer::ConvertTokensToIds(
    std::span<const std::string> tokens) const {
  return impl_->ConvertTokensToIds(tokens);
}
StatusOr<std::string> HFTokenizer::ConvertTokensToString(
    std::span<const std::string> tokens) const {
  return impl_->ConvertTokensToString(tokens);
}
StatusOr<std::string> HFTokenizer::ApplyChatTemplate(std::string_view messages_json,
                                                     const ChatTemplateOptions& options) const {
  return impl_->ApplyChatTemplate(messages_json, options);
}

}  // namespace inferx
