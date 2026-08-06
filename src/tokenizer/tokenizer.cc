#include "inferx/tokenizer/tokenizer.h"

#include <tokenizers_cpp.h>

#include <algorithm>
#include <filesystem>
#include <limits>
#include <unordered_map>
#include <utility>

#include "inferx/support/file_util.h"
#include "inferx/support/json.h"

namespace inferx::tokenizer {
namespace {

enum class ArtifactKind { kHuggingFaceJson, kSentencePiece };

// Returns the length of the first UTF-8 code point. Invalid input is consumed
// one byte at a time; the backend will apply its own invalid-input policy.
size_t FirstCodepointLength(std::string_view text) {
  if (text.empty()) return 0;
  const unsigned char lead = static_cast<unsigned char>(text.front());
  if (lead < 0x80) return 1;
  if ((lead & 0xE0) == 0xC0 && text.size() >= 2) return 2;
  if ((lead & 0xF0) == 0xE0 && text.size() >= 3) return 3;
  if ((lead & 0xF8) == 0xF0 && text.size() >= 4) return 4;
  return 1;
}

size_t Utf8BoundaryAtOrBefore(std::string_view text, size_t offset) {
  offset = std::min(offset, text.size());
  while (offset > 0 && offset < text.size() &&
         (static_cast<unsigned char>(text[offset]) & 0xC0) == 0x80) {
    --offset;
  }
  return offset;
}

}  // namespace

struct Tokenizer::Config {
  ArtifactKind kind;
  std::string artifact;
  std::string tokenizer_config;
  std::vector<std::string> added_contents;
  std::unordered_map<TokenId, bool> special_ids;

  Status LoadAddedTokens(const JsonValue& root) {
    const JsonValue* added = root.Find("added_tokens");
    if (added == nullptr || added->IsNull()) return OkStatus();
    INFERX_ASSIGN_OR_RETURN(const std::vector<JsonValue>* list,
                            added->AsArray());
    for (const JsonValue& entry : *list) {
      INFERX_ASSIGN_OR_RETURN(const std::string_view content,
                              entry.RequiredString("content"));
      INFERX_ASSIGN_OR_RETURN(const int64_t raw_id, entry.RequiredInt("id"));
      if (raw_id < 0 || raw_id > std::numeric_limits<TokenId>::max()) {
        return OutOfRangeError("added token ID does not fit int32_t: ", raw_id);
      }
      const TokenId id = static_cast<TokenId>(raw_id);
      added_contents.push_back(std::string(content));
      INFERX_ASSIGN_OR_RETURN(const bool special,
                              entry.OptionalBool("special", false));
      if (special) special_ids[id] = true;
    }
    std::sort(added_contents.begin(), added_contents.end(),
              [](const std::string& a, const std::string& b) {
                return a.size() > b.size();
              });
    return OkStatus();
  }
};

StatusOr<std::unique_ptr<::tokenizers::Tokenizer>> CreateBackend(
    ArtifactKind kind, const std::string& artifact,
    const std::string& tokenizer_config = {}) {
  std::unique_ptr<::tokenizers::Tokenizer> backend =
      kind == ArtifactKind::kHuggingFaceJson
          ? (tokenizer_config.empty()
                 ? ::tokenizers::Tokenizer::FromBlobJSON(artifact)
                 : ::tokenizers::Tokenizer::FromBlobJSON(artifact,
                                                         tokenizer_config))
          : ::tokenizers::Tokenizer::FromBlobSentencePiece(artifact);
  if (backend == nullptr) {
    return InvalidArgumentError("tokenizers-cpp rejected tokenizer artifact");
  }
  return backend;
}

Tokenizer::Tokenizer(std::unique_ptr<::tokenizers::Tokenizer> backend,
                     std::shared_ptr<Config> config)
    : backend_(std::move(backend)), config_(std::move(config)) {
  info_.backend = config_->kind == ArtifactKind::kHuggingFaceJson
                      ? "tokenizers-cpp/huggingface"
                      : "tokenizers-cpp/sentencepiece";
  info_.backend_version = "c586c52f93f7b060753bd2388eb96a105cb7374d";
  info_.vocabulary_size = static_cast<int64_t>(GetVocabSize());
  info_.supports_incremental_decode = true;
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<TokenId> Tokenizer::Encode(const std::string& text) {
  return backend_->Encode(text);
}

std::vector<std::vector<TokenId>> Tokenizer::EncodeBatch(
    const std::vector<std::string>& texts) {
  return backend_->EncodeBatch(texts);
}

std::string Tokenizer::Decode(const std::vector<TokenId>& ids) {
  return backend_->Decode(ids);
}

size_t Tokenizer::GetVocabSize() { return backend_->GetVocabSize(); }

std::string Tokenizer::IdToToken(TokenId id) {
  return id < 0 ? std::string() : backend_->IdToToken(id);
}

TokenId Tokenizer::TokenToId(const std::string& token) {
  return backend_->TokenToId(token);
}

std::optional<TokenId> Tokenizer::FindTokenId(std::string_view token) {
  const TokenId id = TokenToId(std::string(token));
  return id < 0 ? std::nullopt : std::optional<TokenId>(id);
}

StatusOr<std::vector<TokenId>> Tokenizer::EncodeWithOptions(
    std::string_view text, const EncodeOptions& options) {
  if (options.add_post_processor_tokens) {
    return UnimplementedError(
        "tokenizers-cpp does not expose post-processor token insertion");
  }
  if (options.special_tokens == SpecialTokenMode::kAsControl ||
      config_->added_contents.empty()) {
    return Encode(std::string(text));
  }

  std::vector<TokenId> out;
  size_t span = 0;
  size_t p = 0;
  while (p < text.size()) {
    const std::string* hit = nullptr;
    for (const std::string& content : config_->added_contents) {
      if (!content.empty() && text.substr(p).starts_with(content)) {
        hit = &content;
        break;
      }
    }
    if (hit == nullptr) {
      ++p;
      continue;
    }
    if (p > span) {
      std::vector<TokenId> ids =
          Encode(std::string(text.substr(span, p - span)));
      out.insert(out.end(), ids.begin(), ids.end());
    }

    // The upstream common API cannot toggle added-token recognition. Encode
    // the spelling a code point at a time so no multi-character control token
    // can be recognized. Single-codepoint added tokens cannot be represented
    // safely through this API and are rejected.
    std::string_view remaining(*hit);
    if (FirstCodepointLength(remaining) == remaining.size()) {
      return UnimplementedError(
          "ordinary-text encoding of a single-codepoint added token requires "
          "a tokenizers-cpp API extension");
    }
    while (!remaining.empty()) {
      const size_t length = FirstCodepointLength(remaining);
      std::vector<TokenId> ids =
          Encode(std::string(remaining.substr(0, length)));
      out.insert(out.end(), ids.begin(), ids.end());
      remaining.remove_prefix(length);
    }
    p += hit->size();
    span = p;
  }
  if (span < text.size()) {
    std::vector<TokenId> ids = Encode(std::string(text.substr(span)));
    out.insert(out.end(), ids.begin(), ids.end());
  }
  return out;
}

StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::LoadFromFile(
    const std::string& path) {
  INFERX_ASSIGN_OR_RETURN(const std::string blob, ReadFile(path));

  const std::filesystem::path artifact(path);
  const ArtifactKind kind = artifact.extension() == ".json"
                                ? ArtifactKind::kHuggingFaceJson
                                : ArtifactKind::kSentencePiece;
  auto config = std::make_shared<Config>();
  config->kind = kind;
  config->artifact = blob;
  if (kind == ArtifactKind::kHuggingFaceJson) {
    INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(blob));
    if (root.Find("model") == nullptr) {
      return InvalidArgumentError("tokenizer.json has no model component");
    }
    INFERX_RETURN_IF_ERROR(config->LoadAddedTokens(root));
  }
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<::tokenizers::Tokenizer> backend,
                          CreateBackend(kind, blob));
  auto tokenizer = std::unique_ptr<Tokenizer>(
      new Tokenizer(std::move(backend), std::move(config)));
  INFERX_RETURN_IF_ERROR(tokenizer->Validate());
  return tokenizer;
}

StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::LoadFromDirectory(
    const std::string& dir) {
  std::string artifact;
  for (std::string_view candidate :
       {"tokenizer.json", "tokenizer.model", "spiece.model"}) {
    const std::string path = dir + "/" + std::string(candidate);
    if (std::filesystem::is_regular_file(path)) {
      artifact = path;
      break;
    }
  }
  if (artifact.empty()) {
    return NotFoundError("checkpoint has no supported tokenizer artifact: ",
                         dir);
  }
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                          LoadFromFile(artifact));
  INFERX_RETURN_IF_ERROR(tokenizer->ResolveCheckpointMetadata(dir));
  INFERX_RETURN_IF_ERROR(tokenizer->Validate());
  return tokenizer;
}

StatusOr<std::unique_ptr<Tokenizer>> Tokenizer::Clone() const {
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<::tokenizers::Tokenizer> backend,
                          CreateBackend(config_->kind, config_->artifact,
                                        config_->tokenizer_config));
  auto clone =
      std::unique_ptr<Tokenizer>(new Tokenizer(std::move(backend), config_));
  clone->info_ = info_;
  return clone;
}

Status Tokenizer::ResolveCheckpointMetadata(const std::string& dir) {
  const StatusOr<std::string> text = ReadFile(dir + "/tokenizer_config.json");
  if (text.ok()) {
    config_->tokenizer_config = *text;
    INFERX_ASSIGN_OR_RETURN(backend_,
                            CreateBackend(config_->kind, config_->artifact,
                                          config_->tokenizer_config));
    const auto copy_id = [&](TokenId id, std::optional<TokenId>* destination) {
      if (id >= 0) {
        *destination = id;
        config_->special_ids[id] = true;
      }
    };
    copy_id(backend_->GetBosTokenId(), &info_.bos_id);
    copy_id(backend_->GetEosTokenId(), &info_.eos_id);
    copy_id(backend_->GetPadTokenId(), &info_.pad_id);
  } else if (!absl::IsNotFound(text.status())) {
    return text.status();
  }

  if (!info_.eos_id.has_value()) {
    for (std::string_view candidate : {"<|im_end|>", "<|endoftext|>"}) {
      if (const std::optional<TokenId> id = FindTokenId(candidate)) {
        info_.eos_id = *id;
        config_->special_ids[*id] = true;
        break;
      }
    }
  }
  if (info_.eos_id.has_value()) info_.stop_ids = {*info_.eos_id};
  return OkStatus();
}

Status Tokenizer::Validate() {
  if (info_.vocabulary_size <= 0 ||
      info_.vocabulary_size > std::numeric_limits<TokenId>::max()) {
    return InvalidArgumentError("invalid tokenizer vocabulary size: ",
                                info_.vocabulary_size);
  }
  for (const std::optional<TokenId> id :
       {info_.bos_id, info_.eos_id, info_.pad_id}) {
    if (id.has_value() && IdToToken(*id).empty()) {
      return InvalidArgumentError("configured token ID ", *id,
                                  " is not resolvable by the backend");
    }
  }
  return OkStatus();
}

std::vector<TokenId> Tokenizer::EncodeOrdinary(std::string_view text) {
  StatusOr<std::vector<TokenId>> result = EncodeWithOptions(text, {});
  return result.ok() ? std::move(*result) : std::vector<TokenId>();
}

StatusOr<std::string> Tokenizer::DecodeChecked(const std::vector<TokenId>& ids,
                                               bool skip_special) {
  if (!skip_special || config_->special_ids.empty()) return Decode(ids);
  std::vector<TokenId> filtered;
  filtered.reserve(ids.size());
  for (TokenId id : ids) {
    if (config_->special_ids.count(id) == 0) filtered.push_back(id);
  }
  return Decode(filtered);
}

std::string Tokenizer::Decode(const std::vector<TokenId>& ids,
                              bool skip_special) {
  StatusOr<std::string> result = DecodeChecked(ids, skip_special);
  return result.ok() ? std::move(*result) : std::string();
}

bool Tokenizer::IsSpecial(TokenId id) const {
  return config_->special_ids.count(id) != 0;
}

class IncrementalDecoder::Impl {
 public:
  Impl(Tokenizer* tokenizer, bool skip_special)
      : tokenizer_(tokenizer), skip_special_(skip_special) {}

  std::string Push(TokenId id) {
    if (tokenizer_ == nullptr || id < 0) return {};
    ids_.push_back(id);
    const std::string next = tokenizer_->Decode(ids_, skip_special_);
    size_t common = 0;
    const size_t limit = std::min(decoded_.size(), next.size());
    while (common < limit && decoded_[common] == next[common]) ++common;
    common = Utf8BoundaryAtOrBefore(decoded_, common);

    std::string out;
    if (common > emitted_) out = decoded_.substr(emitted_, common - emitted_);
    emitted_ = common;
    decoded_ = next;
    return out;
  }

  std::string Flush() {
    if (emitted_ >= decoded_.size()) return {};
    std::string out = decoded_.substr(emitted_);
    emitted_ = decoded_.size();
    return out;
  }

 private:
  Tokenizer* tokenizer_;
  bool skip_special_;
  std::vector<TokenId> ids_;
  std::string decoded_;
  size_t emitted_ = 0;
};

IncrementalDecoder::IncrementalDecoder(Tokenizer* tokenizer, bool skip_special)
    : impl_(std::make_unique<Impl>(tokenizer, skip_special)) {}
IncrementalDecoder::~IncrementalDecoder() = default;
IncrementalDecoder::IncrementalDecoder(IncrementalDecoder&&) noexcept = default;
IncrementalDecoder& IncrementalDecoder::operator=(
    IncrementalDecoder&&) noexcept = default;
std::string IncrementalDecoder::Push(TokenId id) { return impl_->Push(id); }
std::string IncrementalDecoder::Flush() { return impl_->Flush(); }

}  // namespace inferx::tokenizer
