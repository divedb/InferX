#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/status.h"

namespace tokenizers {
class Tokenizer;
}

namespace inferx::tokenizer {

using TokenId = int32_t;

/// Whether strings which spell registered special tokens are interpreted as
/// control tokens. User-controlled text must always use kAsText.
enum class SpecialTokenMode {
  kAsControl,
  kAsText,
};

struct EncodeOptions {
  SpecialTokenMode special_tokens = SpecialTokenMode::kAsText;
  bool add_post_processor_tokens = false;
};

struct TokenizerMetadata {
  std::string backend;
  std::string backend_version;
  int64_t vocabulary_size = 0;
  std::optional<TokenId> bos_id;
  std::optional<TokenId> eos_id;
  std::optional<TokenId> pad_id;
  std::vector<TokenId> stop_ids;
  bool supports_incremental_decode = false;
};

class Tokenizer;

/// Per-request decoder state. The borrowed request-confined Tokenizer must
/// outlive it; neither object may be shared with another request.
class IncrementalDecoder {
 public:
  explicit IncrementalDecoder(Tokenizer* tokenizer, bool skip_special = true);
  ~IncrementalDecoder();

  IncrementalDecoder(const IncrementalDecoder&) = delete;
  IncrementalDecoder& operator=(const IncrementalDecoder&) = delete;
  IncrementalDecoder(IncrementalDecoder&&) noexcept;
  IncrementalDecoder& operator=(IncrementalDecoder&&) noexcept;

  /// Adds one token and returns text which is now stable and safe to emit.
  std::string Push(TokenId id);

  /// Finishes the stream and returns the suffix retained until it became safe.
  std::string Flush();

 private:
  class Impl;

  std::unique_ptr<Impl> impl_;
};

/// Backend-neutral tokenizer used by the rest of InferX.
///
/// Thin InferX adapter around tokenizers::Tokenizer. The methods matching the
/// tokenizers-cpp interface forward directly to the owned backend. Because that
/// backend owns mutable result buffers, an instance is request-confined; use
/// Clone() to give concurrent requests independent backend handles.
class Tokenizer {
 public:
  ~Tokenizer();

  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;

  /// Loads a complete Hugging Face tokenizer graph from tokenizer.json.
  static StatusOr<std::unique_ptr<Tokenizer>> LoadFromFile(
      const std::string& path);

  /// Discovers tokenizer artifacts and checkpoint token metadata in dir.
  static StatusOr<std::unique_ptr<Tokenizer>> LoadFromDirectory(
      const std::string& dir);

  /// Creates an independent tokenizer with its own backend handle and mutable
  /// third-party buffers. Prefer one clone per concurrent request.
  StatusOr<std::unique_ptr<Tokenizer>> Clone() const;

  // tokenizers::Tokenizer-compatible forwarding surface.
  std::vector<TokenId> Encode(const std::string& text);
  std::vector<std::vector<TokenId>> EncodeBatch(
      const std::vector<std::string>& texts);
  std::string Decode(const std::vector<TokenId>& ids);
  size_t GetVocabSize();
  std::string IdToToken(TokenId id);
  TokenId TokenToId(const std::string& token);

  // InferX-specific policy and checked operations.
  StatusOr<std::vector<TokenId>> EncodeWithOptions(
      std::string_view text, const EncodeOptions& options);

  /// Untrusted text path: registered special-token spellings remain text.
  std::vector<TokenId> EncodeOrdinary(std::string_view text);

  StatusOr<std::string> DecodeChecked(const std::vector<TokenId>& ids,
                                      bool skip_special = false);
  std::string Decode(const std::vector<TokenId>& ids, bool skip_special);

  int32_t VocabSize() { return static_cast<int32_t>(GetVocabSize()); }
  std::optional<TokenId> FindTokenId(std::string_view token);
  bool IsSpecial(TokenId id) const;

  const TokenizerMetadata& info() const { return info_; }
  TokenId eos_id() const { return info_.eos_id.value_or(-1); }

 private:
  struct Config;
  friend class IncrementalDecoder;

  Tokenizer(std::unique_ptr<::tokenizers::Tokenizer> backend,
            std::shared_ptr<Config> config);

  Status ResolveCheckpointMetadata(const std::string& dir);
  Status Validate();

  std::unique_ptr<::tokenizers::Tokenizer> backend_;
  std::shared_ptr<Config> config_;
  TokenizerMetadata info_;
};

}  // namespace inferx::tokenizer
