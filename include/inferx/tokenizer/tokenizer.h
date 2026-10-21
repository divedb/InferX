#ifndef INFERX_TOKENIZER_H_
#define INFERX_TOKENIZER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "inferx/core/status.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

/// \brief Options for encoding text into token ids.
struct EncodeOptions {
  /// False for chat-templated prompts that already contain BOS/EOS; true for
  /// raw API text that expects the model's special tokens prepended.
  bool add_special_tokens = false;
  bool return_token_type_ids = false;
};

/// \brief Options for decoding token ids into text.
struct DecodeOptions {
  /// OpenAI-style special-token skipping; an adapter that wants it must set
  /// it explicitly. The compatibility default matches the backend's plain
  /// Decode.
  bool skip_special_tokens = false;
  bool spaces_between_special_tokens = true;
};

/// \brief Vocabulary metadata needed for validation and stop-token policy.
struct TokenizerInfo {
  int64_t vocab_size = 0;
  std::vector<TokenId> eos_token_ids;
};

/// \brief Interface for converting between text and token ids.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  /// \brief Encodes text into token ids with the compatibility defaults.
  virtual StatusOr<std::vector<int>> Encode(absl::string_view text) const = 0;
  /// \brief Decodes token ids into text with the compatibility defaults.
  virtual StatusOr<std::string> Decode(absl::Span<const int> ids) const = 0;
  /// \brief Returns the vocabulary size.
  virtual int64_t vocab_size() const = 0;

  /// \brief Encodes with explicit options. Backends without special-token
  /// control return Unimplemented rather than silently ignoring the option.
  virtual StatusOr<std::vector<int>> Encode(absl::string_view text,
                                            const EncodeOptions& options) const;

  /// \brief Decodes with explicit options; same Unimplemented rule.
  virtual StatusOr<std::string> Decode(absl::Span<const int> ids,
                                       const DecodeOptions& options) const;

  /// \brief Encodes many texts; a backend with native batching overrides
  /// this. Per-item failures fail the whole call; callers that need
  /// per-item errors encode individually.
  virtual StatusOr<std::vector<std::vector<int>>> EncodeBatch(
      absl::Span<const std::string> texts, const EncodeOptions& options) const;

  /// \brief Vocabulary metadata for validation and stop-token policy.
  virtual TokenizerInfo Info() const;

  /// \brief Creates a tokenizer from a `tokenizer.json` file.
  ///
  /// \param path Path to the tokenizer file.
  /// \return The tokenizer, or an error status if the file cannot be loaded
  /// or tokenizer support is disabled at build time.
  static StatusOr<std::shared_ptr<Tokenizer>> FromFile(const std::string& path);
};

inline StatusOr<std::vector<int>> Tokenizer::Encode(absl::string_view text,
                                                    const EncodeOptions& options) const {
  if (options.add_special_tokens || options.return_token_type_ids) {
    return UnimplementedError("this tokenizer does not expose encode options");
  }
  return Encode(text);
}

inline StatusOr<std::string> Tokenizer::Decode(absl::Span<const int> ids,
                                               const DecodeOptions& options) const {
  if (options.skip_special_tokens || !options.spaces_between_special_tokens) {
    return UnimplementedError("this tokenizer does not expose decode options");
  }
  return Decode(ids);
}

inline StatusOr<std::vector<std::vector<int>>> Tokenizer::EncodeBatch(
    absl::Span<const std::string> texts, const EncodeOptions& options) const {
  std::vector<std::vector<int>> out;
  out.reserve(texts.size());
  for (const std::string& text : texts) {
    auto ids = Encode(text, options);
    if (!ids.ok()) return ids.status();
    out.push_back(*std::move(ids));
  }
  return out;
}

inline TokenizerInfo Tokenizer::Info() const { return TokenizerInfo{vocab_size(), {}}; }

}  // namespace inferx

#endif  // INFERX_TOKENIZER_H_
