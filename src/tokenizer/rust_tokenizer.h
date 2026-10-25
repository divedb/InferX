/// \brief RAII owner of a tokenizers-cpp Rust tokenizer handle.
///
/// Thin typed wrapper over the C API in tokenizers_c.h. The C++ façade in
/// tokenizers_cpp.h hides the add_special_tokens / skip_special_tokens
/// flags (its virtual methods hardcode false), so the tokenizer subsystem
/// drives the C API directly; every entry point here pre-validates inputs
/// because the Rust side unwraps errors and panics (aborting the process)
/// on malformed JSON or invalid UTF-8.
///
/// The Rust wrapper keeps per-call mutable state (`&mut self`), so a
/// RustTokenizer must not be used concurrently from multiple threads.
#ifndef INFERX_TOKENIZER_RUST_TOKENIZER_H_
#define INFERX_TOKENIZER_RUST_TOKENIZER_H_

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/tokenizer/id.h"

namespace inferx::tokenizers {

/// \brief Owns one `TokenizerHandle` from tokenizers-cpp.
class RustTokenizer {
 public:
  /// \brief Creates a tokenizer from a `tokenizer.json` blob.
  ///
  /// The blob must already be validated: valid UTF-8, valid JSON, and a
  /// fast-tokenizer document (top-level object with a `model` member), so
  /// the Rust constructor cannot hit its deserialization panic.
  static StatusOr<RustTokenizer> Create(std::string_view tokenizer_json);

  RustTokenizer() = default;
  RustTokenizer(const RustTokenizer&) = delete;
  RustTokenizer& operator=(const RustTokenizer&) = delete;
  RustTokenizer(RustTokenizer&& other) noexcept;
  RustTokenizer& operator=(RustTokenizer&& other) noexcept;
  ~RustTokenizer();

  /// \brief Encodes UTF-8 text, optionally adding template special tokens.
  std::vector<TokenId> Encode(std::string_view text, bool add_special_tokens) const;

  /// \brief Encodes a batch; every text must be valid UTF-8.
  std::vector<std::vector<TokenId>> EncodeBatch(const std::vector<std::string>& texts,
                                                bool add_special_tokens) const;

  /// \brief Decodes token ids to text; ids must be within [0, VocabSize()).
  std::string Decode(std::span<const TokenId> ids, bool skip_special_tokens) const;

  /// \brief Returns the token string for an id, or "" when unknown.
  std::string IdToToken(TokenId id) const;

  /// \brief Returns the id for a token string, or -1 when unknown.
  TokenId TokenToId(std::string_view token) const;

  /// \brief Total vocabulary size including added tokens (len(tokenizer)).
  size_t VocabSize() const;

 private:
  void* handle_ = nullptr;  // TokenizerHandle, kept opaque for non-backend builds.
};

}  // namespace inferx::tokenizers

#endif  // INFERX_TOKENIZER_RUST_TOKENIZER_H_
