/// \file
/// \brief Tokenizer facade.
///
/// The engine depends only on this interface. The default implementation wraps
/// HuggingFace `tokenizers` via tokenizers-cpp; if tokenizer support is
/// disabled at build time the factory returns an error and callers can operate
/// on raw token ids.

#ifndef INFERX_TOKENIZER_H_
#define INFERX_TOKENIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace inferx {

/// \brief Interface for converting between text and token ids.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  /// \brief Encodes text into token ids.
  ///
  /// \param text Input text.
  /// \return The token ids, or an error status.
  virtual absl::StatusOr<std::vector<int>> Encode(
      absl::string_view text) const = 0;
  /// \brief Decodes token ids into text.
  ///
  /// \param ids Token ids to decode.
  /// \return The decoded text, or an error status.
  virtual absl::StatusOr<std::string> Decode(
      absl::Span<const int> ids) const = 0;
  /// \brief Returns the vocabulary size.
  virtual int64_t vocab_size() const = 0;

  /// \brief Creates a tokenizer from a `tokenizer.json` or `tokenizer.model`
  /// file.
  ///
  /// \param path Path to the tokenizer file.
  /// \return The tokenizer, or an error status if the file cannot be loaded
  /// or tokenizer support is disabled at build time.
  static absl::StatusOr<std::shared_ptr<Tokenizer>> FromFile(
      const std::string& path);
};

}  // namespace inferx

#endif  // INFERX_TOKENIZER_H_
