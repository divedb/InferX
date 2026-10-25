
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

/// \brief Specifies which side of a token sequence to truncate.
enum class TruncationSide {
  kLeft,   ///< Remove tokens from the beginning of the sequence.
  kRight,  ///< Remove tokens from the end of the sequence.
};

/// \brief Configuration options for tokenization.
///
/// Controls special-token insertion and sequence truncation.
/// Mirrors the relevant options of Hugging Face's encode() API.
struct EncodeOptions {
  /// Whether to automatically insert model-specific special tokens.
  bool add_special_tokens = true;

  /// Whether to truncate sequences exceeding the maximum length.
  bool truncation = false;

  /// Maximum number of tokens allowed when truncation is enabled.
  ///
  /// If not specified, the tokenizer's model_max_length is used
  /// when available.
  std::optional<size_t> max_length = std::nullopt;
};

/// \brief Configuration options for applying a chat template.
///
/// Controls how structured chat messages are rendered into a prompt.
/// Mirrors the relevant options of Hugging Face's apply_chat_template().
///
/// The rendered prompt is returned as text rather than token IDs.
/// It should generally be encoded with add_special_tokens=false,
/// since chat templates may already contain special tokens.
struct ChatTemplateOptions {
  /// Whether to append the assistant generation prefix.
  ///
  /// Typically enabled when the model should generate a new
  /// assistant response.
  bool add_generation_prompt = true;

  /// Whether to continue the final message instead of starting
  /// a new assistant message.
  ///
  /// \note Mutually exclusive with add_generation_prompt.
  bool continue_final_message = false;

  /// Optional chat template override for the current request.
  ///
  /// If not specified, the tokenizer's configured template is used.
  std::optional<std::string> chat_template;

  /// JSON array describing tools available to the model.
  ///
  /// An empty string indicates that no tools are provided.
  std::string tools_json;

  /// JSON object containing additional chat template arguments.
  ///
  /// An empty string indicates that no additional arguments
  /// are provided.
  std::string template_kwargs_json;
};

/// \brief Abstract tokenizer interface for model inference.
///
/// Defines the tokenizer operations required by the inference engine and OpenAI-compatible
/// serving frontend.
///
/// The interface is modeled on vLLM's TokenizerLike protocol and is independent of the
/// underlying tokenization library.
///
/// Implementations are responsible for tokenization, decoding, vocabulary lookup, and chat
/// template rendering.
///
/// Tokenizer properties should be cached during construction to avoid repeated backend queries.
///
/// \note Implementations must define their own thread-safety guarantees. Callers must not
///       assume concurrent access is safe.
class Tokenizer {
 public:
  virtual ~Tokenizer() = default;

  /// \brief Returns all registered special token strings.
  ///
  /// \return A reference to the cached special token strings.
  virtual const std::vector<std::string>& AllSpecialTokens() const noexcept = 0;

  /// \brief Returns all registered special token IDs.
  ///
  /// \return A reference to the cached special token IDs.
  virtual const std::vector<TokenId>& AllSpecialIds() const noexcept = 0;

  /// \brief Returns the beginning-of-sequence token ID.
  ///
  /// \return The BOS token ID, or std::nullopt if undefined.
  virtual std::optional<TokenId> BosTokenId() const noexcept = 0;

  /// \brief Returns the end-of-sequence token ID.
  ///
  /// \return The EOS token ID, or std::nullopt if undefined.
  virtual std::optional<TokenId> EosTokenId() const noexcept = 0;

  /// \brief Returns the padding token ID.
  ///
  /// \return The PAD token ID, or std::nullopt if undefined.
  virtual std::optional<TokenId> PadTokenId() const noexcept = 0;

  /// \brief Returns the base vocabulary size.
  ///
  /// Excludes tokens added after the base vocabulary was defined.
  /// Corresponds to Hugging Face's vocab_size property.
  ///
  /// \return The base vocabulary size.
  virtual size_t VocabSize() const noexcept = 0;

  /// \brief Returns the total tokenizer vocabulary size.
  ///
  /// Includes both base vocabulary and added tokens.
  /// Corresponds to Python's len(tokenizer).
  ///
  /// \return The total vocabulary size.
  virtual size_t Size() const noexcept = 0;

  /// \brief Returns the largest valid token ID.
  ///
  /// Corresponds to vLLM's max_token_id property.
  /// Useful for validating token IDs supplied by clients.
  ///
  /// \return The maximum valid token ID.
  ///
  /// \note Token IDs are not necessarily contiguous. A value within the valid numeric range is
  ///       not guaranteed to exist.
  virtual TokenId MaxTokenId() const noexcept = 0;

  /// \brief Returns the configured truncation direction.
  ///
  /// \return The side from which tokens are removed.
  virtual TruncationSide GetTruncationSide() const noexcept = 0;

  /// \brief Returns the configured maximum model sequence length.
  ///
  /// \return The maximum sequence length, or std::nullopt if unknown.
  virtual std::optional<int64_t> ModelMaxLength() const noexcept = 0;

  /// \brief Checks whether a token ID is a special token.
  ///
  /// Used to determine which tokens are removed when decoding
  /// with skip_special_tokens enabled.
  ///
  /// \param id Token ID to inspect.
  /// \return   True if the ID represents a registered special token.
  virtual bool IsSpecialTokenId(int32_t id) const noexcept = 0;

  /// \brief Returns the complete token-to-ID vocabulary.
  ///
  /// Includes base vocabulary entries and added tokens.
  /// The vocabulary may be constructed lazily on first access.
  ///
  /// \return A reference to the complete vocabulary mapping.
  virtual const std::unordered_map<std::string, TokenId>& GetVocab() const = 0;

  /// \brief Returns the added token vocabulary.
  ///
  /// Contains tokens registered in addition to the base vocabulary.
  ///
  /// \return A reference to the added token mapping.
  virtual const std::unordered_map<std::string, TokenId>& GetAddedVocab() const = 0;

  /// \brief Encodes text into a sequence of token IDs.
  ///
  /// Applies the configured tokenization rules, optional special
  /// token insertion, and truncation.
  ///
  /// \param text    Input text to encode.
  /// \param options Encoding configuration.
  /// \return        Encoded token IDs, or a tokenization error.
  virtual StatusOr<std::vector<TokenId>> Encode(std::string_view text,
                                                const EncodeOptions& options = {}) const = 0;

  /// \brief Encodes multiple texts into token ID sequences.
  ///
  /// Applies the same encoding options to every input.
  /// Output sequences preserve the order of the input texts.
  ///
  /// \param texts   Input texts to encode.
  /// \param options Encoding configuration.
  /// \return        Encoded token sequences, or a tokenization error.
  virtual StatusOr<std::vector<std::vector<TokenId>>> EncodeBatch(
      std::span<const std::string> texts, const EncodeOptions& options = {}) const = 0;

  /// \brief Decodes token IDs into text.
  ///
  /// Token IDs outside the supported vocabulary are ignored.
  /// Special tokens may optionally be omitted from the output.
  ///
  /// \param ids                 Token IDs to decode.
  /// \param skip_special_tokens Whether to omit special tokens.
  /// \return                    The decoded text.
  virtual std::string Decode(std::span<const TokenId> ids,
                             bool skip_special_tokens = false) const = 0;

  /// \brief Converts token IDs to their string representations.
  ///
  /// \param ids                 Token IDs to convert.
  /// \param skip_special_tokens Whether to omit special tokens.
  /// \return                    Token strings corresponding to the input IDs.
  virtual std::vector<std::string> ConvertIdsToTokens(
      std::span<const TokenId> ids, bool skip_special_tokens = false) const = 0;

  /// \brief Converts token strings to token IDs.
  ///
  /// \param tokens Token strings to convert.
  /// \return       Corresponding token IDs, or a conversion error.
  virtual StatusOr<std::vector<TokenId>> ConvertTokensToIds(
      std::span<const std::string> tokens) const = 0;

  /// \brief Converts token strings into readable text.
  ///
  /// Reconstructs text according to the tokenizer's decoding
  /// rules, including whitespace and subword handling.
  ///
  /// \param tokens Token strings to combine.
  /// \return       Reconstructed text, or a conversion error.
  virtual StatusOr<std::string> ConvertTokensToString(
      std::span<const std::string> tokens) const = 0;

  /// \brief Renders structured chat messages using a chat template.
  ///
  /// Converts a JSON array of chat messages into the prompt format
  /// expected by the model.
  ///
  /// Each message is a JSON object containing fields such as
  /// "role", "content", and optional model-specific attributes.
  ///
  /// Supports generation prompts, assistant message continuation,
  /// custom templates, tool definitions, and template arguments.
  ///
  /// The returned prompt is not tokenized. Call Encode() with
  /// add_special_tokens=false when the template already includes
  /// the required special tokens.
  ///
  /// \param messages_json JSON array of chat messages.
  /// \param options       Chat template configuration.
  /// \return              The rendered prompt, or a template processing error.
  virtual StatusOr<std::string> ApplyChatTemplate(
      std::string_view messages_json, const ChatTemplateOptions& options = {}) const = 0;
};

}  // namespace inferx
