/// \brief HuggingFace fast-tokenizer backend (docs/tokenizer_design.md §8.1).
///
/// HFTokenizer implements the Tokenizer interface over the vendored
/// tokenizers-cpp Rust backend (`tokenizer.json` semantics) with metadata
/// merged from `tokenizer_config.json` / `special_tokens_map.json` and chat
/// templates rendered through the vendored minja subset, following the
/// behavior of vLLM's tokenizer group and Hugging Face transformers:
///
/// * Encode defaults to `add_special_tokens=true`, the HF Python default;
///   callers that render a chat template first encode with
///   `add_special_tokens=false` because the template already inserted the
///   model's control tokens.
/// * `Size()` is `len(tokenizer)` (base vocabulary plus added tokens);
///   `VocabSize()` is the base vocabulary only.
/// * ApplyChatTemplate renders over minja with the transformers-compatible
///   polyfills, supports tools and extra template kwargs, per-request
///   template overrides, and transformers' continue_final_message
///   truncation semantics.
///
/// Thread safety: a single instance must not be used concurrently. The
/// backend keeps mutable per-call state in the Rust handle. The prompt pool
/// gives each worker its own instance, loaded from the same directory; the
/// decode side owns a separate one.
#ifndef INFERX_TOKENIZER_HF_TOKENIZER_H_
#define INFERX_TOKENIZER_HF_TOKENIZER_H_

#include <memory>
#include <string>

#include "inferx/tokenizer/tokenizer.h"

namespace inferx {

/// \brief Raw HuggingFace tokenizer assets in memory.
struct HFTokenizerAssets {
  /// Contents of `tokenizer.json`; required. Must be a valid fast-tokenizer
  /// serialization (top-level object with a `model` object carrying `vocab`).
  std::string tokenizer_json;

  /// Contents of `tokenizer_config.json`; optional. Contributes special
  /// tokens, model_max_length, truncation side, and the chat template.
  std::string tokenizer_config_json;

  /// Contents of `special_tokens_map.json`; optional. Overrides the
  /// tokenizer_config special-token entries, as in transformers.
  std::string special_tokens_map_json;

  /// Directory the assets were loaded from; used in error messages only.
  std::string source_dir;
};

/// \brief Tokenizer over HuggingFace fast-tokenizer assets (tokenizers-cpp).
class HFTokenizer final : public Tokenizer {
 public:
  /// \brief Loads tokenizer assets from a checkpoint directory.
  ///
  /// `tokenizer.json` must exist; `tokenizer_config.json` and
  /// `special_tokens_map.json` are merged in when present.
  ///
  /// \param tokenizer_dir Directory holding the tokenizer assets.
  /// \return               The loaded tokenizer, or a NotFound /
  ///                       InvalidArgument status.
  static StatusOr<std::shared_ptr<HFTokenizer>> Load(const std::string& tokenizer_dir);

  /// \brief Builds a tokenizer over in-memory asset blobs.
  ///
  /// Same merge rules as Load(); used by tests and embedders that already
  /// hold the bytes.
  static StatusOr<std::shared_ptr<HFTokenizer>> FromAssets(const HFTokenizerAssets& assets);

  HFTokenizer(const HFTokenizer&) = delete;
  HFTokenizer& operator=(const HFTokenizer&) = delete;

  /// \name Tokenizer interface
  ///@{
  const std::vector<std::string>& AllSpecialTokens() const noexcept override;
  const std::vector<TokenId>& AllSpecialIds() const noexcept override;
  std::optional<TokenId> BosTokenId() const noexcept override;
  std::optional<TokenId> EosTokenId() const noexcept override;
  std::optional<TokenId> PadTokenId() const noexcept override;
  size_t VocabSize() const noexcept override;
  size_t Size() const noexcept override;
  TokenId MaxTokenId() const noexcept override;
  TruncationSide GetTruncationSide() const noexcept override;
  std::optional<int64_t> ModelMaxLength() const noexcept override;
  bool IsSpecialTokenId(int32_t id) const noexcept override;
  const std::unordered_map<std::string, TokenId>& GetVocab() const override;
  const std::unordered_map<std::string, TokenId>& GetAddedVocab() const override;
  // Default arguments are repeated here because C++ defaults do not
  // inherit through virtual overrides; static calls through HFTokenizer
  // need them too.
  StatusOr<std::vector<TokenId>> Encode(
      std::string_view text, const EncodeOptions& options = {}) const override;
  StatusOr<std::vector<std::vector<TokenId>>> EncodeBatch(
      std::span<const std::string> texts, const EncodeOptions& options = {}) const override;
  std::string Decode(std::span<const TokenId> ids,
                     bool skip_special_tokens = false) const override;
  std::vector<std::string> ConvertIdsToTokens(
      std::span<const TokenId> ids, bool skip_special_tokens = false) const override;
  StatusOr<std::vector<TokenId>> ConvertTokensToIds(
      std::span<const std::string> tokens) const override;
  StatusOr<std::string> ConvertTokensToString(
      std::span<const std::string> tokens) const override;
  StatusOr<std::string> ApplyChatTemplate(
      std::string_view messages_json, const ChatTemplateOptions& options = {}) const override;
  ///@}

 private:
  HFTokenizer() = default;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx

#endif  // INFERX_TOKENIZER_HF_TOKENIZER_H_
