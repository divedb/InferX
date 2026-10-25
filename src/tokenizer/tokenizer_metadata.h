/// \brief Metadata extraction from HuggingFace tokenizer asset files.
///
/// Mirrors the merge order of transformers' PreTrainedTokenizerFast init:
/// `tokenizer.json` defines the base vocabulary and its `added_tokens`
/// array; `tokenizer_config.json` contributes named special tokens,
/// model_max_length, truncation side, and the chat template;
/// `special_tokens_map.json` (when present) overrides the special-token
/// entries. Nothing here touches the Rust backend; outputs feed
/// HFTokenizer's cached properties.
#ifndef INFERX_TOKENIZER_TOKENIZER_METADATA_H_
#define INFERX_TOKENIZER_TOKENIZER_METADATA_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "inferx/core/status.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {

/// \brief Structural facts extracted from `tokenizer.json`.
struct TokenizerJsonInfo {
  /// Base vocabulary size: entries of `model.vocab` (HF's vocab_size).
  size_t base_vocab_size = 0;

  /// Largest id over the base vocabulary and `added_tokens`
  /// (vLLM's max_token_id).
  TokenId max_token_id = 0;

  /// `added_tokens` entries in file order: {content, id, special}.
  struct AddedToken {
    std::string content;
    TokenId id = 0;
    bool special = false;
  };
  std::vector<AddedToken> added_tokens;

  /// Base-vocabulary pairs in file order, for the lazy GetVocab() merge.
  std::vector<std::pair<std::string, TokenId>> base_vocab;
};

/// \brief Special-token and template settings from the config side.
struct TokenizerMetadata {
  /// Named special-token contents; empty string when unset.
  std::string bos_token;
  std::string eos_token;
  std::string pad_token;
  std::string unk_token;

  /// `additional_special_tokens` contents, file order, deduplicated.
  std::vector<std::string> additional_special_tokens;

  /// From tokenizer_config.json; nullopt when absent or the 1e30 sentinel.
  std::optional<int64_t> model_max_length;

  TruncationSide truncation_side = TruncationSide::kRight;

  /// The checkpoint's chat template source; empty when none is configured.
  std::string chat_template;
};

/// \brief Validates and extracts the structure of a `tokenizer.json`.
///
/// Rejects documents the Rust backend could panic on (missing model/vocab)
/// before they reach tokenizers-cpp.
StatusOr<TokenizerJsonInfo> ParseTokenizerJson(const nlohmann::json& tokenizer_json);

/// \brief Parses a config blob (tokenizer_config or special_tokens_map).
///
/// \param blob   File contents; empty means "absent file".
/// \param what   File name for error messages.
/// \return       Parsed JSON, or nullptr when absent; InvalidArgument on
///               malformed JSON.
StatusOr<std::unique_ptr<nlohmann::json>> ParseConfigBlob(std::string_view blob,
                                                          std::string_view what);

/// \brief Merges tokenizer_config and special_tokens_map into metadata.
///
/// special_tokens_map entries win over tokenizer_config ones, matching
/// transformers. Null inputs leave the corresponding fields unset.
TokenizerMetadata MergeMetadata(const nlohmann::json* tokenizer_config,
                                const nlohmann::json* special_tokens_map);

}  // namespace inferx

#endif  // INFERX_TOKENIZER_TOKENIZER_METADATA_H_
