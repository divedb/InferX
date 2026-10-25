#include "rust_tokenizer.h"

#include <utility>

#ifdef INFERX_HAVE_TOKENIZERS
#include <tokenizers_c.h>

namespace inferx::tokenizers {
namespace {

/// Frees the malloc-ed id arrays the C API returns for encode results.
class EncodeResults {
 public:
  explicit EncodeResults(size_t count) : results_(count) {}

  ~EncodeResults() { tokenizers_free_encode_results(results_.data(), results_.size()); }

  TokenizerEncodeResult* data() { return results_.data(); }
  TokenizerEncodeResult& operator[](size_t i) { return results_[i]; }

 private:
  std::vector<TokenizerEncodeResult> results_;
};

}  // namespace

StatusOr<RustTokenizer> RustTokenizer::Create(std::string_view tokenizer_json) {
  RustTokenizer tokenizer;
  tokenizer.handle_ = tokenizers_new_from_str(tokenizer_json.data(), tokenizer_json.size());
  if (tokenizer.handle_ == nullptr) {
    return InvalidArgumentError("tokenizers-cpp rejected the tokenizer.json blob");
  }
  return tokenizer;
}

RustTokenizer::RustTokenizer(RustTokenizer&& other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

RustTokenizer& RustTokenizer::operator=(RustTokenizer&& other) noexcept {
  if (this != &other) {
    if (handle_ != nullptr) tokenizers_free(handle_);
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

RustTokenizer::~RustTokenizer() {
  if (handle_ != nullptr) tokenizers_free(handle_);
}

std::vector<TokenId> RustTokenizer::Encode(std::string_view text,
                                           bool add_special_tokens) const {
  TokenizerEncodeResult result;
  tokenizers_encode(handle_, text.data(), text.size(), add_special_tokens ? 1 : 0, &result);
  std::vector<TokenId> ids(result.token_ids, result.token_ids + result.len);
  tokenizers_free_encode_results(&result, 1);
  return ids;
}

std::vector<std::vector<TokenId>> RustTokenizer::EncodeBatch(
    const std::vector<std::string>& texts, bool add_special_tokens) const {
  std::vector<const char*> raw;
  std::vector<size_t> lengths;
  raw.reserve(texts.size());
  lengths.reserve(texts.size());
  for (const std::string& text : texts) {
    raw.push_back(text.data());
    lengths.push_back(text.size());
  }
  EncodeResults results(texts.size());
  tokenizers_encode_batch(handle_, raw.data(), lengths.data(), texts.size(),
                          add_special_tokens ? 1 : 0, results.data());

  std::vector<std::vector<TokenId>> batch;
  batch.reserve(texts.size());
  for (size_t i = 0; i < texts.size(); ++i) {
    batch.emplace_back(results[i].token_ids, results[i].token_ids + results[i].len);
  }
  return batch;
}

std::string RustTokenizer::Decode(std::span<const TokenId> ids, bool skip_special_tokens) const {
  // The C API reinterprets ids as u32; negatives would become huge values the
  // backend treats as unknown, but filtering keeps that contract explicit.
  std::vector<uint32_t> clean;
  clean.reserve(ids.size());
  for (TokenId id : ids) {
    if (id >= 0) clean.push_back(static_cast<uint32_t>(id));
  }
  tokenizers_decode(handle_, clean.data(), clean.size(), skip_special_tokens ? 1 : 0);
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_get_decode_str(handle_, &data, &len);
  return std::string(data, len);
}

std::string RustTokenizer::IdToToken(TokenId id) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_id_to_token(handle_, static_cast<uint32_t>(id), &data, &len);
  return std::string(data, len);
}

TokenId RustTokenizer::TokenToId(std::string_view token) const {
  int32_t id = -1;
  tokenizers_token_to_id(handle_, token.data(), token.size(), &id);
  return id;
}

size_t RustTokenizer::VocabSize() const {
  size_t size = 0;
  tokenizers_get_vocab_size(handle_, &size);
  return size;
}

}  // namespace inferx::tokenizers

#else  // !INFERX_HAVE_TOKENIZERS

namespace inferx::tokenizers {

// The backend was compiled out. Construction fails, so the remaining
// methods are unreachable link-time stubs that never touch the C API.

StatusOr<RustTokenizer> RustTokenizer::Create(std::string_view) {
  return UnimplementedError("tokenizers-cpp is not built into this binary");
}

RustTokenizer::RustTokenizer(RustTokenizer&&) noexcept = default;
RustTokenizer& RustTokenizer::operator=(RustTokenizer&&) noexcept = default;
RustTokenizer::~RustTokenizer() = default;

std::vector<TokenId> RustTokenizer::Encode(std::string_view, bool) const { return {}; }
std::vector<std::vector<TokenId>> RustTokenizer::EncodeBatch(const std::vector<std::string>&,
                                                             bool) const {
  return {};
}
std::string RustTokenizer::Decode(std::span<const TokenId>, bool) const { return {}; }
std::string RustTokenizer::IdToToken(TokenId) const { return {}; }
TokenId RustTokenizer::TokenToId(std::string_view) const { return -1; }
size_t RustTokenizer::VocabSize() const { return 0; }

}  // namespace inferx::tokenizers

#endif  // INFERX_HAVE_TOKENIZERS
