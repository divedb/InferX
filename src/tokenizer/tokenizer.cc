// tokenizers-cpp backed Tokenizer (docs/tokenizer_design.md §8.1). The
// backend's Encode/Decode take no options: add_special_tokens and
// skip_special_tokens therefore surface through the option overloads as
// Unimplemented rather than being silently ignored. EncodeBatch has a native
// backend and is overridden.
#include "inferx/tokenizer/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef INFERX_HAVE_TOKENIZERS
#include "tokenizers_cpp.h"
#endif

namespace inferx {
namespace {

#ifdef INFERX_HAVE_TOKENIZERS

/// \brief Reads an entire file into a byte string.
absl::StatusOr<std::string> ReadBlob(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return NotFoundError("could not open tokenizer file: ", path);
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

/// \brief Tokenizer backed by HuggingFace `tokenizers` via tokenizers-cpp.
class TokenizersCppTokenizer : public Tokenizer {
 public:
  explicit TokenizersCppTokenizer(std::unique_ptr<tokenizers::Tokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {}

  StatusOr<std::vector<int>> Encode(absl::string_view text) const override {
    try {
      std::vector<int32_t> ids = tokenizer_->Encode(std::string(text));
      return std::vector<int>(ids.begin(), ids.end());
    } catch (const std::exception& e) {
      return InvalidArgumentError("tokenizer encode failed: ", e.what());
    }
  }

  StatusOr<std::string> Decode(absl::Span<const int> ids) const override {
    try {
      std::vector<int32_t> input(ids.begin(), ids.end());
      return tokenizer_->Decode(input);
    } catch (const std::exception& e) {
      return InvalidArgumentError("tokenizer decode failed: ", e.what());
    }
  }

  int64_t vocab_size() const override {
    return static_cast<int64_t>(tokenizer_->GetVocabSize());
  }

  StatusOr<std::vector<std::vector<int>>> EncodeBatch(
      absl::Span<const std::string> texts,
      const EncodeOptions& options) const override {
    if (options.add_special_tokens || options.return_token_type_ids) {
      return UnimplementedError("this tokenizer does not expose encode options");
    }
    try {
      std::vector<std::string> input(texts.begin(), texts.end());
      std::vector<std::vector<int32_t>> batches = tokenizer_->EncodeBatch(input);
      std::vector<std::vector<int>> out;
      out.reserve(batches.size());
      for (auto& batch : batches) {
        out.emplace_back(batch.begin(), batch.end());
      }
      return out;
    } catch (const std::exception& e) {
      return InvalidArgumentError("tokenizer batch encode failed: ", e.what());
    }
  }

 private:
  // unique_ptr keeps the FFI object unshared: one owner per worker thread.
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

#endif  // INFERX_HAVE_TOKENIZERS

}  // namespace

StatusOr<std::shared_ptr<Tokenizer>> Tokenizer::FromFile(const std::string& path) {
#ifdef INFERX_HAVE_TOKENIZERS
  StatusOr<std::string> blob = ReadBlob(path);
  if (!blob.ok()) {
    return blob.status();
  }
  if (path.size() >= 6 && path.compare(path.size() - 6, 6, ".model") == 0) {
    return UnimplementedError("SentencePiece tokenizers are not enabled in this build");
  }
  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(*blob);
  if (tokenizer == nullptr) {
    return InvalidArgumentError("failed to load tokenizer: ", path);
  }
  return std::make_shared<TokenizersCppTokenizer>(std::move(tokenizer));
#else
  return UnimplementedError(
      "InferX was built without tokenizer support (set INFERX_ENABLE_TOKENIZERS=ON)");
#endif
}

}  // namespace inferx
