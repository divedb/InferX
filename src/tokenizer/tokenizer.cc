#include "inferx/tokenizer/tokenizer.h"

#include <cstdint>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "absl/status/status.h"

#ifdef INFERX_HAVE_TOKENIZERS
#include "tokenizers_cpp.h"
#endif

namespace inferx {
namespace {

#ifdef INFERX_HAVE_TOKENIZERS

/// \brief Reads an entire file into a byte string.
///
/// \param path Path to the tokenizer file.
/// \return The file contents, or an error status.
absl::StatusOr<std::string> ReadBlob(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return absl::NotFoundError("could not open tokenizer file: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(input),
                     std::istreambuf_iterator<char>());
}

/// \brief Tokenizer backed by HuggingFace `tokenizers` via tokenizers-cpp.
class TokenizersCppTokenizer : public Tokenizer {
 public:
  explicit TokenizersCppTokenizer(
      std::unique_ptr<tokenizers::Tokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {}

  absl::StatusOr<std::vector<int>> Encode(
      absl::string_view text) const override {
    std::vector<int32_t> ids = tokenizer_->Encode(std::string(text));
    return std::vector<int>(ids.begin(), ids.end());
  }

  absl::StatusOr<std::string> Decode(absl::Span<const int> ids) const override {
    std::vector<int32_t> input(ids.begin(), ids.end());
    return tokenizer_->Decode(input);
  }

  int64_t vocab_size() const override {
    return static_cast<int64_t>(tokenizer_->GetVocabSize());
  }

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

#endif  // INFERX_HAVE_TOKENIZERS

}  // namespace

absl::StatusOr<std::shared_ptr<Tokenizer>> Tokenizer::FromFile(
    const std::string& path) {
#ifdef INFERX_HAVE_TOKENIZERS
  absl::StatusOr<std::string> blob = ReadBlob(path);
  if (!blob.ok()) {
    return blob.status();
  }
  if (path.size() >= 6 && path.compare(path.size() - 6, 6, ".model") == 0) {
    return absl::UnimplementedError(
        "SentencePiece tokenizers are not enabled in this build");
  }
  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(*blob);
  if (tokenizer == nullptr) {
    return absl::InvalidArgumentError("failed to load tokenizer: " + path);
  }
  return std::make_shared<TokenizersCppTokenizer>(std::move(tokenizer));
#else
  return absl::UnimplementedError(
      "InferX was built without tokenizer support (set "
      "INFERX_ENABLE_TOKENIZERS=ON)");
#endif
}

}  // namespace inferx
