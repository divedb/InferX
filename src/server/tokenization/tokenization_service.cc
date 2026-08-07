#include "inferx/server/tokenization/tokenization_service.h"

#include <utility>

namespace inferx::server::tokenization {

InProcessTokenizationService::InProcessTokenizationService(
    tokenizer::Tokenizer* tokenizer, request::ModelVersion model_version)
    : tokenizer_(tokenizer), model_version_(std::move(model_version)) {}

StatusOr<TokenizedPrompt> InProcessTokenizationService::TokenizeCompletion(
    const request::ModelVersion& model_version, std::string_view prompt) {
  if (tokenizer_ == nullptr) return FailedPreconditionError("tokenizer is null");
  if (model_version != model_version_) {
    return FailedPreconditionError("tokenization model version mismatch");
  }
  auto clone = tokenizer_->Clone();
  if (!clone.ok()) return clone.status();
  INFERX_ASSIGN_OR_RETURN(std::vector<tokenizer::TokenId> ids,
                          (*clone)->EncodeWithOptions(prompt, {}));
  if (ids.empty()) return InvalidArgumentError("prompt encodes to no tokens");
  const uint32_t count = static_cast<uint32_t>(ids.size());
  return TokenizedPrompt{model_version_, std::move(ids), count};
}

StatusOr<TokenizedPrompt> InProcessTokenizationService::TokenizeChat(
    const request::ModelVersion& model_version,
    const std::vector<tokenizer::ChatMessage>& messages) {
  if (tokenizer_ == nullptr) return FailedPreconditionError("tokenizer is null");
  if (model_version != model_version_) {
    return FailedPreconditionError("tokenization model version mismatch");
  }
  INFERX_ASSIGN_OR_RETURN(
      const std::string prompt,
      tokenizer::ApplyQwen2ChatTemplate(messages, /*add_generation_prompt=*/true));
  auto clone = tokenizer_->Clone();
  if (!clone.ok()) return clone.status();
  tokenizer::EncodeOptions options;
  options.special_tokens = tokenizer::SpecialTokenMode::kAsControl;
  INFERX_ASSIGN_OR_RETURN(std::vector<tokenizer::TokenId> ids,
                          (*clone)->EncodeWithOptions(prompt, options));
  if (ids.empty()) return InvalidArgumentError("chat encodes to no tokens");
  const uint32_t count = static_cast<uint32_t>(ids.size());
  return TokenizedPrompt{model_version_, std::move(ids), count};
}

}  // namespace inferx::server::tokenization
