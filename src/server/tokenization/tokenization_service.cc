#include "inferx/server/tokenization/tokenization_service.h"

#include <utility>

namespace inferx::server::tokenization {

StatusOr<std::vector<std::string>> TokenizationService::DecodeTokens(
    const request::ModelVersion&, const std::vector<tokenizer::TokenId>& ids) {
  std::vector<std::string> out;
  out.reserve(ids.size());
  for (const tokenizer::TokenId id : ids) {
    out.push_back("token_id:" + std::to_string(id));
  }
  return out;
}

StatusOr<std::string> TokenizationService::Detokenize(
    const request::ModelVersion&, const std::vector<tokenizer::TokenId>&) {
  return UnimplementedError(
      "sequence detokenization is not available in this deployment");
}

InProcessTokenizationService::InProcessTokenizationService(
    const tokenizer::Tokenizer* tokenizer, request::ModelVersion model_version,
    tokenizer::ChatTemplateKind chat_template)
    : tokenizer_(tokenizer),
      model_version_(std::move(model_version)),
      chat_template_(chat_template) {}

StatusOr<TokenizedPrompt> InProcessTokenizationService::TokenizeCompletion(
    const request::ModelVersion& model_version, std::string_view prompt,
    bool add_special_tokens) {
  if (tokenizer_ == nullptr) return FailedPreconditionError("tokenizer is null");
  if (model_version != model_version_) {
    return FailedPreconditionError("tokenization model version mismatch");
  }
  auto clone = tokenizer_->Clone();
  if (!clone.ok()) return clone.status();
  tokenizer::EncodeOptions options;
  options.special_tokens = tokenizer::SpecialTokenMode::kAsText;
  options.add_post_processor_tokens = add_special_tokens;
  INFERX_ASSIGN_OR_RETURN(std::vector<tokenizer::TokenId> ids,
                          (*clone)->EncodeWithOptions(prompt, options));
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
      tokenizer::ApplyChatTemplate(chat_template_, messages,
                                   /*add_generation_prompt=*/true));
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

StatusOr<std::vector<std::string>> InProcessTokenizationService::DecodeTokens(
    const request::ModelVersion& model_version,
    const std::vector<tokenizer::TokenId>& ids) {
  if (tokenizer_ == nullptr) return FailedPreconditionError("tokenizer is null");
  if (model_version != model_version_) {
    return FailedPreconditionError("tokenization model version mismatch");
  }
  // One clone for the whole batch: Clone() re-parses the tokenizer artifact,
  // so cloning per id would turn a 20-alternative logprob request into 20
  // artifact parses per generated token.
  INFERX_ASSIGN_OR_RETURN(const std::unique_ptr<tokenizer::Tokenizer> clone,
                          tokenizer_->Clone());
  std::vector<std::string> out;
  out.reserve(ids.size());
  for (const tokenizer::TokenId id : ids) {
    // Special tokens stay visible: a logprob report that names EOS as the
    // chosen token must not render it as an empty string.
    out.push_back(clone->Decode({id}, /*skip_special=*/false));
  }
  return out;
}

StatusOr<std::string> InProcessTokenizationService::Detokenize(
    const request::ModelVersion& model_version,
    const std::vector<tokenizer::TokenId>& ids) {
  if (tokenizer_ == nullptr) return FailedPreconditionError("tokenizer is null");
  if (model_version != model_version_) {
    return FailedPreconditionError("tokenization model version mismatch");
  }
  INFERX_ASSIGN_OR_RETURN(const std::unique_ptr<tokenizer::Tokenizer> clone,
                          tokenizer_->Clone());
  // Special tokens stay visible for the same reason as in DecodeTokens: a
  // client detokenizing its own prompt ids must get back what it encoded.
  return clone->Decode(ids, /*skip_special=*/false);
}

}  // namespace inferx::server::tokenization
