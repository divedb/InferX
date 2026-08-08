#pragma once

#include <string_view>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/request/request_context.h"
#include "inferx/tokenizer/chat_template.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx::server::tokenization {

struct TokenizedPrompt {
  request::ModelVersion model_version;
  std::vector<tokenizer::TokenId> token_ids;
  uint32_t prompt_tokens = 0;
};

class TokenizationService {
 public:
  virtual ~TokenizationService() = default;

  virtual StatusOr<TokenizedPrompt> TokenizeCompletion(
      const request::ModelVersion& model_version,
      std::string_view prompt, bool add_special_tokens = false) = 0;
  virtual StatusOr<TokenizedPrompt> TokenizeChat(
      const request::ModelVersion& model_version,
      const std::vector<tokenizer::ChatMessage>& messages) = 0;
};

/// Compatibility adapter for the currently loaded in-process model.
class InProcessTokenizationService final : public TokenizationService {
 public:
  /// \param chat_template Which transcribed template `TokenizeChat` renders.
  ///                      Chosen at composition from the model's architecture
  ///                      (or the gateway's configuration); the default keeps
  ///                      existing Qwen2 call sites unchanged.
  InProcessTokenizationService(
      const tokenizer::Tokenizer* tokenizer, request::ModelVersion model_version,
      tokenizer::ChatTemplateKind chat_template =
          tokenizer::ChatTemplateKind::kQwen2);

  StatusOr<TokenizedPrompt> TokenizeCompletion(
      const request::ModelVersion& model_version,
      std::string_view prompt, bool add_special_tokens = false) override;
  StatusOr<TokenizedPrompt> TokenizeChat(
      const request::ModelVersion& model_version,
      const std::vector<tokenizer::ChatMessage>& messages) override;

 private:
  const tokenizer::Tokenizer* tokenizer_;
  request::ModelVersion model_version_;
  tokenizer::ChatTemplateKind chat_template_;
};

}  // namespace inferx::server::tokenization
