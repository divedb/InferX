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
      std::string_view prompt) = 0;
  virtual StatusOr<TokenizedPrompt> TokenizeChat(
      const request::ModelVersion& model_version,
      const std::vector<tokenizer::ChatMessage>& messages) = 0;
};

}  // namespace inferx::server::tokenization
