#pragma once

#include <chrono>

#include "inferx/server/model_registry/registry.h"
#include "inferx/server/request/request_service.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::handlers {

class ChatCompletionHandler final : public transport::RequestHandler {
 public:
  ChatCompletionHandler(
      const model_registry::Registry* registry,
      tokenization::TokenizationService* tokenization,
      request::RequestService* requests,
      std::chrono::seconds timeout = std::chrono::seconds(300))
      : registry_(registry),
        tokenization_(tokenization),
        requests_(requests),
        timeout_(timeout) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
  tokenization::TokenizationService* tokenization_;
  request::RequestService* requests_;
  std::chrono::seconds timeout_;
};

}  // namespace inferx::server::handlers
