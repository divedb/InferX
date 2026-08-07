#pragma once

#include "inferx/server/model_registry/registry.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::handlers {

class TokenizeHandler final : public transport::RequestHandler {
 public:
  TokenizeHandler(const model_registry::Registry* registry,
                  tokenization::TokenizationService* tokenization)
      : registry_(registry), tokenization_(tokenization) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
  tokenization::TokenizationService* tokenization_;
};

}  // namespace inferx::server::handlers
