#pragma once

#include "inferx/server/model_registry/registry.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::handlers {

class ModelsHandler final : public transport::RequestHandler {
 public:
  explicit ModelsHandler(const model_registry::Registry* registry)
      : registry_(registry) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
};

}  // namespace inferx::server::handlers
