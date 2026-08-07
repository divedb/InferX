#pragma once

#include "inferx/server/model_registry/registry.h"
#include "inferx/server/transport/routes.h"

namespace inferx::server::handlers {

class AdminModelsHandler final : public transport::RequestHandler {
 public:
  explicit AdminModelsHandler(const model_registry::Registry* registry)
      : registry_(registry) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
};

class UnimplementedModelOperationHandler final
    : public transport::RequestHandler {
 public:
  explicit UnimplementedModelOperationHandler(std::string operation)
      : operation_(std::move(operation)) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  std::string operation_;
};

StatusOr<std::shared_ptr<transport::Routes>> BuildAdminRoutes(
    const model_registry::Registry* registry,
    std::shared_ptr<transport::RouteGuard> guard,
    size_t max_body_bytes = 1u << 20);

}  // namespace inferx::server::handlers
