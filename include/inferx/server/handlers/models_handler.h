#pragma once

#include <string>
#include <utility>

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

/// `GET /v1/models/{id}`: the OpenAI retrieve-model endpoint. Registered as a
/// prefix route; the id is whatever follows `prefix` in the request target.
class ModelRetrieveHandler final : public transport::RequestHandler {
 public:
  ModelRetrieveHandler(const model_registry::Registry* registry,
                       std::string prefix)
      : registry_(registry), prefix_(std::move(prefix)) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
  std::string prefix_;
};

}  // namespace inferx::server::handlers
