#pragma once

#include <memory>

#include "inferx/server/handlers/embeddings_handler.h"
#include "inferx/server/handlers/chat_completion_handler.h"
#include "inferx/server/handlers/health_handler.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/request/request_service.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/routes.h"

namespace inferx::server::handlers {

struct ApiRouteDependencies {
  const HealthState* health = nullptr;
  const model_registry::Registry* models = nullptr;
  tokenization::TokenizationService* tokenization = nullptr;
  request::RequestService* requests = nullptr;
  std::shared_ptr<EmbeddingsService> embeddings;
  std::shared_ptr<transport::RouteGuard> guard;
  size_t max_inference_body_bytes = 1u << 20;
};

/// Builds the public API route table. Authentication policy and body limits
/// live here rather than being repeated by individual handlers.
StatusOr<std::shared_ptr<transport::Routes>> BuildApiRoutes(
    ApiRouteDependencies dependencies);

}  // namespace inferx::server::handlers
