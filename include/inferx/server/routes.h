#ifndef INFERX_SERVER_ROUTES_H_
#define INFERX_SERVER_ROUTES_H_

#include <memory>
#include <vector>

#include "inferx/server/middleware.h"
#include "inferx/server/router.h"
#include "inferx/server/server_config.h"
#include "inferx/server/services.h"

namespace inferx::server {

/// \brief Phase 1 route table: /health, /livez, /readyz, /v1/models.
///        Later phases add /v1/chat/completions, /v1/completions,
///        /v1/embeddings; route availability is catalog-driven.
[[nodiscard]] std::vector<RouteSpec> Phase1RouteSpecs();

[[nodiscard]] std::shared_ptr<Router> BuildRouter(
    const ServerConfig& config,
    std::shared_ptr<IModelCatalog> models,
    std::shared_ptr<HealthService> health);

[[nodiscard]] std::shared_ptr<MiddlewareChain> BuildMiddleware(
    const ServerConfig& config);

}  // namespace inferx::server

#endif  // INFERX_SERVER_ROUTES_H_
