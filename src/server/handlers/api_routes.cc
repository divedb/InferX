#include "inferx/server/handlers/api_routes.h"

#include "inferx/server/handlers/models_handler.h"
#include "inferx/server/handlers/completion_handler.h"
#include "inferx/server/handlers/tokenize_handler.h"

namespace inferx::server::handlers {
namespace {

Status AddPublicProbe(transport::Routes* routes, std::string path,
                      std::string name, const HealthState* health,
                      HealthProbe probe) {
  return routes->Add(
      boost::beast::http::verb::get, std::move(path),
      {.name = std::move(name),
       .max_body_bytes = 1,
       .authentication_required = false},
      std::make_shared<HealthHandler>(health, probe));
}

}  // namespace

StatusOr<std::shared_ptr<transport::Routes>> BuildApiRoutes(
    ApiRouteDependencies dependencies) {
  if (dependencies.health == nullptr) {
    return InvalidArgumentError("health state is required");
  }
  if (dependencies.models == nullptr) {
    return InvalidArgumentError("model registry is required");
  }
  if (dependencies.tokenization == nullptr) {
    return InvalidArgumentError("tokenization service is required");
  }
  if (dependencies.requests == nullptr) {
    return InvalidArgumentError("request service is required");
  }
  if (dependencies.guard == nullptr) {
    return InvalidArgumentError("route guard is required");
  }
  if (dependencies.max_inference_body_bytes == 0) {
    return InvalidArgumentError("inference body limit must be positive");
  }

  auto routes = std::make_shared<transport::Routes>(dependencies.guard);
  INFERX_RETURN_IF_ERROR(AddPublicProbe(routes.get(), "/health/live",
                                       "health.live", dependencies.health,
                                       HealthProbe::kLive));
  INFERX_RETURN_IF_ERROR(AddPublicProbe(routes.get(), "/health/ready",
                                       "health.ready", dependencies.health,
                                       HealthProbe::kReady));
  INFERX_RETURN_IF_ERROR(AddPublicProbe(routes.get(), "/health/startup",
                                       "health.startup", dependencies.health,
                                       HealthProbe::kStartup));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::get, "/v1/models",
      {.name = "models.list",
       .max_body_bytes = 1,
       .authentication_required = true,
       .required_scope = "models.read"},
      std::make_shared<ModelsHandler>(dependencies.models)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/v1/chat/completions",
      {.name = "chat.completions",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<ChatCompletionHandler>(
          dependencies.models, dependencies.tokenization,
          dependencies.requests)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/v1/completions",
      {.name = "completions",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<CompletionHandler>(
          dependencies.models, dependencies.tokenization,
          dependencies.requests)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/v1/tokenize",
      {.name = "tokenize",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<TokenizeHandler>(dependencies.models,
                                        dependencies.tokenization)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/v1/embeddings",
      {.name = "embeddings",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<EmbeddingsHandler>(dependencies.models,
                                          dependencies.embeddings)));
  return routes;
}

}  // namespace inferx::server::handlers
