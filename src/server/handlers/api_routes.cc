#include "inferx/server/handlers/api_routes.h"

#include "inferx/server/handlers/models_handler.h"
#include "inferx/server/handlers/metrics_handler.h"
#include "inferx/server/handlers/completion_handler.h"
#include "inferx/server/handlers/tokenize_handler.h"
#include "inferx/server/handlers/vllm_compat_handler.h"

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
  if (dependencies.metrics == nullptr) {
    return InvalidArgumentError("metrics source is required");
  }
  if (dependencies.guard == nullptr) {
    return InvalidArgumentError("route guard is required");
  }
  if (dependencies.max_inference_body_bytes == 0) {
    return InvalidArgumentError("inference body limit must be positive");
  }
  if (dependencies.request_timeout <= std::chrono::seconds::zero()) {
    return InvalidArgumentError("request timeout must be positive");
  }

  auto routes = std::make_shared<transport::Routes>(dependencies.guard);
  INFERX_RETURN_IF_ERROR(AddPublicProbe(routes.get(), "/health", "health",
                                       dependencies.health,
                                       HealthProbe::kLive));
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
      boost::beast::http::verb::get, "/metrics",
      {.name = "metrics",
       .max_body_bytes = 1,
       .authentication_required = false},
      std::make_shared<MetricsHandler>(dependencies.metrics,
                                       MetricsPresentation::kPrometheus)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::get, "/stats",
      {.name = "legacy.stats",
       .max_body_bytes = 1,
       .authentication_required = false},
      std::make_shared<MetricsHandler>(dependencies.metrics,
                                       MetricsPresentation::kLegacyJson)));
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
          dependencies.requests, dependencies.request_timeout)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/v1/completions",
      {.name = "completions",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<CompletionHandler>(
          dependencies.models, dependencies.tokenization,
          dependencies.requests, dependencies.request_timeout)));
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

  // vLLM compatibility surface.
  INFERX_RETURN_IF_ERROR(routes->AddPrefix(
      boost::beast::http::verb::get, "/v1/models/",
      {.name = "models.retrieve",
       .max_body_bytes = 1,
       .authentication_required = true,
       .required_scope = "models.read"},
      std::make_shared<ModelRetrieveHandler>(dependencies.models,
                                             "/v1/models/")));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::get, "/version",
      {.name = "version",
       .max_body_bytes = 1,
       .authentication_required = false},
      std::make_shared<VersionHandler>()));
  auto ping = std::make_shared<PingHandler>();
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::get, "/ping",
      {.name = "ping",
       .max_body_bytes = 1,
       .authentication_required = false},
      ping));
  // vLLM's POST /ping ignores any body; a small allowance keeps a client that
  // sends `{}` from a 413.
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/ping",
      {.name = "ping",
       .max_body_bytes = 1024,
       .authentication_required = false},
      ping));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/tokenize",
      {.name = "vllm.tokenize",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<VllmTokenizeHandler>(dependencies.models,
                                            dependencies.tokenization)));
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::post, "/detokenize",
      {.name = "vllm.detokenize",
       .max_body_bytes = dependencies.max_inference_body_bytes,
       .authentication_required = true,
       .required_scope = "inference.invoke"},
      std::make_shared<DetokenizeHandler>(dependencies.models,
                                          dependencies.tokenization)));
  // Pooling-model endpoints a vLLM client may probe: a 501 that names the
  // path tells it "never here" rather than the 404 that means "wrong URL".
  // Unauthenticated on purpose -- the answer is the same for everyone and
  // holds no tenant data. The inference body limit applies so a real scoring
  // payload reaches the 501 instead of dying as a 413.
  auto not_supported = std::make_shared<NotSupportedHandler>("pooling models");
  for (const char* path : {"/pooling", "/score", "/rerank", "/v1/pooling",
                           "/v1/score", "/v1/rerank"}) {
    INFERX_RETURN_IF_ERROR(routes->Add(
        boost::beast::http::verb::post, path,
        {.name = std::string("not_supported") + path,
         .max_body_bytes = dependencies.max_inference_body_bytes,
         .authentication_required = false},
        not_supported));
  }
  return routes;
}

}  // namespace inferx::server::handlers
