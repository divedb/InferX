#include "inferx/server/routes.h"

#include <chrono>
#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/beast/http/field.hpp>
#include <nlohmann/json.hpp>

#include "inferx/server/http_types.h"
#include "inferx/server/json_codec.h"
#include "inferx/server/request_context.h"
#include "inferx/server/router.h"
#include "inferx/server/status.h"

namespace inferx::server {

namespace {

namespace http = boost::beast::http;

HttpResponse JsonResponse(nlohmann::json body, int status = 200) {
  HttpResponse response;
  response.status = status;
  response.headers.set(http::field::content_type, "application/json");
  auto dumped = DumpJson(std::move(body));
  const std::string text = dumped.ok() ? std::move(dumped.value()) : "{}";
  const auto* bytes = reinterpret_cast<const std::byte*>(text.data());
  response.body.assign(bytes, bytes + text.size());
  return response;
}

std::int64_t ProcessStartUnixSeconds() {
  static const std::int64_t value = [] {
    return static_cast<std::int64_t>(std::time(nullptr));
  }();
  return value;
}

/// \brief /health: vLLM-compatible engine-aware health. Empty 200 when the
///        snapshot is healthy, empty 503 otherwise.
RouteResult HandleHealth(RequestContext& ctx,
                         std::shared_ptr<HealthService> health) {
  (void)ctx;
  const HealthSnapshot snapshot = health->Snapshot();
  const bool healthy =
      snapshot.state == HealthState::kReady && snapshot.engine_ready;
  HttpResponse response;
  response.status = healthy ? 200 : 503;
  response.headers.set(http::field::content_type, "text/plain");
  return response;
}

/// \brief /livez: the event loop is responsive because this handler ran.
RouteResult HandleLivez(RequestContext& ctx) {
  (void)ctx;
  return JsonResponse(nlohmann::json{{"status", "ok"}});
}

/// \brief /readyz: serving, engine ready, and not draining; JSON reason on
///        503. Intentionally distinct from /health (design §8.2).
RouteResult HandleReadyz(RequestContext& ctx,
                         std::shared_ptr<HealthService> health) {
  (void)ctx;
  const HealthSnapshot snapshot = health->Snapshot();
  const bool ready =
      snapshot.state == HealthState::kReady && snapshot.engine_ready;
  nlohmann::json body;
  body["status"] = ready ? "ok" : "unavailable";
  if (!ready) {
    body["reason"] = snapshot.detail.empty()
                         ? std::string("server state is not ready")
                         : snapshot.detail;
  }
  return JsonResponse(std::move(body), ready ? 200 : 503);
}

/// \brief /v1/models: OpenAI ModelList from the (static Phase 1) catalog.
///        Task/capability extensions live under a versioned `metadata`
///        object, not as bare OpenAI fields (design §8.2).
boost::asio::awaitable<RouteResult> HandleModels(
    RequestContext& ctx, std::shared_ptr<IModelCatalog> models) {
  auto listed = co_await models->List(ctx.tenant());
  if (!listed) co_return listed.error();
  nlohmann::json data = nlohmann::json::array();
  for (const auto& model : listed.value()) {
    nlohmann::json item;
    item["id"] = model.id;
    item["object"] = "model";
    item["created"] = ProcessStartUnixSeconds();
    item["owned_by"] = model.owned_by;
    if (model.max_model_len.has_value()) {
      item["metadata"] = nlohmann::json{{"max_model_len", *model.max_model_len}};
    }
    data.push_back(std::move(item));
  }
  co_return JsonResponse(
      nlohmann::json{{"object", "list"}, {"data", std::move(data)}});
}

}  // namespace

std::vector<RouteSpec> Phase1RouteSpecs() {
  return {
      RouteSpec{"/health", HttpMethodMask::kGet, "health",
                /*requires_auth=*/false, /*allows_streaming=*/false},
      RouteSpec{"/livez", HttpMethodMask::kGet, "livez",
                /*requires_auth=*/false, /*allows_streaming=*/false},
      RouteSpec{"/readyz", HttpMethodMask::kGet, "readyz",
                /*requires_auth=*/false, /*allows_streaming=*/false},
      RouteSpec{"/v1/models", HttpMethodMask::kGet, "list_models",
                /*requires_auth=*/true, /*allows_streaming=*/false},
  };
}

std::shared_ptr<Router> BuildRouter(const ServerConfig& config,
                                    std::shared_ptr<IModelCatalog> models,
                                    std::shared_ptr<HealthService> health) {
  (void)config;
  auto router = std::make_shared<Router>(Phase1RouteSpecs());
  router->Add(RouteSpec{"/health", HttpMethodMask::kGet, "health", false, false},
              [health](RequestContext& ctx) -> boost::asio::awaitable<RouteResult> {
                co_return HandleHealth(ctx, health);
              });
  router->Add(RouteSpec{"/livez", HttpMethodMask::kGet, "livez", false, false},
              [](RequestContext& ctx) -> boost::asio::awaitable<RouteResult> {
                co_return HandleLivez(ctx);
              });
  router->Add(RouteSpec{"/readyz", HttpMethodMask::kGet, "readyz", false, false},
              [health](RequestContext& ctx) -> boost::asio::awaitable<RouteResult> {
                co_return HandleReadyz(ctx, health);
              });
  router->Add(RouteSpec{"/v1/models", HttpMethodMask::kGet, "list_models", true,
                        false},
              [models](RequestContext& ctx) -> boost::asio::awaitable<RouteResult> {
                co_return co_await HandleModels(ctx, models);
              });
  return router;
}

std::shared_ptr<MiddlewareChain> BuildMiddleware(const ServerConfig& config) {
  (void)config;  // auth/CORS/rate-limit middleware arrive in later phases
  return std::make_shared<MiddlewareChain>(
      std::vector<Middleware>{MakeRequestIdMiddleware()});
}

}  // namespace inferx::server
