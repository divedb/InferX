#include "inferx/server/handlers/admin_model_handler.h"

#include <algorithm>

#include "inferx/server/api/error_mapping.h"
#include "inferx/support/json.h"

namespace inferx::server::handlers {
namespace {

const char* StateName(model_registry::ModelState state) {
  using model_registry::ModelState;
  switch (state) {
    case ModelState::kDiscovered: return "discovered";
    case ModelState::kDownloading: return "downloading";
    case ModelState::kLoading: return "loading";
    case ModelState::kWarming: return "warming";
    case ModelState::kReady: return "ready";
    case ModelState::kDraining: return "draining";
    case ModelState::kUnloading: return "unloading";
    case ModelState::kUnloaded: return "unloaded";
    case ModelState::kFailed: return "failed";
  }
  return "unknown";
}

folly::coro::Task<void> Write(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    unsigned status, std::string body, folly::CancellationToken cancellation) {
  transport::HttpResponse response{
      static_cast<boost::beast::http::status>(status), request.version()};
  response.keep_alive(request.keep_alive());
  response.set(boost::beast::http::field::content_type, "application/json");
  response.body() = std::move(body);
  response.prepare_payload();
  (void)co_await writer.WriteResponse(std::move(response), cancellation);
}

}  // namespace

folly::coro::Task<void> AdminModelsHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  if (!context.authenticated || registry_ == nullptr) {
    auto error = api::MapStatus(
        !context.authenticated
            ? absl::UnauthenticatedError("authenticated context is required")
            : InternalError("model registry is not configured"));
    co_await Write(request, response, error.status, error.Json({}), cancellation);
    co_return;
  }
  auto models = registry_->Models(context.tenant_id);
  std::sort(models.begin(), models.end(), [](const auto& left, const auto& right) {
    if (left.id != right.id) return left.id < right.id;
    return left.version < right.version;
  });
  std::string body = "{\"models\":[";
  for (size_t index = 0; index < models.size(); ++index) {
    if (index != 0) body += ',';
    body += "{\"id\":";
    AppendJsonString(models[index].id, &body);
    body += ",\"version\":";
    AppendJsonString(models[index].version, &body);
    body += ",\"alias\":";
    if (models[index].alias.empty()) body += "null";
    else AppendJsonString(models[index].alias, &body);
    body += ",\"state\":";
    AppendJsonString(StateName(models[index].state), &body);
    body += ",\"supports_generation\":";
    body += models[index].supports_generation ? "true" : "false";
    body += ",\"supports_embeddings\":";
    body += models[index].supports_embeddings ? "true" : "false";
    body += "}";
  }
  body += "]}";
  co_await Write(request, response, 200, std::move(body), cancellation);
}

folly::coro::Task<void> UnimplementedModelOperationHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  (void)context;
  auto error = api::MapStatus(
      UnimplementedError(operation_ + " is not implemented by this backend"));
  co_await Write(request, response, error.status, error.Json({}), cancellation);
}

StatusOr<std::shared_ptr<transport::Routes>> BuildAdminRoutes(
    const model_registry::Registry* registry,
    std::shared_ptr<transport::RouteGuard> guard, size_t max_body_bytes) {
  if (registry == nullptr) return InvalidArgumentError("registry is required");
  if (guard == nullptr) return InvalidArgumentError("admin guard is required");
  if (max_body_bytes == 0) return InvalidArgumentError("body limit is required");
  auto routes = std::make_shared<transport::Routes>(std::move(guard));
  const transport::RouteMetadata read{
      .name = "admin.models.list",
      .max_body_bytes = 1,
      .authentication_required = true,
      .required_scope = "models.manage"};
  INFERX_RETURN_IF_ERROR(routes->Add(
      boost::beast::http::verb::get, "/admin/v1/models", read,
      std::make_shared<AdminModelsHandler>(registry)));
  for (const auto& [path, operation] :
       {std::pair{"/admin/v1/models/load", "model load"},
        std::pair{"/admin/v1/models/unload", "model unload"}}) {
    INFERX_RETURN_IF_ERROR(routes->Add(
        boost::beast::http::verb::post, path,
        {.name = operation,
         .max_body_bytes = max_body_bytes,
         .authentication_required = true,
         .required_scope = "models.manage"},
        std::make_shared<UnimplementedModelOperationHandler>(operation)));
  }
  return routes;
}

}  // namespace inferx::server::handlers
