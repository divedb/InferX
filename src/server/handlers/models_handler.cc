#include "inferx/server/handlers/models_handler.h"

#include <algorithm>
#include <string_view>

#include "inferx/server/api/error_mapping.h"
#include "inferx/support/json.h"

namespace inferx::server::handlers {
namespace {

transport::HttpResponse Response(const transport::HttpRequest& request,
                                 boost::beast::http::status status,
                                 std::string body) {
  transport::HttpResponse result{status, request.version()};
  result.keep_alive(request.keep_alive());
  result.set(boost::beast::http::field::content_type, "application/json");
  result.body() = std::move(body);
  result.prepare_payload();
  return result;
}

}  // namespace

folly::coro::Task<void> ModelsHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  if (!context.authenticated) {
    (void)co_await response.WriteResponse(
        Response(request, boost::beast::http::status::unauthorized,
                 "{\"error\":{\"message\":\"authenticated request context "
                 "is required\",\"type\":\"authentication_error\","
                 "\"code\":\"invalid_api_key\"}}"),
        cancellation);
    co_return;
  }
  if (registry_ == nullptr) {
    (void)co_await response.WriteResponse(
        Response(request, boost::beast::http::status::internal_server_error,
                 "{\"error\":{\"message\":\"model registry is not "
                 "configured\",\"type\":\"server_error\","
                 "\"code\":\"internal_error\"}}"),
        cancellation);
    co_return;
  }

  auto models = registry_->ReadyModels(context.tenant_id);
  std::sort(models.begin(), models.end(),
            [](const auto& left, const auto& right) {
              if (left.id != right.id) return left.id < right.id;
              return left.version < right.version;
            });
  std::string body = "{\"object\":\"list\",\"data\":[";
  for (size_t i = 0; i < models.size(); ++i) {
    if (i != 0) body += ',';
    body += "{\"id\":";
    AppendJsonString(models[i].alias.empty() ? models[i].id : models[i].alias,
                     &body);
    body += ",\"object\":\"model\",\"created\":" +
            std::to_string(models[i].created) +
            ",\"owned_by\":\"platform\",\"status\":\"ready\"}";
  }
  body += "]}";
  (void)co_await response.WriteResponse(
      Response(request, boost::beast::http::status::ok, std::move(body)),
      cancellation);
}

folly::coro::Task<void> ModelRetrieveHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  if (!context.authenticated) {
    (void)co_await response.WriteResponse(
        Response(request, boost::beast::http::status::unauthorized,
                 "{\"error\":{\"message\":\"authenticated request context "
                 "is required\",\"type\":\"authentication_error\","
                 "\"code\":\"invalid_api_key\"}}"),
        cancellation);
    co_return;
  }
  if (registry_ == nullptr) {
    (void)co_await response.WriteResponse(
        Response(request, boost::beast::http::status::internal_server_error,
                 "{\"error\":{\"message\":\"model registry is not "
                 "configured\",\"type\":\"server_error\","
                 "\"code\":\"internal_error\"}}"),
        cancellation);
    co_return;
  }

  std::string_view id{request.target().data(), request.target().size()};
  // The router guarantees the prefix; what remains is the id, taken verbatim
  // (query strings are not stripped anywhere in this server).
  id.remove_prefix(std::min(prefix_.size(), id.size()));
  auto model = registry_->Resolve(id, context.tenant_id);
  if (!model.ok()) {
    const auto error = api::MapStatus(model.status(), "model");
    (void)co_await response.WriteResponse(
        Response(request,
                 static_cast<boost::beast::http::status>(error.status),
                 error.Json({})),
        cancellation);
    co_return;
  }
  std::string body = "{\"id\":";
  AppendJsonString(id, &body);
  body += ",\"object\":\"model\",\"created\":" +
          std::to_string(model->created) + ",\"owned_by\":\"inferx\"}";
  (void)co_await response.WriteResponse(
      Response(request, boost::beast::http::status::ok, std::move(body)),
      cancellation);
}

}  // namespace inferx::server::handlers
