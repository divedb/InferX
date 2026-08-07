#include "inferx/server/transport/routes.h"

#include <boost/beast/http.hpp>

#include <utility>

namespace inferx::server::transport {
namespace {

namespace http = boost::beast::http;

folly::coro::Task<void> WriteRouteError(
    ResponseWriter& writer, unsigned version, bool keep_alive,
    boost::beast::http::status status, std::string message,
    folly::CancellationToken cancellation) {
  HttpResponse response{status, version};
  response.set(boost::beast::http::field::content_type, "application/json");
  response.keep_alive(keep_alive);
  response.body() = "{\"error\":{\"message\":\"" + std::move(message) +
                    "\",\"type\":\"invalid_request_error\"}}";
  response.prepare_payload();
  (void)co_await writer.WriteResponse(std::move(response), cancellation);
}

boost::beast::http::status GuardStatus(const Status& status) {
  switch (status.code()) {
    case absl::StatusCode::kUnauthenticated:
      return boost::beast::http::status::unauthorized;
    case absl::StatusCode::kPermissionDenied:
      return boost::beast::http::status::forbidden;
    case absl::StatusCode::kResourceExhausted:
      return boost::beast::http::status::too_many_requests;
    case absl::StatusCode::kUnavailable:
      return boost::beast::http::status::service_unavailable;
    default:
      return boost::beast::http::status::internal_server_error;
  }
}

}  // namespace

Routes::Routes(std::shared_ptr<RouteGuard> guard) : guard_(std::move(guard)) {}

Status Routes::Add(boost::beast::http::verb method, std::string path,
                   std::shared_ptr<RequestHandler> handler) {
  RouteMetadata metadata;
  metadata.name = path;
  metadata.authentication_required = false;
  return Add(method, std::move(path), std::move(metadata), std::move(handler));
}

Status Routes::Add(boost::beast::http::verb method, std::string path,
                   RouteMetadata metadata,
                   std::shared_ptr<RequestHandler> handler) {
  if (path.empty() || path.front() != '/') {
    return InvalidArgumentError("route path must start with '/'");
  }
  if (handler == nullptr) return InvalidArgumentError("route handler is null");
  for (const Route& route : routes_) {
    if (route.method == method && route.path == path) {
      return FailedPreconditionError("duplicate route: ", path);
    }
  }
  if (metadata.name.empty()) metadata.name = path;
  if (metadata.max_body_bytes == 0) {
    return InvalidArgumentError("route body limit must be positive");
  }
  routes_.push_back(Route{method, std::move(path), std::move(metadata),
                          std::move(handler)});
  return OkStatus();
}

folly::coro::Task<void> Routes::Handle(
    HttpRequest request, ResponseWriter& response,
    folly::CancellationToken cancellation) {
  bool path_exists = false;
  for (const Route& route : routes_) {
    if (request.target() != route.path) continue;
    path_exists = true;
    if (request.method() == route.method) {
      if (request.body().size() > route.metadata.max_body_bytes) {
        co_await WriteRouteError(response, request.version(),
                                 request.keep_alive(),
                                 http::status::payload_too_large,
                                 "request body too large", cancellation);
        co_return;
      }
      if (guard_ != nullptr) {
        const Status guarded =
            co_await guard_->Check(request, route.metadata, cancellation);
        if (!guarded.ok()) {
          co_await WriteRouteError(response, request.version(),
                                   request.keep_alive(), GuardStatus(guarded),
                                   std::string(guarded.message()), cancellation);
          co_return;
        }
      } else if (route.metadata.authentication_required) {
        co_await WriteRouteError(response, request.version(),
                                 request.keep_alive(),
                                 http::status::internal_server_error,
                                 "route guard is not configured", cancellation);
        co_return;
      }
      co_await route.handler->Handle(std::move(request), response, cancellation);
      co_return;
    }
  }
  co_await WriteRouteError(
      response, request.version(), request.keep_alive(),
      path_exists ? boost::beast::http::status::method_not_allowed
                  : boost::beast::http::status::not_found,
      path_exists ? "method not allowed" : "route not found", cancellation);
}

}  // namespace inferx::server::transport
