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
  return AddRoute(method, std::move(path), std::move(metadata),
                  std::move(handler), /*prefix=*/false);
}

Status Routes::AddPrefix(boost::beast::http::verb method, std::string prefix,
                         RouteMetadata metadata,
                         std::shared_ptr<RequestHandler> handler) {
  // A prefix that does not end in '/' could shadow an exact sibling
  // ("/v1/models" the listing vs "/v1/modelsanything"); the slash makes the
  // parameter boundary part of the route.
  if (prefix.size() < 2 || prefix.back() != '/') {
    return InvalidArgumentError("prefix route must end with '/'");
  }
  return AddRoute(method, std::move(prefix), std::move(metadata),
                  std::move(handler), /*prefix=*/true);
}

Status Routes::AddRoute(boost::beast::http::verb method, std::string path,
                        RouteMetadata metadata,
                        std::shared_ptr<RequestHandler> handler, bool prefix) {
  if (path.empty() || path.front() != '/') {
    return InvalidArgumentError("route path must start with '/'");
  }
  if (handler == nullptr) return InvalidArgumentError("route handler is null");
  for (const Route& route : routes_) {
    if (route.method == method && route.path == path &&
        route.prefix == prefix) {
      return FailedPreconditionError("duplicate route: ", path);
    }
  }
  if (metadata.name.empty()) metadata.name = path;
  if (metadata.max_body_bytes == 0) {
    return InvalidArgumentError("route body limit must be positive");
  }
  routes_.push_back(Route{method, std::move(path), std::move(metadata),
                          std::move(handler), prefix});
  return OkStatus();
}

folly::coro::Task<void> Routes::Handle(
    HttpRequest request, RequestContext context, ResponseWriter& response,
    folly::CancellationToken cancellation) {
  // Exact routes take precedence over prefix routes unconditionally, so
  // registration order can never change which handler owns a fixed path.
  const Route* match = nullptr;
  bool path_exists = false;
  const std::string_view target{request.target().data(),
                                request.target().size()};
  for (const bool prefix_pass : {false, true}) {
    for (const Route& route : routes_) {
      if (route.prefix != prefix_pass) continue;
      const bool covers = route.prefix ? target.starts_with(route.path)
                                       : target == route.path;
      if (!covers) continue;
      path_exists = true;
      if (request.method() == route.method) {
        match = &route;
        break;
      }
    }
    if (match != nullptr) break;
  }
  if (match == nullptr) {
    co_await WriteRouteError(
        response, request.version(), request.keep_alive(),
        path_exists ? boost::beast::http::status::method_not_allowed
                    : boost::beast::http::status::not_found,
        path_exists ? "method not allowed" : "route not found", cancellation);
    co_return;
  }
  if (request.body().size() > match->metadata.max_body_bytes) {
    co_await WriteRouteError(response, request.version(), request.keep_alive(),
                             http::status::payload_too_large,
                             "request body too large", cancellation);
    co_return;
  }
  if (guard_ != nullptr) {
    const Status guarded =
        co_await guard_->Check(request, match->metadata, context, cancellation);
    if (!guarded.ok()) {
      co_await WriteRouteError(response, request.version(),
                               request.keep_alive(), GuardStatus(guarded),
                               std::string(guarded.message()), cancellation);
      co_return;
    }
  } else if (match->metadata.authentication_required) {
    co_await WriteRouteError(response, request.version(), request.keep_alive(),
                             http::status::internal_server_error,
                             "route guard is not configured", cancellation);
    co_return;
  }
  co_await match->handler->Handle(std::move(request), std::move(context),
                                  response, cancellation);
}

}  // namespace inferx::server::transport
