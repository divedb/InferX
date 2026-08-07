#include "inferx/server/transport/routes.h"

#include <boost/beast/http.hpp>

#include <utility>

namespace inferx::server::transport {
namespace {

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

}  // namespace

Status Routes::Add(boost::beast::http::verb method, std::string path,
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
  routes_.push_back(Route{method, std::move(path), std::move(handler)});
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
