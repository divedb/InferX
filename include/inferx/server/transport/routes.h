#pragma once

#include <boost/beast/http/verb.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/server/transport/response_writer.h"

namespace inferx::server::transport {

struct RouteMetadata {
  std::string name;
  size_t max_body_bytes = 1u << 20;
  bool authentication_required = true;
  std::string required_scope;
};

class RouteGuard {
 public:
  virtual ~RouteGuard() = default;
  virtual folly::coro::Task<Status> Check(
      const HttpRequest& request, const RouteMetadata& metadata,
      RequestContext& context,
      folly::CancellationToken cancellation) = 0;
};

/// Exact-path router. Query strings are deliberately not stripped implicitly.
/// The one concession to parameterized paths is AddPrefix, whose routes match
/// only after every exact route has been tried.
class Routes final : public RequestHandler {
 public:
  explicit Routes(std::shared_ptr<RouteGuard> guard = nullptr);
  Status Add(boost::beast::http::verb method, std::string path,
             std::shared_ptr<RequestHandler> handler);
  Status Add(boost::beast::http::verb method, std::string path,
             RouteMetadata metadata,
             std::shared_ptr<RequestHandler> handler);

  /// Registers a trailing-parameter route: any target beginning with `prefix`
  /// dispatches to `handler`, which reads the remainder from the request
  /// target itself. The prefix must end with '/' so it can never shadow an
  /// exact route ("/v1/models" stays exact while "/v1/models/" captures ids).
  Status AddPrefix(boost::beast::http::verb method, std::string prefix,
                   RouteMetadata metadata,
                   std::shared_ptr<RequestHandler> handler);

  folly::coro::Task<void> Handle(
      HttpRequest request, RequestContext context, ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  struct Route {
    boost::beast::http::verb method;
    std::string path;
    RouteMetadata metadata;
    std::shared_ptr<RequestHandler> handler;
    bool prefix = false;
  };
  Status AddRoute(boost::beast::http::verb method, std::string path,
                  RouteMetadata metadata,
                  std::shared_ptr<RequestHandler> handler, bool prefix);
  std::shared_ptr<RouteGuard> guard_;
  std::vector<Route> routes_;
};

}  // namespace inferx::server::transport
