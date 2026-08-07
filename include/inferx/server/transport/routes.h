#pragma once

#include <boost/beast/http/verb.hpp>

#include <memory>
#include <string>
#include <vector>

#include "inferx/server/transport/response_writer.h"

namespace inferx::server::transport {

/// Exact-path router. Query strings are deliberately not stripped implicitly.
class Routes final : public RequestHandler {
 public:
  Status Add(boost::beast::http::verb method, std::string path,
             std::shared_ptr<RequestHandler> handler);

  folly::coro::Task<void> Handle(
      HttpRequest request, ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  struct Route {
    boost::beast::http::verb method;
    std::string path;
    std::shared_ptr<RequestHandler> handler;
  };
  std::vector<Route> routes_;
};

}  // namespace inferx::server::transport
