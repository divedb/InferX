#ifndef INFERX_SERVER_MIDDLEWARE_H_
#define INFERX_SERVER_MIDDLEWARE_H_

#include <functional>
#include <memory>
#include <vector>
#include <variant>

#include <boost/asio/awaitable.hpp>

#include "inferx/server/request_context.h"
#include "inferx/server/router.h"

namespace inferx::server {

struct MiddlewareContinue {};
using MiddlewareOutcome =
    std::variant<MiddlewareContinue, HttpResponse, ApiError>;

using Next = std::function<boost::asio::awaitable<MiddlewareOutcome>()>;
using Middleware = std::function<boost::asio::awaitable<MiddlewareOutcome>(
    RequestContext&, const Next&)>;

/// \brief Ordered middleware pipeline. Each stage either continues, replaces
///        the response, or fails with an ApiError. Invoked on the session
///        strand.
class MiddlewareChain {
 public:
  explicit MiddlewareChain(std::vector<Middleware> middleware);

  [[nodiscard]] boost::asio::awaitable<MiddlewareOutcome> Run(
      RequestContext& ctx, Next terminal) const;

 private:
  std::vector<Middleware> middleware_;
};

/// \brief Request-ID middleware: stamps the resolved client request ID onto
///        every outgoing response as `X-Request-Id`. The ID itself is
///        resolved when the RequestContext is created (docs §4.3).
[[nodiscard]] Middleware MakeRequestIdMiddleware();

}  // namespace inferx::server

#endif  // INFERX_SERVER_MIDDLEWARE_H_
