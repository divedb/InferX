#include "inferx/server/middleware.h"

#include <utility>

#include <boost/beast/http/fields.hpp>

namespace inferx::server {

namespace {

boost::asio::awaitable<MiddlewareOutcome> RunFrom(
    const std::vector<Middleware>& chain, std::size_t index,
    RequestContext& ctx, const Next& terminal) {
  if (index == chain.size()) {
    co_return co_await terminal();
  }
  const Next next = [&chain, index, &ctx, &terminal]() {
    return RunFrom(chain, index + 1, ctx, terminal);
  };
  co_return co_await chain[index](ctx, next);
}

}  // namespace

MiddlewareChain::MiddlewareChain(std::vector<Middleware> middleware)
    : middleware_(std::move(middleware)) {}

boost::asio::awaitable<MiddlewareOutcome> MiddlewareChain::Run(
    RequestContext& ctx, Next terminal) const {
  co_return co_await RunFrom(middleware_, 0, ctx, terminal);
}

Middleware MakeRequestIdMiddleware() {
  return [](RequestContext& ctx, const Next& next)
             -> boost::asio::awaitable<MiddlewareOutcome> {
    auto outcome = co_await next();
    if (std::holds_alternative<HttpResponse>(outcome)) {
      auto& response = std::get<HttpResponse>(outcome);
      response.headers.set("X-Request-Id", ctx.client_request_id().str());
    }
    co_return outcome;
  };
}

}  // namespace inferx::server
