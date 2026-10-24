#ifndef INFERX_SERVER_ROUTER_H_
#define INFERX_SERVER_ROUTER_H_

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <boost/asio/awaitable.hpp>
#include <boost/beast/http/verb.hpp>

#include "inferx/server/http_types.h"
#include "inferx/server/request_context.h"
#include "inferx/server/status.h"

namespace inferx::server {

struct StreamCompleted {};

using RouteResult = std::variant<HttpResponse, StreamCompleted, ApiError>;
using RouteHandler =
    std::function<boost::asio::awaitable<RouteResult>(RequestContext&)>;

/// Router-side method bitmask over the supported verbs.
enum class HttpMethodMask : std::uint16_t {
  kNone = 0,
  kGet = 1u << 0,
  kPost = 1u << 1,
  kPut = 1u << 2,
  kPatch = 1u << 3,
  kDelete = 1u << 4,
  kOptions = 1u << 5,
  kHead = 1u << 6,
};

HttpMethodMask MethodMaskOf(HttpMethod method);
[[nodiscard]] constexpr HttpMethodMask operator|(HttpMethodMask a,
                                                 HttpMethodMask b) {
  return static_cast<HttpMethodMask>(static_cast<std::uint16_t>(a) |
                                     static_cast<std::uint16_t>(b));
}
[[nodiscard]] constexpr bool MaskContains(HttpMethodMask mask,
                                          HttpMethod method) {
  return (static_cast<std::uint16_t>(mask) &
          static_cast<std::uint16_t>(MethodMaskOf(method))) != 0;
}

struct RouteSpec {
  std::string pattern;  // "/v1/chat/completions"; ":" segments are params
  HttpMethodMask methods = HttpMethodMask::kPost;
  std::string operation_id;  // OpenAPI / metrics label; low cardinality
  bool requires_auth = true;
  bool allows_streaming = false;
};

/// \brief Immutable-after-start route table. Dispatch only; middleware is run
///        by the session before `Dispatch`.
class Router {
 public:
  explicit Router(std::vector<RouteSpec> specs);
  ~Router();

  void Add(RouteSpec spec, RouteHandler handler);

  /// Matches `ctx.request().path` + method. On a path match with a
  /// non-matching method, returns a 405 ApiError carrying `Allow`.
  /// On no match, returns a 404 ApiError. Route params (when present) are
  /// installed on the context before the handler runs.
  [[nodiscard]] boost::asio::awaitable<RouteResult> Dispatch(
      RequestContext& ctx) const;

  [[nodiscard]] bool HasRoute(HttpMethod method, std::string_view path) const;

  /// Wire method names allowed for a path ("GET, POST"); empty when the
  /// path matches no route.
  [[nodiscard]] std::string AllowedMethods(std::string_view path) const;

 private:
  struct Route;
  std::vector<Route> routes_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_ROUTER_H_
