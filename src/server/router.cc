#include "inferx/server/router.h"

#include <string>
#include <utility>
#include <vector>

#include <boost/beast/http/verb.hpp>

#include "inferx/server/api_error_mapper.h"

namespace inferx::server {

HttpMethodMask MethodMaskOf(HttpMethod method) {
  switch (method) {
    case HttpMethod::get:
      return HttpMethodMask::kGet;
    case HttpMethod::post:
      return HttpMethodMask::kPost;
    case HttpMethod::put:
      return HttpMethodMask::kPut;
    case HttpMethod::patch:
      return HttpMethodMask::kPatch;
    case HttpMethod::delete_:
      return HttpMethodMask::kDelete;
    case HttpMethod::options:
      return HttpMethodMask::kOptions;
    case HttpMethod::head:
      return HttpMethodMask::kHead;
    default:
      return HttpMethodMask::kNone;
  }
}

struct Router::Route {
  RouteSpec spec;
  std::vector<std::string> segments;  // pre-split pattern; ":name" is a param
  RouteHandler handler;
};

namespace {

std::vector<std::string> SplitPath(std::string_view path) {
  std::vector<std::string> segments;
  std::size_t start = 0;
  if (!path.empty() && path.front() == '/') start = 1;
  while (start <= path.size()) {
    const std::size_t slash = path.find('/', start);
    if (slash == std::string_view::npos) {
      segments.emplace_back(path.substr(start));
      break;
    }
    segments.emplace_back(path.substr(start, slash - start));
    start = slash + 1;
  }
  return segments;
}

/// \brief Matches request segments against pattern segments; fills `params`.
bool MatchSegments(const std::vector<std::string>& pattern,
                   const std::vector<std::string>& actual,
                   RouteParams* params) {
  if (pattern.size() != actual.size()) return false;
  RouteParams matched;
  for (std::size_t i = 0; i < pattern.size(); ++i) {
    const std::string& piece = pattern[i];
    if (!piece.empty() && piece.front() == ':') {
      matched.values.emplace_back(piece.substr(1), actual[i]);
    } else if (piece != actual[i]) {
      return false;
    }
  }
  if (params) *params = std::move(matched);
  return true;
}

}  // namespace

Router::Router(std::vector<RouteSpec> specs) {
  routes_.reserve(specs.size());
  for (auto& spec : specs) {
    RouteHandler handler;  // handlers are attached via Add
    routes_.push_back(Route{std::move(spec), {}, std::move(handler)});
    routes_.back().segments = SplitPath(routes_.back().spec.pattern);
  }
}

Router::~Router() = default;

void Router::Add(RouteSpec spec, RouteHandler handler) {
  Route route{std::move(spec), {}, std::move(handler)};
  route.segments = SplitPath(route.spec.pattern);
  // Re-adding a pattern upgrades its spec in place (the constructor
  // pre-registers specs so route discovery works before handlers attach).
  for (auto& existing : routes_) {
    if (existing.spec.pattern == route.spec.pattern) {
      existing = std::move(route);
      return;
    }
  }
  routes_.push_back(std::move(route));
}

boost::asio::awaitable<RouteResult> Router::Dispatch(
    RequestContext& ctx) const {
  const std::vector<std::string> segments = SplitPath(ctx.request().path);
  const HttpMethod method = ctx.request().method;
  // HEAD is served as GET-without-body by the session writer.
  const HttpMethod effective =
      method == HttpMethod::head ? HttpMethod::get : method;
  const Router::Route* method_match = nullptr;
  const Router::Route* path_match = nullptr;
  RouteParams matched_params;
  for (const auto& route : routes_) {
    RouteParams candidate;
    if (!MatchSegments(route.segments, segments, &candidate)) continue;
    path_match = &route;
    if (MaskContains(route.spec.methods, effective)) {
      method_match = &route;
      matched_params = std::move(candidate);
      break;
    }
  }
  if (method_match == nullptr) {
    if (path_match != nullptr) {
      co_return ApiErrorMapper::MethodNotAllowed(
          AllowedMethods(ctx.request().path));
    }
    co_return ApiErrorMapper::NotFound(ctx.request().path);
  }
  ctx.set_route_params(std::move(matched_params));
  co_return co_await method_match->handler(ctx);
}

bool Router::HasRoute(HttpMethod method, std::string_view path) const {
  const std::vector<std::string> segments = SplitPath(path);
  for (const auto& route : routes_) {
    if (MaskContains(route.spec.methods, method) &&
        MatchSegments(route.segments, segments, nullptr)) {
      return true;
    }
  }
  return false;
}

std::string Router::AllowedMethods(std::string_view path) const {
  const std::vector<std::string> segments = SplitPath(path);
  std::vector<std::string_view> allowed;
  const auto all = {HttpMethod::get,    HttpMethod::post,
                    HttpMethod::put,    HttpMethod::patch,
                    HttpMethod::delete_, HttpMethod::options,
                    HttpMethod::head};
  for (const HttpMethod method : all) {
    for (const auto& route : routes_) {
      if (MaskContains(route.spec.methods, method) &&
          MatchSegments(route.segments, segments, nullptr)) {
        allowed.push_back(to_string(method));
        break;
      }
    }
  }
  std::string joined;
  for (std::size_t i = 0; i < allowed.size(); ++i) {
    if (i != 0) joined.append(", ");
    joined.append(allowed[i]);
  }
  return joined;
}

}  // namespace inferx::server
