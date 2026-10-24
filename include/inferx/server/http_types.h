#ifndef INFERX_SERVER_HTTP_TYPES_H_
#define INFERX_SERVER_HTTP_TYPES_H_

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <boost/beast/http/fields.hpp>
#include <boost/beast/http/verb.hpp>

#include "inferx/server/status.h"

namespace inferx::server {

/// Methods are Beast's `http::verb` throughout; the parser produces it
/// directly. `IsSupportedMethod` defines the serving allowlist on top.
using HttpMethod = boost::beast::http::verb;
/// RFC 7230 field container: case-insensitive lookup, ordered iteration.
using HeaderMap = boost::beast::http::fields;
using Body = std::vector<std::byte>;

/// \brief The methods this server dispatches; everything else is a 400.
bool IsSupportedMethod(HttpMethod method) noexcept;

/// \brief The parsed, normalized request handed to middleware and handlers.
struct HttpRequest {
  HttpMethod method = HttpMethod::get;
  std::string target;  // request-target, including query
  std::string path;    // decoded path, without query
  std::string query;
  HeaderMap headers;
  Body body;  // move-only use; body is never copied
  std::string peer_ip;
  std::uint16_t peer_port = 0;
  bool keep_alive = true;
};

/// \brief The response produced by a handler or the error mapper.
struct HttpResponse {
  int status = 200;
  HeaderMap headers;
  Body body;
};

struct RouteParams {
  std::vector<std::pair<std::string, std::string>> values;
  [[nodiscard]] std::string_view get(std::string_view key) const noexcept {
    for (const auto& [name, value] : values) {
      if (name == key) return value;
    }
    return {};
  }
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_HTTP_TYPES_H_
