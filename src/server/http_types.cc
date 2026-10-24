#include "inferx/server/http_types.h"

namespace inferx::server {

bool IsSupportedMethod(HttpMethod method) noexcept {
  switch (method) {
    case HttpMethod::get:
    case HttpMethod::post:
    case HttpMethod::put:
    case HttpMethod::patch:
    case HttpMethod::delete_:
    case HttpMethod::options:
    case HttpMethod::head:
      return true;
    default:
      return false;
  }
}

}  // namespace inferx::server
