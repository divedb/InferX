#pragma once

#include <string>
#include <string_view>

#include "inferx/core/status.h"

namespace inferx::server::api {

struct HttpError {
  int status = 500;
  std::string type = "server_error";
  std::string code = "internal_error";
  std::string param;
  std::string message;
  std::string retry_after;

  std::string Json(std::string_view request_id) const;
};

/// Converts internal failures into the stable external error contract.
HttpError MapStatus(const Status& status, std::string_view param = {});

}  // namespace inferx::server::api
