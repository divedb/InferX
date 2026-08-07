#include "inferx/server/api/error_mapping.h"

#include "inferx/support/json.h"

namespace inferx::server::api {

std::string HttpError::Json(std::string_view request_id) const {
  std::string out = "{\"error\":{\"message\":";
  AppendJsonString(message, &out);
  out += ",\"type\":";
  AppendJsonString(type, &out);
  out += ",\"param\":";
  if (param.empty()) out += "null";
  else AppendJsonString(param, &out);
  out += ",\"code\":";
  AppendJsonString(code, &out);
  out += ",\"request_id\":";
  AppendJsonString(request_id, &out);
  out += "}}";
  return out;
}

HttpError MapStatus(const Status& status, std::string_view param) {
  HttpError error;
  error.message = std::string(status.message());
  error.param = std::string(param);
  switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
      error.status = 400;
      error.type = "invalid_request_error";
      error.code = "invalid_request";
      break;
    case absl::StatusCode::kUnauthenticated:
      error.status = 401;
      error.type = "authentication_error";
      error.code = "invalid_api_key";
      break;
    case absl::StatusCode::kPermissionDenied:
      error.status = 403;
      error.type = "permission_error";
      error.code = "permission_denied";
      break;
    case absl::StatusCode::kNotFound:
      error.status = 404;
      error.type = "invalid_request_error";
      error.code = "model_not_found";
      break;
    case absl::StatusCode::kAlreadyExists:
    case absl::StatusCode::kAborted:
    case absl::StatusCode::kFailedPrecondition:
      error.status = 409;
      error.type = "conflict_error";
      error.code = "conflict";
      break;
    case absl::StatusCode::kResourceExhausted:
      error.status = 429;
      error.type = "rate_limit_error";
      error.code = "capacity_exceeded";
      error.retry_after = "1";
      break;
    case absl::StatusCode::kDeadlineExceeded:
      error.status = 504;
      error.type = "timeout_error";
      error.code = "inference_timeout";
      break;
    case absl::StatusCode::kUnimplemented:
      error.status = 422;
      error.type = "invalid_request_error";
      error.code = "unsupported_parameter";
      break;
    case absl::StatusCode::kUnavailable:
      error.status = 503;
      error.type = "server_error";
      error.code = "service_unavailable";
      error.retry_after = "1";
      break;
    default:
      error.status = 500;
      error.type = "server_error";
      error.code = "internal_error";
      break;
  }
  return error;
}

}  // namespace inferx::server::api
