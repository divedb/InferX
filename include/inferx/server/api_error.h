#ifndef INFERX_SERVER_API_ERROR_H_
#define INFERX_SERVER_API_ERROR_H_

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "inferx/core/status.h"
#include "nlohmann/json.hpp"

namespace inferx {

struct ApiError {
  int http_status = 500;
  std::string type = "server_error";    ///< OpenAI error type.
  std::string code;                     ///< Stable machine-readable code.
  std::string message;                  ///< Sanitized, client-safe.
  std::string param;                    ///< Offending field, when applicable.
  std::chrono::seconds retry_after{0};  ///< >0 => emit Retry-After.
  bool retryable = false;

  /// OpenAI error envelope: {"error":{message,type,param,code}}.
  nlohmann::json ToJson() const;
};

/// \brief Stable error factories (§8.1 table). Codes are API contract.
inline ApiError InvalidJson(const std::string& detail) {
  return {400,  "invalid_request_error", "invalid_json", "invalid JSON body: " + detail, "", {},
          false};
}

inline ApiError InvalidRequest(const std::string& message, std::string param = {},
                               std::string code = {}) {
  return {400, "invalid_request_error", std::move(code), message, std::move(param), {}, false};
}

inline ApiError UnsupportedParameter(const std::string& message, std::string param) {
  return {400,  "invalid_request_error", "unsupported_parameter", message, std::move(param), {},
          false};
}

inline ApiError ModelNotFound(const std::string& model) {
  return {404,
          "invalid_request_error",
          "model_not_found",
          "model '" + model + "' not found",
          "model",
          {},
          false};
}
inline ApiError Unauthorized(const std::string& message) {
  return {401, "authentication_error", "invalid_api_key", message, "", {}, false};
}
inline ApiError PayloadTooLarge() {
  return {413,
          "request_too_large",
          "payload_too_large",
          "request body exceeds the configured limit",
          "",
          {},
          false};
}
inline ApiError UnsupportedMediaType() {
  return {415,
          "invalid_request_error",
          "unsupported_media_type",
          "content type must be application/json",
          "",
          {},
          false};
}
inline ApiError ServerOverloaded(std::chrono::seconds retry_after) {
  return {503,
          "server_error",
          "server_overloaded",
          "the server is overloaded; retry later",
          "",
          retry_after,
          true};
}
inline ApiError EngineUnavailable(const std::string& message) {
  return {503, "server_error", "engine_unavailable", message, {}, {}, true};
}
inline ApiError NotImplemented(const std::string& message,
                               std::string code = "not_implemented") {
  return {501, "not_implemented_error", std::move(code), message, "", {}, false};
}
inline ApiError ShuttingDown() {
  return {503,
          "server_error",
          "shutting_down",
          "the server is draining and not accepting new requests",
          "",
          std::chrono::seconds(1),
          true};
}
inline ApiError Internal(const std::string& message) {
  return {500, "server_error", "internal_error", message, "", {}, false};
}

/// \brief Maps an engine/prompt-pool Status onto the stable error model.
/// Overload (queue full) is retryable 503; invalid input is 400; shutting
/// down is 503; everything else is a sanitized 500/503.
ApiError MapStatusToApiError(const Status& status);

/// \brief The error frame body used after an SSE stream has started (the
/// HTTP status is already committed; the status field is informational).
nlohmann::json MakeStreamErrorFrame(const ApiError& error);

}  // namespace inferx

#endif  // INFERX_SERVER_API_ERROR_H_
