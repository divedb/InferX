#ifndef INFERX_SERVER_API_ERROR_MAPPER_H_
#define INFERX_SERVER_API_ERROR_MAPPER_H_

#include <cstdint>
#include <exception>
#include <string>
#include <string_view>

#include "inferx/server/http_types.h"
#include "inferx/server/status.h"

namespace inferx::server {

enum class ErrorOrigin : std::uint8_t {
  kHttp,
  kAuth,
  kAdmission,
  kEngine,
  kInternal,
};

/// \brief Pure-function mapping of internal failures to stable, sanitized
///        client errors. Never exposes exception types, stack traces, file
///        paths, prompts, or engine internals.
class ApiErrorMapper {
 public:
  /// Sanitizes messages; `std::bad_alloc` and unknown exceptions map to a
  /// generic 500. Details belong in logs, not in the client response.
  [[nodiscard]] static ApiError FromException(std::exception_ptr error,
                                              ErrorOrigin origin);

  /// Maps a normalized engine request error. Fatal engine errors additionally
  /// trigger a readiness transition in the caller.
  [[nodiscard]] static ApiError FromEngine(const ApiError& engine_error,
                                           bool engine_fatal);

  /// OpenAI-compatible error envelope; used for unary responses and for the
  /// data frame after an SSE response has started.
  [[nodiscard]] static std::string ToJson(const ApiError& error);

  [[nodiscard]] static HttpResponse ToHttpResponse(const ApiError& error);

  // ---- Stable canned errors used by the HTTP layer ----
  [[nodiscard]] static ApiError NotFound(std::string_view path);
  [[nodiscard]] static ApiError MethodNotAllowed(std::string_view allow);
  [[nodiscard]] static ApiError MalformedRequest(std::string_view detail);
  [[nodiscard]] static ApiError HeadersTooLarge();
  [[nodiscard]] static ApiError PayloadTooLarge();
  [[nodiscard]] static ApiError RequestTimeout();
  [[nodiscard]] static ApiError ShuttingDown();
  [[nodiscard]] static ApiError Internal(std::string_view code,
                                         std::string_view detail);
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_API_ERROR_MAPPER_H_
