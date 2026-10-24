#include "inferx/server/api_error_mapper.h"

#include <string>
#include <utility>

#include <boost/beast/http/field.hpp>
#include <nlohmann/json.hpp>

#include "inferx/server/json_codec.h"

namespace inferx::server {

namespace {

namespace http = boost::beast::http;

std::string_view KindType(ApiErrorKind kind) noexcept {
  switch (kind) {
    case ApiErrorKind::kInvalidRequest:
      return "invalid_request_error";
    case ApiErrorKind::kAuthentication:
      return "authentication_error";
    case ApiErrorKind::kPermission:
      return "permission_error";
    case ApiErrorKind::kNotFound:
      return "not_found_error";
    case ApiErrorKind::kUnsupported:
      return "unsupported_error";
    case ApiErrorKind::kRateLimit:
      return "rate_limit_error";
    case ApiErrorKind::kOverloaded:
      return "overloaded_error";
    case ApiErrorKind::kTimeout:
      return "timeout_error";
    case ApiErrorKind::kEngine:
      return "engine_error";
    case ApiErrorKind::kInternal:
      return "internal_error";
  }
  return "internal_error";
}

}  // namespace

ApiError ApiErrorMapper::FromException(std::exception_ptr error,
                                       ErrorOrigin origin) {
  if (!error) {
    return Internal("internal_error", "unknown failure");
  }
  // The exception detail is logged by the session; the client message stays
  // sanitized regardless of what the exception carried.
  (void)origin;
  try {
    std::rethrow_exception(error);
  } catch (const std::bad_alloc&) {
    return ApiError{500, ApiErrorKind::kInternal, "internal_error",
                    "internal_error", "server resources exhausted",
                    std::nullopt, std::nullopt, false, {}};
  } catch (const std::exception&) {
    return Internal("internal_error", "an internal error occurred");
  } catch (...) {
    return Internal("internal_error", "an unrecognized internal error occurred");
  }
}

ApiError ApiErrorMapper::FromEngine(const ApiError& engine_error,
                                    bool engine_fatal) {
  ApiError mapped = engine_error;
  if (engine_fatal) {
    mapped.retryable = true;
    if (!mapped.retry_after.has_value()) {
      mapped.retry_after = std::chrono::seconds(1);
    }
  }
  return mapped;
}

std::string ApiErrorMapper::ToJson(const ApiError& error) {
  nlohmann::json envelope;
  nlohmann::json body;
  body["message"] = error.message;
  body["type"] = error.type.empty() ? std::string(KindType(error.kind))
                                    : error.type;
  body["param"] = error.param ? nlohmann::json(*error.param) : nlohmann::json();
  body["code"] = error.code.empty() ? std::string(KindType(error.kind))
                                    : error.code;
  envelope["error"] = std::move(body);
  auto dumped = DumpJson(envelope);
  return dumped.ok() ? std::move(dumped.value())
                     : std::string(R"({"error":{}})");
}

HttpResponse ApiErrorMapper::ToHttpResponse(const ApiError& error) {
  HttpResponse response;
  response.status = error.http_status;
  response.headers.set(http::field::content_type, "application/json");
  if (error.retry_after.has_value()) {
    response.headers.set(http::field::retry_after,
                         std::to_string(error.retry_after->count()));
  }
  for (const auto& [name, value] : error.headers) {
    response.headers.set(name, value);
  }
  const std::string json = ToJson(error);
  const auto* bytes = reinterpret_cast<const std::byte*>(json.data());
  response.body.assign(bytes, bytes + json.size());
  return response;
}

ApiError ApiErrorMapper::NotFound(std::string_view path) {
  return ApiError{404,
                  ApiErrorKind::kNotFound,
                  "invalid_request_error",
                  "not_found",
                  "The requested URL was not found: " + std::string(path),
                  std::nullopt,
                  std::nullopt,
                  false,
                  {}};
}

ApiError ApiErrorMapper::MethodNotAllowed(std::string_view allow) {
  return ApiError{405,
                  ApiErrorKind::kInvalidRequest,
                  "invalid_request_error",
                  "method_not_allowed",
                  "The HTTP method is not allowed for this URL",
                  std::nullopt,
                  std::nullopt,
                  false,
                  {{"Allow", std::string(allow)}}};
}

ApiError ApiErrorMapper::MalformedRequest(std::string_view detail) {
  return ApiError{400,
                  ApiErrorKind::kInvalidRequest,
                  "invalid_request_error",
                  "invalid_json",
                  detail.empty() ? "malformed HTTP request" : std::string(detail),
                  std::nullopt,
                  std::nullopt,
                  false,
                  {}};
}

ApiError ApiErrorMapper::HeadersTooLarge() {
  return ApiError{431,
                  ApiErrorKind::kInvalidRequest,
                  "invalid_request_error",
                  "headers_too_large",
                  "request headers exceed the configured limit",
                  std::nullopt,
                  std::nullopt,
                  false,
                  {}};
}

ApiError ApiErrorMapper::PayloadTooLarge() {
  return ApiError{413,
                  ApiErrorKind::kInvalidRequest,
                  "invalid_request_error",
                  "payload_too_large",
                  "request body exceeds the configured limit",
                  std::nullopt,
                  std::nullopt,
                  false,
                  {}};
}

ApiError ApiErrorMapper::RequestTimeout() {
  return ApiError{408,
                  ApiErrorKind::kTimeout,
                  "timeout_error",
                  "deadline_exceeded",
                  "request timed out",
                  std::nullopt,
                  std::nullopt,
                  false,
                  {}};
}

ApiError ApiErrorMapper::ShuttingDown() {
  return ApiError{503,
                  ApiErrorKind::kOverloaded,
                  "overloaded_error",
                  "server_shutting_down",
                  "the server is draining and not accepting new requests",
                  std::nullopt,
                  std::chrono::seconds(1),
                  true,
                  {}};
}

ApiError ApiErrorMapper::Internal(std::string_view code,
                                  std::string_view detail) {
  return ApiError{500,
                  ApiErrorKind::kInternal,
                  "internal_error",
                  std::string(code),
                  detail.empty() ? "an internal error occurred"
                                 : std::string(detail),
                  std::nullopt,
                  std::nullopt,
                  false,
                  {}};
}

}  // namespace inferx::server
