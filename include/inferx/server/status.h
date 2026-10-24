#ifndef INFERX_SERVER_STATUS_H_
#define INFERX_SERVER_STATUS_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace inferx::server {

/// \brief Stable, machine-readable error kinds. Part of the public API
///        contract; the wire `type` strings are derived per endpoint family.
enum class ApiErrorKind : std::uint8_t {
  kInvalidRequest,
  kAuthentication,
  kPermission,
  kNotFound,
  kUnsupported,
  kRateLimit,
  kOverloaded,
  kTimeout,
  kEngine,
  kInternal,
};

/// \brief The server boundary error type. Carries everything needed to render
///        an OpenAI-compatible error envelope plus transport headers.
///
/// `code` is stable across releases; `message` is client-safe but unstable.
struct ApiError {
  int http_status = 500;
  ApiErrorKind kind = ApiErrorKind::kInternal;
  std::string type;  // e.g. "invalid_request_error"
  std::string code;  // e.g. "model_not_found"; stable across releases
  std::string message;  // sanitized, client-safe
  std::optional<std::string> param;
  std::optional<std::chrono::seconds> retry_after;
  bool retryable = false;
  /// Extra response headers (e.g. "Allow" on 405); rendered by
  /// ApiErrorMapper::ToHttpResponse.
  std::vector<std::pair<std::string, std::string>> headers;

  bool operator==(const ApiError& other) const {
    return http_status == other.http_status && kind == other.kind &&
           type == other.type && code == other.code && message == other.message;
  }
};

/// \brief Value-or-ApiError result for the HTTP boundary. Successful values
///        must be movable; errors are explicit, never exceptions.
template <class T>
class [[nodiscard]] Result {
 public:
  // Present when T is default-constructible: completion-handler signatures
  // (e.g. asio::co_spawn) require a default-constructible value type.
  Result() = default;
  Result(T value) : value_(std::move(value)) {}  // NOLINT
  Result(ApiError error) : value_(std::move(error)) {}  // NOLINT

  [[nodiscard]] bool ok() const noexcept {
    return std::holds_alternative<T>(value_);
  }
  explicit operator bool() const noexcept { return ok(); }

  [[nodiscard]] T& value() & { return std::get<T>(value_); }
  [[nodiscard]] const T& value() const& { return std::get<T>(value_); }
  [[nodiscard]] T&& value() && { return std::get<T>(std::move(value_)); }
  [[nodiscard]] const ApiError& error() const& {
    return std::get<ApiError>(value_);
  }
  [[nodiscard]] ApiError&& error() && { return std::get<ApiError>(std::move(value_)); }

 private:
  std::variant<T, ApiError> value_;
};

/// \brief Void specialization: success carries no value.
template <>
class [[nodiscard]] Result<void> {
 public:
  Result() = default;  // NOLINT
  Result(ApiError error) : error_(std::move(error)) {}  // NOLINT

  [[nodiscard]] bool ok() const noexcept { return !error_.has_value(); }
  explicit operator bool() const noexcept { return ok(); }
  [[nodiscard]] const ApiError& error() const& { return *error_; }
  [[nodiscard]] ApiError&& error() && { return std::move(*error_); }

 private:
  std::optional<ApiError> error_;
};

/// \brief Type-distinct string ID; cross-kind assignment is a compile error.
template <class Tag>
class StrongId {
 public:
  StrongId() = default;
  explicit StrongId(std::string value) : value_(std::move(value)) {}

  [[nodiscard]] const std::string& str() const noexcept { return value_; }
  [[nodiscard]] bool empty() const noexcept { return value_.empty(); }
  friend bool operator==(const StrongId&, const StrongId&) = default;

 private:
  std::string value_;
};

/// \brief std::hash support for the StrongId types.
template <class Tag>
struct StrongIdHash {
  std::size_t operator()(const StrongId<Tag>& id) const noexcept {
    return std::hash<std::string>{}(id.str());
  }
};

struct ClientRequestTag;
struct EngineRequestTag;
struct ConnectionTag;
struct TenantTag;

using ClientRequestId = StrongId<ClientRequestTag>;
using EngineRequestId = StrongId<EngineRequestTag>;
using ConnectionId = StrongId<ConnectionTag>;
using TenantId = StrongId<TenantTag>;

using Clock = std::chrono::steady_clock;
using Deadline = std::optional<Clock::time_point>;

}  // namespace inferx::server

#endif  // INFERX_SERVER_STATUS_H_
