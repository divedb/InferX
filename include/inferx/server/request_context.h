#ifndef INFERX_SERVER_REQUEST_CONTEXT_H_
#define INFERX_SERVER_REQUEST_CONTEXT_H_

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

#include <boost/asio/any_io_executor.hpp>

#include "inferx/server/engine_client.h"
#include "inferx/server/engine_types.h"
#include "inferx/server/http_types.h"
#include "inferx/server/status.h"

namespace inferx::server {

class IStreamWriter;

enum class RequestPhase : std::uint8_t {
  kCreated,
  kValidating,
  kQueued,
  kSubmitting,
  kAccepted,
  kStreaming,
  kCompleted,
  kCancelled,
  kFailed,
};

/// \brief Resolves the client-visible request ID: a validated `X-Request-Id`
///        header when one is supplied (charset [A-Za-z0-9._~-], length
///        8..128), otherwise a freshly generated UUIDv7. Spoofed or
///        malformed IDs are replaced, not echoed.
[[nodiscard]] ClientRequestId ResolveClientRequestId(const HeaderMap& headers);

/// \brief Generates a time-ordered UUIDv7 string (lowercase hex).
[[nodiscard]] std::string GenerateRequestId();

/// \brief Per-HTTP-request lifetime object: identity, cancellation, deadline,
///        route params, engine handle, usage counters, terminal state.
///
/// Created and destroyed on the session strand. `RequestCancel` and
/// `EngineRequestHandle::Cancel` are the thread-safe entry points.
class RequestContext {
 public:
  RequestContext(ClientRequestId client_id, ConnectionId connection_id,
                 TenantId tenant, HttpRequest request,
                 boost::asio::any_io_executor executor,
                 Clock::time_point received_at, Clock::time_point deadline,
                 IStreamWriter* writer);

  RequestContext(const RequestContext&) = delete;
  RequestContext& operator=(const RequestContext&) = delete;
  RequestContext(RequestContext&&) = delete;
  RequestContext& operator=(RequestContext&&) = delete;
  ~RequestContext();

  [[nodiscard]] ClientRequestId client_request_id() const noexcept;
  [[nodiscard]] ConnectionId connection_id() const noexcept;
  [[nodiscard]] TenantId tenant() const noexcept;
  [[nodiscard]] const HttpRequest& request() const noexcept;
  [[nodiscard]] HttpRequest& request() noexcept;

  [[nodiscard]] boost::asio::any_io_executor executor() const noexcept;
  [[nodiscard]] Clock::time_point deadline() const noexcept;
  [[nodiscard]] bool expired() const noexcept;

  /// Shared cancellation state for engine submit/stream and server timers.
  /// The asio::cancellation_signal used for socket operations is private to
  /// the session and is not exposed to handlers.
  [[nodiscard]] CancelToken cancel_token() const noexcept;
  void RequestCancel(CancelReason reason) noexcept;  // thread-safe
  [[nodiscard]] CancelReason cancel_reason() const noexcept;
  [[nodiscard]] bool cancelled() const noexcept;

  [[nodiscard]] IStreamWriter& stream() noexcept;  // valid for the request
  [[nodiscard]] bool response_committed() const noexcept;
  void set_response_committed() noexcept;

  void set_route_params(RouteParams params);
  [[nodiscard]] const RouteParams& route_params() const noexcept;

  void set_engine_handle(EngineRequestHandle handle);
  [[nodiscard]] EngineRequestHandle* engine_handle() noexcept;
  [[nodiscard]] std::optional<EngineRequestId> engine_request_id() const noexcept;

  void set_usage_prompt_tokens(std::uint64_t value) noexcept;
  void set_usage_completion_tokens(std::uint64_t value) noexcept;
  [[nodiscard]] std::uint64_t prompt_tokens() const noexcept;
  [[nodiscard]] std::uint64_t completion_tokens() const noexcept;

  /// Called exactly once on the session strand when the request reaches a
  /// terminal state. Idempotent; returns false if another terminal already
  /// won.
  bool finalize(RequestPhase terminal) noexcept;
  [[nodiscard]] RequestPhase phase() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_REQUEST_CONTEXT_H_
