#ifndef INFERX_SERVER_ENGINE_CLIENT_H_
#define INFERX_SERVER_ENGINE_CLIENT_H_

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio/awaitable.hpp>

#include "inferx/server/engine_types.h"
#include "inferx/server/status.h"

namespace inferx::server {

struct EngineCallContext {
  std::chrono::steady_clock::time_point deadline;
  CancelToken cancel;
  TraceContext trace;
  bool allow_streaming = true;
};

/// \brief Pull-based asynchronous engine event stream.
///
/// `Next()` returns `Accepted` exactly once, then zero or more
/// `Delta`/`Usage` events, then exactly one `Finished` or `EngineError`.
/// A pending `Next()` must wake and return an error when the handle is
/// cancelled or the stream is closed. Completions may arrive on an engine
/// thread; implementations must honor the caller's associated executor.
class IEngineEventStream {
 public:
  virtual ~IEngineEventStream() = default;

  virtual boost::asio::awaitable<Result<EngineGenerateEvent>> Next() = 0;

  /// Closes the stream and releases transport buffers. Idempotent and
  /// thread-safe: it must wake a pending `Next()`.
  virtual void Close() noexcept = 0;

  [[nodiscard]] virtual std::size_t buffered_events() const noexcept = 0;
};

/// \brief Same discipline as IEngineEventStream for embedding requests.
class IEngineEmbedStream {
 public:
  virtual ~IEngineEmbedStream() = default;

  virtual boost::asio::awaitable<Result<EngineEmbedEvent>> Next() = 0;
  virtual void Close() noexcept = 0;
  [[nodiscard]] virtual std::size_t buffered_events() const noexcept = 0;
};

struct EngineHealth {
  bool running = false;
  bool ready = false;
  bool draining = false;
  std::string detail;
};

struct EngineModelInfo {
  std::string id;
  std::string owned_by;
  std::vector<ModelTask> tasks;
  std::optional<std::uint64_t> max_model_len;
  std::vector<std::string> aliases;
  bool supports_streaming = true;
  bool supports_tools = false;
  bool supports_vision = false;
  bool supports_audio = false;
  bool supports_video = false;
};

/// \brief Transport-neutral engine boundary. The server never schedules GPU
///        work; it submits normalized requests and pulls events.
class IEngineClient {
 public:
  virtual ~IEngineClient() = default;

  /// Completes with an engine acceptance, or a transport acceptance when
  /// `disposition == kTransportAcceptedAwaitingEngine`. In the latter case
  /// the caller must not commit HTTP headers until `Accepted` is received.
  /// Failures use a sanitized ApiError (400/429/503).
  virtual boost::asio::awaitable<Result<EngineSubmission>> SubmitGenerate(
      EngineGenerateRequest request, const EngineCallContext& call) = 0;

  virtual boost::asio::awaitable<Result<EngineEmbedSubmission>> SubmitEmbed(
      EngineEmbedRequest request, const EngineCallContext& call) = 0;

  /// Idempotent. Safe to call from any thread. After this returns, no new
  /// events are produced, though a final event may already be in flight.
  virtual void Abort(EngineRequestId id, CancelReason reason) noexcept = 0;

  virtual boost::asio::awaitable<Result<std::vector<EngineModelInfo>>>
  ListModels() = 0;

  virtual boost::asio::awaitable<EngineHealth> CheckHealth() = 0;

  /// Stops accepting work and waits for outstanding transport work up to
  /// `grace`. Must not hang indefinitely.
  virtual boost::asio::awaitable<Result<void>> Shutdown(
      std::chrono::milliseconds grace) = 0;
};

/// \brief RAII generate-request handle with exactly-once cancellation.
///
/// `Cancel` is non-blocking, thread-safe, and schedules an abort on the
/// engine client when one is still reachable. The destructor cancels with
/// `CancelReason::kInternal` when the request never reached a terminal
/// state; it never blocks.
class EngineRequestHandle {
 public:
  EngineRequestHandle() = default;
  EngineRequestHandle(std::weak_ptr<IEngineClient> client, EngineRequestId id,
                      std::unique_ptr<IEngineEventStream> events);
  ~EngineRequestHandle();

  EngineRequestHandle(const EngineRequestHandle&) = delete;
  EngineRequestHandle& operator=(const EngineRequestHandle&) = delete;
  EngineRequestHandle(EngineRequestHandle&& other) noexcept;
  EngineRequestHandle& operator=(EngineRequestHandle&& other) noexcept;

  [[nodiscard]] const EngineRequestId& id() const noexcept;
  [[nodiscard]] IEngineEventStream& events() const;
  [[nodiscard]] bool valid() const noexcept;

  /// Exactly-once per handle. Non-blocking; safe from any thread.
  void Cancel(CancelReason reason) noexcept;

  [[nodiscard]] bool cancel_requested() const noexcept;

 private:
  struct State;
  std::shared_ptr<State> state_;
};

/// \brief Embedding counterpart of EngineRequestHandle.
class EngineEmbedHandle {
 public:
  EngineEmbedHandle() = default;
  EngineEmbedHandle(std::weak_ptr<IEngineClient> client, EngineRequestId id,
                    std::unique_ptr<IEngineEmbedStream> events);
  ~EngineEmbedHandle();

  EngineEmbedHandle(const EngineEmbedHandle&) = delete;
  EngineEmbedHandle& operator=(const EngineEmbedHandle&) = delete;
  EngineEmbedHandle(EngineEmbedHandle&& other) noexcept;
  EngineEmbedHandle& operator=(EngineEmbedHandle&& other) noexcept;

  [[nodiscard]] const EngineRequestId& id() const noexcept;
  [[nodiscard]] IEngineEmbedStream& events() const;
  [[nodiscard]] bool valid() const noexcept;
  void Cancel(CancelReason reason) noexcept;
  [[nodiscard]] bool cancel_requested() const noexcept;

 private:
  struct State;
  std::shared_ptr<State> state_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_ENGINE_CLIENT_H_
