#pragma once

#include <memory>
#include <string>

#include "inferx/core/status.h"
#include "inferx/server/engine.h"

namespace inferx::server {

struct HttpServerConfig {
  std::string host = "127.0.0.1";
  int port = 8000;

  /// Rejected above this, before the body is buffered. A prompt is bounded by
  /// `max_seq_len` anyway, so anything larger is either a mistake or an attempt
  /// to make the server allocate on demand.
  size_t max_request_bytes = 8u * 1024 * 1024;

  int read_timeout_seconds = 30;

  /// Generous, because a non-streaming request holds the connection open for
  /// the whole generation and the default socket timeout would cut it off
  /// mid-answer.
  int write_timeout_seconds = 600;
};

/// \brief An OpenAI-compatible HTTP front end for an `Engine`.
///
/// ARCHITECTURE.md §9 nominates Boost.Beast, and §5.1 describes the I/O around
/// an Asio `io_context` with C++20 coroutines. This uses cpp-httplib instead --
/// a single vendored header, thread-per-connection -- because Beast would have
/// meant fetching the Boost superproject to serve a handful of endpoints, and
/// the interface here is small enough that swapping the implementation is
/// contained.
///
/// What that defers is real and worth naming: thread-per-connection puts a
/// ceiling on concurrent streams that an event loop would not have, and §5.1's
/// coroutine design is not exercised at all. Neither binds at the concurrency a
/// single 4080 SUPER can serve, since the batch is capped by `max_running`
/// long before the connection count is. When it does bind, Beast is the answer.
///
/// Endpoints:
///
///   POST /v1/chat/completions   blocking and SSE
///   POST /v1/completions        blocking and SSE
///   GET  /v1/models
///   GET  /health
///   GET  /metrics               Prometheus text exposition
///   GET  /stats                 legacy engine-statistics JSON
class HttpServer {
 public:
  /// \param engine Borrowed; must outlive the server.
  static StatusOr<std::unique_ptr<HttpServer>> Create(
      Engine* engine, const HttpServerConfig& config);

  ~HttpServer();

  HttpServer(const HttpServer&) = delete;
  HttpServer& operator=(const HttpServer&) = delete;

  /// \brief Binds and serves until `Stop`. Blocks.
  Status Listen();

  /// \brief Stops `Listen`. Safe from any thread, including a signal handler's
  ///        continuation.
  void Stop();

  /// \brief Blocks until the listener is accepting. For tests.
  bool WaitUntilReady();

  int port() const;

 private:
  struct Impl;

  explicit HttpServer(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server
