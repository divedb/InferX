#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/engine/engine.h"

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

  /// Total generation deadline after scheduler submission. Expiry requests
  /// cancellation; engine cleanup remains asynchronous.
  int request_timeout_seconds = 600;

  /// Socket event-loop threads. Zero selects a small hardware-based default.
  size_t io_threads = 0;

  /// Fixed-size executor used for parsing/tokenization and the Engine's
  /// blocking event interface. Keeping it separate ensures a slow inference
  /// request can never stall accepts, reads, or response writes.
  size_t application_threads = 0;

  /// Hard cap across queued, tokenizing, generating, and streaming requests.
  /// Admission happens before work enters the application executor.
  size_t max_active_requests = 1024;

  /// Empty selects the in-process scheduler. A non-empty gRPC target (for
  /// example, "dns:///scheduler:50051") selects RemoteSchedulerClient.
  std::string scheduler_endpoint;

  /// Lower- or upper-case SHA-256 hex digests of accepted bearer tokens.
  /// Empty keeps authentication disabled for local deployments.
  std::vector<std::string> api_key_sha256;
};

/// \brief An OpenAI-compatible HTTP front end for an `Engine`.
///
/// Boost.Beast owns HTTP parsing and serialization and Boost.Asio owns socket
/// execution. Blocking application work is isolated on a bounded executor;
/// sockets and Beast message state never leave their owning I/O executor.
///
/// Endpoints:
///
///   POST /v1/chat/completions   blocking and SSE
///   POST /v1/completions        blocking and SSE
///   POST /v1/tokenize           exact token IDs and count
///   GET  /v1/models
///   GET  /health
///   GET  /metrics               Prometheus text exposition
///   GET  /stats                 legacy engine-statistics JSON
class HttpServer {
 public:
  /// \param engine Borrowed; must outlive the server.
  static StatusOr<std::unique_ptr<HttpServer>> Create(
      engine::Engine* engine, const HttpServerConfig& config);

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
