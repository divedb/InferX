#ifndef INFERX_CONFIG_SERVER_CONFIG_H_
#define INFERX_CONFIG_SERVER_CONFIG_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "nlohmann/json_fwd.hpp"

#include "inferx/config/api_server_config.h"
#include "inferx/core/status.h"

namespace inferx {

/// \brief All tunables of the serving frontend: network and auth settings,
///        executors, limits, and timeouts.
///
/// All sizes are bytes or element counts as named; all timeouts use the
/// monotonic clock and are integer seconds in JSON. Values come from
/// `inferx serve` flags, a JSON document (FromJson), and environment
/// overrides (MergeEnvironmentOverrides).
struct ServerConfig {
  /// Listen address, port, and bearer API keys guarding /v1 and /metrics;
  /// empty api_keys disables auth.
  /// EXAMPLE: --api-key secret
  ApiServerConfig api;

  /// I/O threads running the accept loop, session coroutines, parsing, and
  /// timers; the default is capped at 4.
  /// EXAMPLE: "io_threads": 4
  std::size_t io_threads =
      std::min<std::size_t>(4, std::max(1u, std::thread::hardware_concurrency()));

  /// Worker threads preparing prompts (chat-template rendering and
  /// tokenization); each worker owns a private tokenizer instance.
  /// EXAMPLE: --tokenizer-workers 2
  std::size_t tokenizer_workers = 2;

  /// Maximum prompt-preparation jobs waiting for a worker; beyond it,
  /// requests fail fast with 503 instead of queueing.
  /// EXAMPLE: --tokenizer-queue-limit 64
  std::size_t tokenizer_queue_limit = 64;

  /// Maximum simultaneously open connections; excess accepts are closed
  /// with a best-effort 503 and counted.
  /// EXAMPLE: "max_connections": 4096
  std::size_t max_connections = 4096;

  /// Maximum inference requests admitted at once — preparing, streaming,
  /// or cancelled but not yet released; excess requests get 503 with
  /// Retry-After.
  /// EXAMPLE: "max_inflight_requests": 512
  std::size_t max_inflight_requests = 512;

  /// Maximum HTTP header bytes per request; larger is rejected with 431.
  /// EXAMPLE: "header_limit": 16384
  std::size_t header_limit = 16 * 1024;

  /// Maximum request body bytes; larger is rejected with 413.
  /// EXAMPLE: "body_limit": 4194304
  std::size_t body_limit = 4 * 1024 * 1024;

  /// Hard per-request cap on tokens buffered between the engine and the
  /// HTTP writer; overflow cancels the request as a slow client.
  /// EXAMPLE: "max_pending_tokens": 1024
  std::size_t max_pending_tokens = 1024;

  /// Maximum decoded output bytes per request (must be at least
  /// max_write_batch_bytes); excess fails the request with
  /// output_limit_exceeded.
  /// EXAMPLE: "max_output_bytes": 8388608
  std::size_t max_output_bytes = 8 * 1024 * 1024;

  /// Maximum bytes of one serialized SSE write batch; larger deltas are
  /// split on UTF-8 boundaries.
  /// EXAMPLE: "max_write_batch_bytes": 65536
  std::size_t max_write_batch_bytes = 64 * 1024;

  /// Bytes the disconnect watcher may read ahead of the current request
  /// before the connection is closed.
  /// EXAMPLE: "max_read_ahead_bytes": 16384
  std::size_t max_read_ahead_bytes = 16 * 1024;

  /// Default completions max_tokens when both the request and the
  /// checkpoint are silent (the OpenAI completions default).
  /// EXAMPLE: "default_completion_max_tokens": 16
  std::uint32_t default_completion_max_tokens = 16;

  /// Commit SSE headers at acceptance with keep-alive comment lines
  /// instead of at the first token; trades real HTTP error statuses for
  /// earlier first bytes.
  /// EXAMPLE: "sse_early_headers": true
  bool sse_early_headers = false;

  /// Time allowed to receive request headers; expiry answers 408 when
  /// possible and closes.
  /// EXAMPLE: "header_read_timeout": 10
  std::chrono::seconds header_read_timeout{10};

  /// Time allowed to receive a request body; expiry answers 408 and closes.
  /// EXAMPLE: "body_read_timeout": 30
  std::chrono::seconds body_read_timeout{30};

  /// Idle time a keep-alive connection may wait for the next request
  /// before being closed silently.
  /// EXAMPLE: "keepalive_idle_timeout": 60
  std::chrono::seconds keepalive_idle_timeout{60};

  /// Time from engine acceptance to the first token or terminal event,
  /// including prefill; expiry answers 503 queue_timeout and cancels.
  /// EXAMPLE: "queue_wait_timeout": 30
  std::chrono::seconds queue_wait_timeout{30};

  /// Total per-request budget; expiry answers 504 before the headers are
  /// committed, or an in-stream error event after.
  /// EXAMPLE: "request_timeout": 600
  std::chrono::seconds request_timeout{600};

  /// Time allowed for one logical stream write; a client that stops
  /// draining is cancelled and the connection closed.
  /// EXAMPLE: "stream_write_timeout": 15
  std::chrono::seconds stream_write_timeout{15};

  /// Grace period for in-flight requests to finish after a shutdown signal
  /// before they are cancelled.
  /// EXAMPLE: "drain_timeout": 30
  std::chrono::seconds drain_timeout{30};

  /// Time cleanup waits for the engine to confirm a cancellation before
  /// abandoning the request.
  /// EXAMPLE: "cancel_grace": 5
  std::chrono::seconds cancel_grace{5};

  /// Log prompt and completion payloads; debug builds only — never enable
  /// in production.
  /// EXAMPLE: "log_payloads": true
  bool log_payloads = false;

  /// \brief Range and consistency checks for every field.
  ///
  /// \return OK, or kInvalidArgument naming the offending field.
  [[nodiscard]] Status Validate() const;

  /// \brief Parses a configuration object. Field names are the struct field
  ///        names; durations are integer seconds; the `api` sub-object
  ///        carries host/port/api_keys. Unknown keys are rejected (typo
  ///        safety), wrong types are rejected, absent keys keep defaults.
  ///
  /// \param json Configuration object.
  /// \return     The parsed configuration, or kInvalidArgument.
  static StatusOr<ServerConfig> FromJson(const nlohmann::json& json);

  /// \brief Serializes the full configuration; FromJson(ToJson(c)) == c for
  ///        any valid c.
  ///
  /// \return The configuration as a JSON object.
  [[nodiscard]] nlohmann::json ToJson() const;
};

/// \brief Merges environment overrides into `config`.
///
/// \param config Configuration to update in place. INFERX_API_KEY appends a
///               non-empty bearer key alongside --api-key values.
void MergeEnvironmentOverrides(ServerConfig* config);

}  // namespace inferx

#endif  // INFERX_CONFIG_SERVER_CONFIG_H_
