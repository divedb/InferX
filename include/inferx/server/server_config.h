#ifndef INFERX_SERVER_SERVER_CONFIG_H_
#define INFERX_SERVER_SERVER_CONFIG_H_

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "inferx/server/status.h"

namespace inferx::server {

struct TlsConfig {
  bool enabled = false;
  std::string cert_file;
  std::string key_file;
  std::string ca_file;  // optional client CA
  bool require_client_cert = false;
  std::string min_version = "TLS1.2";
  std::string ciphers;  // OpenSSL cipher list, optional
  bool reload_on_sighup = false;
};

struct HttpLimits {
  std::size_t max_request_line_bytes = 8u * 1024u;
  std::size_t max_header_bytes = 32u * 1024u;
  std::size_t max_header_count = 100;
  std::size_t max_body_bytes = 8u * 1024u * 1024u;
  std::size_t max_json_depth = 64;
  std::size_t max_json_string_bytes = 1u * 1024u * 1024u;
  std::size_t max_url_bytes = 8u * 1024u;
  std::size_t max_media_items = 8;
  std::size_t max_media_bytes = 16u * 1024u * 1024u;
  std::size_t max_connections = 4096;
  std::size_t max_requests_per_connection = 1000;
};

struct StreamLimits {
  std::size_t engine_event_buffer_events = 256;
  std::size_t engine_event_buffer_bytes = 1u * 1024u * 1024u;
  std::size_t engine_low_watermark_bytes = 256u * 1024u;
  std::size_t coalesce_max_bytes = 32u * 1024u;
  std::chrono::milliseconds keep_alive_interval{15'000};
  std::chrono::milliseconds max_idle{120'000};
};

struct TimeoutConfig {
  std::chrono::milliseconds header_read{10'000};
  std::chrono::milliseconds body_read{30'000};
  std::chrono::milliseconds keep_alive{5'000};
  std::chrono::milliseconds request_total{600'000};
  std::chrono::milliseconds queue_wait{5'000};
  std::chrono::milliseconds engine_accept{30'000};
  std::chrono::milliseconds stream_write{30'000};
  std::chrono::milliseconds engine_cancel_ack{2'000};
  std::chrono::milliseconds drain_grace{30'000};
  std::chrono::milliseconds cancel_grace{5'000};
};

struct AuthConfig {
  bool enabled = false;
  // Keys may come from a file or environment. Plaintext in process config is
  // hashed at startup and never logged.
  std::vector<std::string> api_keys;
  std::string api_keys_file;
  std::vector<std::string> guarded_prefixes{"/v1", "/v2", "/inference"};
};

struct CorsConfig {
  bool enabled = false;
  std::vector<std::string> allow_origins;
  std::vector<std::string> allow_methods{"GET", "POST", "OPTIONS"};
  std::vector<std::string> allow_headers{"Authorization", "Content-Type",
                                         "X-Request-Id"};
  bool allow_credentials = false;
  std::chrono::seconds max_age{600};
};

enum class EngineTransport : std::uint8_t {
  kLocal,
  kUnixDomainSocket,
  kGrpc,
};

struct EngineConfig {
  EngineTransport transport = EngineTransport::kUnixDomainSocket;
  std::string uds_path = "/run/inferx/engine.sock";
  std::string grpc_target = "127.0.0.1:50051";
  std::chrono::milliseconds connect_timeout{5'000};
  std::chrono::milliseconds rpc_timeout{30'000};
  std::chrono::milliseconds health_timeout{1'000};
  std::chrono::milliseconds reconnect_backoff_min{100};
  std::chrono::milliseconds reconnect_backoff_max{10'000};
  std::size_t max_inflight = 4096;
  bool reject_on_engine_death = true;
};

enum class UnknownFieldPolicy : std::uint8_t { kIgnoreAndLog, kReject };

struct ObservabilityConfig {
  std::string log_level = "info";
  bool json_logs = true;
  bool log_prompts = false;  // off by default
  bool metrics_enabled = true;
  bool metrics_require_auth = true;
  std::string otlp_endpoint;  // empty disables tracing
  double trace_sample_ratio = 0.01;
};

/// \brief Fully validated typed configuration. Immutable after Start.
struct ServerConfig {
  std::string bind_host = "0.0.0.0";
  std::uint16_t bind_port = 8000;
  std::size_t io_threads = 4;
  std::size_t cpu_workers = 2;
  std::size_t max_inflight_bytes = 256u * 1024u * 1024u;

  TlsConfig tls;
  HttpLimits http;
  StreamLimits stream;
  TimeoutConfig timeouts;
  AuthConfig auth;
  CorsConfig cors;
  EngineConfig engine;
  ObservabilityConfig observability;

  std::string model_id;
  std::vector<std::string> served_model_names;
  std::optional<std::uint64_t> max_model_len;
  bool allow_runtime_lora = false;
  bool trust_request_chat_template = false;
  UnknownFieldPolicy unknown_fields = UnknownFieldPolicy::kIgnoreAndLog;
  bool enable_legacy_completions = true;
  bool enable_embeddings = true;
  bool enable_tokenize = false;
};

/// \brief Validates ranges, mutually exclusive options, stream watermarks,
///        and positive timeouts. Must be called before any socket is bound.
[[nodiscard]] Result<void> ValidateConfig(const ServerConfig& config);

/// \brief Loads the JSON config file at `path`. Unknown keys are rejected
///        (unlike API requests); values are merged over the defaults.
[[nodiscard]] Result<ServerConfig> LoadConfigFile(const std::string& path);

/// \brief Applies INFERX_SERVER_* environment overrides in place.
///        Recognized: INFERX_SERVER_HOST, INFERX_SERVER_PORT,
///        INFERX_SERVER_IO_THREADS, INFERX_SERVER_MODEL,
///        INFERX_SERVED_MODEL_NAMES (comma-separated). Returns the first
///        parse failure; earlier variables stay applied.
[[nodiscard]] Result<void> ApplyEnvOverrides(ServerConfig& config);

/// \brief Precedence: CLI > environment > config file > defaults. The CLI
///        layer recognizes --config <path>, --host, --port, --model, and
///        --served-model-name (repeatable); unknown flags are rejected.
[[nodiscard]] Result<ServerConfig> LoadConfig(int argc, char** argv);

}  // namespace inferx::server

#endif  // INFERX_SERVER_SERVER_CONFIG_H_
