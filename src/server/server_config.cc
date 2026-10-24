#include "inferx/server/server_config.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "inferx/server/json_codec.h"

namespace inferx::server {

namespace {

namespace fs = std::filesystem;

ApiError Invalid(std::string message) {
  return ApiError{400,                    ApiErrorKind::kInvalidRequest,
                  "invalid_request_error", "invalid_config",
                  std::move(message),     std::nullopt,
                  std::nullopt,           false,
                  {}};
}

bool IsReadableFile(const std::string& path) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) return false;
  std::ifstream in(path);
  return in.good();
}

template <class Rep, class Period>
bool Positive(std::chrono::duration<Rep, Period> value) {
  return value.count() > 0;
}

const nlohmann::json* FindMember(const nlohmann::json& object,
                                 const char* key) {
  const auto it = object.find(key);
  return it == object.end() ? nullptr : &*it;
}

Result<void> RequireObject(const nlohmann::json& value, const char* section) {
  if (!value.is_object()) {
    return Invalid(std::string("config section '") + section +
                   "' must be an object");
  }
  return Result<void>{};
}

Result<void> CheckUnknownKeys(const nlohmann::json& object,
                              const char* section,
                              std::vector<std::string> allowed) {
  for (const auto& [key, value] : object.items()) {
    (void)value;
    if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
      return Invalid("unknown config key '" + std::string(section) + "." +
                     key + "'");
    }
  }
  return Result<void>{};
}

std::optional<std::size_t> GetSize(const nlohmann::json& value) {
  if (value.is_number_unsigned()) {
    return static_cast<std::size_t>(value.get<std::uint64_t>());
  }
  return std::nullopt;
}

Result<void> ApplyTlsConfig(const nlohmann::json& section, ServerConfig& out) {
  if (auto* v = FindMember(section, "enabled")) {
    if (!v->is_boolean()) return Invalid("tls.enabled must be a boolean");
    out.tls.enabled = v->get<bool>();
  }
  for (const auto& [key, slot] : std::initializer_list<
           std::pair<const char*, std::string*>>{
           {"cert_file", &out.tls.cert_file},
           {"key_file", &out.tls.key_file},
           {"ca_file", &out.tls.ca_file},
           {"min_version", &out.tls.min_version},
           {"ciphers", &out.tls.ciphers}}) {
    if (auto* v = FindMember(section, key)) {
      if (!v->is_string()) return Invalid(std::string("tls.") + key + " must be a string");
      *slot = v->get<std::string>();
    }
  }
  if (auto* v = FindMember(section, "require_client_cert")) {
    if (!v->is_boolean()) return Invalid("tls.require_client_cert must be a boolean");
    out.tls.require_client_cert = v->get<bool>();
  }
  return Result<void>{};
}

Result<void> ApplyLimitsConfig(const nlohmann::json& section, ServerConfig& out) {
  auto assign = [&](const char* key, std::size_t* slot) -> Result<void> {
    if (auto* v = FindMember(section, key)) {
      const auto parsed = GetSize(*v);
      if (!parsed) return Invalid(std::string("http.") + key + " must be a non-negative integer");
      *slot = *parsed;
    }
    return Result<void>{};
  };
  for (const auto& [key, slot] : std::initializer_list<
           std::pair<const char*, std::size_t*>>{
           {"max_request_line_bytes", &out.http.max_request_line_bytes},
           {"max_header_bytes", &out.http.max_header_bytes},
           {"max_header_count", &out.http.max_header_count},
           {"max_body_bytes", &out.http.max_body_bytes},
           {"max_json_depth", &out.http.max_json_depth},
           {"max_json_string_bytes", &out.http.max_json_string_bytes},
           {"max_url_bytes", &out.http.max_url_bytes},
           {"max_media_items", &out.http.max_media_items},
           {"max_media_bytes", &out.http.max_media_bytes},
           {"max_connections", &out.http.max_connections},
           {"max_requests_per_connection", &out.http.max_requests_per_connection}}) {
    auto applied = assign(key, slot);
    if (!applied) return applied;
  }
  return Result<void>{};
}

Result<void> ApplyTimeoutsConfig(const nlohmann::json& section,
                                 ServerConfig& out) {
  struct Entry {
    const char* key;
    std::chrono::milliseconds* slot;
  };
  const std::vector<Entry> entries = {
      {"header_read_ms", &out.timeouts.header_read},
      {"body_read_ms", &out.timeouts.body_read},
      {"keep_alive_ms", &out.timeouts.keep_alive},
      {"request_total_ms", &out.timeouts.request_total},
      {"queue_wait_ms", &out.timeouts.queue_wait},
      {"engine_accept_ms", &out.timeouts.engine_accept},
      {"stream_write_ms", &out.timeouts.stream_write},
      {"engine_cancel_ack_ms", &out.timeouts.engine_cancel_ack},
      {"drain_grace_ms", &out.timeouts.drain_grace},
      {"cancel_grace_ms", &out.timeouts.cancel_grace}};
  for (const auto& entry : entries) {
    if (auto* v = FindMember(section, entry.key)) {
      if (!v->is_number_unsigned()) {
        return Invalid(std::string("timeouts.") + entry.key +
                       " must be a non-negative integer (milliseconds)");
      }
      *entry.slot = std::chrono::milliseconds(
          static_cast<std::int64_t>(v->get<std::uint64_t>()));
    }
  }
  return Result<void>{};
}

Result<void> ApplyModelConfig(const nlohmann::json& root, ServerConfig& out) {
  if (auto* v = FindMember(root, "model_id")) {
    if (!v->is_string()) return Invalid("model_id must be a string");
    out.model_id = v->get<std::string>();
  }
  if (auto* v = FindMember(root, "served_model_names")) {
    if (!v->is_array()) return Invalid("served_model_names must be an array");
    out.served_model_names.clear();
    for (const auto& item : *v) {
      if (!item.is_string()) return Invalid("served_model_names entries must be strings");
      out.served_model_names.push_back(item.get<std::string>());
    }
  }
  if (auto* v = FindMember(root, "max_model_len")) {
    const auto parsed = GetSize(*v);
    if (!parsed) return Invalid("max_model_len must be a non-negative integer");
    out.max_model_len = *parsed;
  }
  return Result<void>{};
}

Result<void> ApplyJsonConfig(const nlohmann::json& root, ServerConfig& out) {
  if (!root.is_object()) return Invalid("config file root must be an object");
  std::vector<std::string> allowed = {
      "bind_host", "bind_port",     "io_threads", "model_id",
      "served_model_names", "max_model_len", "tls", "http", "timeouts"};
  if (auto applied = CheckUnknownKeys(root, "", std::move(allowed)); !applied) {
    return applied;
  }
  if (auto* v = FindMember(root, "bind_host")) {
    if (!v->is_string()) return Invalid("bind_host must be a string");
    out.bind_host = v->get<std::string>();
  }
  if (auto* v = FindMember(root, "bind_port")) {
    if (!v->is_number_unsigned() || v->get<std::uint64_t>() > 65535) {
      return Invalid("bind_port must be an integer in [0, 65535]");
    }
    out.bind_port = static_cast<std::uint16_t>(v->get<std::uint64_t>());
  }
  if (auto* v = FindMember(root, "io_threads")) {
    if (!v->is_number_unsigned() || v->get<std::uint64_t>() == 0) {
      return Invalid("io_threads must be a positive integer");
    }
    out.io_threads = static_cast<std::size_t>(v->get<std::uint64_t>());
  }
  if (auto applied = ApplyModelConfig(root, out); !applied) return applied;
  if (auto* section = FindMember(root, "tls")) {
    if (auto applied = RequireObject(*section, "tls"); !applied) return applied;
    std::vector<std::string> allowed = {"enabled",       "cert_file",
                                        "key_file",      "ca_file",
                                        "require_client_cert", "min_version",
                                        "ciphers"};
    if (auto applied = CheckUnknownKeys(*section, "tls", std::move(allowed));
        !applied) {
      return applied;
    }
    if (auto applied = ApplyTlsConfig(*section, out); !applied) return applied;
  }
  if (auto* section = FindMember(root, "http")) {
    if (auto applied = RequireObject(*section, "http"); !applied) return applied;
    if (auto applied = CheckUnknownKeys(*section, "http",
                                        {"max_request_line_bytes",
                                         "max_header_bytes", "max_header_count",
                                         "max_body_bytes", "max_json_depth",
                                         "max_json_string_bytes",
                                         "max_url_bytes", "max_media_items",
                                         "max_media_bytes", "max_connections",
                                         "max_requests_per_connection"});
        !applied) {
      return applied;
    }
    if (auto applied = ApplyLimitsConfig(*section, out); !applied) {
      return applied;
    }
  }
  if (auto* section = FindMember(root, "timeouts")) {
    if (auto applied = RequireObject(*section, "timeouts"); !applied) {
      return applied;
    }
    if (auto applied = CheckUnknownKeys(*section, "timeouts",
                                        {"header_read_ms", "body_read_ms",
                                         "keep_alive_ms", "request_total_ms",
                                         "queue_wait_ms", "engine_accept_ms",
                                         "stream_write_ms",
                                         "engine_cancel_ack_ms",
                                         "drain_grace_ms", "cancel_grace_ms"});
        !applied) {
      return applied;
    }
    if (auto applied = ApplyTimeoutsConfig(*section, out); !applied) {
      return applied;
    }
  }
  return Result<void>{};
}

Result<std::uint16_t> ParsePort(const std::string& text) {
  try {
    const unsigned long value = std::stoul(text);
    if (value > 65535) return Invalid("port out of range: " + text);
    return static_cast<std::uint16_t>(value);
  } catch (const std::exception&) {
    return Invalid("port is not a number: " + text);
  }
}

}  // namespace

Result<void> ValidateConfig(const ServerConfig& config) {
  if (config.bind_port > 65535) return Invalid("bind_port out of range");
  if (config.io_threads == 0) return Invalid("io_threads must be >= 1");
  if (config.cpu_workers == 0) return Invalid("cpu_workers must be >= 1");
  if (config.model_id.empty() && config.served_model_names.empty()) {
    return Invalid("model_id (or served_model_names) is required");
  }
  for (const auto& name : config.served_model_names) {
    if (name.empty()) return Invalid("served_model_names entries must be non-empty");
  }
  if (config.http.max_body_bytes == 0) return Invalid("max_body_bytes must be >= 1");
  if (config.http.max_json_depth < 2) return Invalid("max_json_depth must be >= 2");
  if (config.http.max_json_string_bytes == 0) {
    return Invalid("max_json_string_bytes must be >= 1");
  }
  if (config.http.max_url_bytes == 0) return Invalid("max_url_bytes must be >= 1");
  if (config.http.max_connections == 0) return Invalid("max_connections must be >= 1");
  if (config.http.max_header_count == 0) return Invalid("max_header_count must be >= 1");
  if (config.http.max_header_bytes == 0) return Invalid("max_header_bytes must be >= 1");
  if (config.http.max_request_line_bytes < 16) {
    return Invalid("max_request_line_bytes must be >= 16");
  }
  if (config.stream.engine_low_watermark_bytes > config.stream.engine_event_buffer_bytes) {
    return Invalid("engine_low_watermark_bytes must be <= engine_event_buffer_bytes");
  }
  if (config.stream.coalesce_max_bytes == 0) {
    return Invalid("coalesce_max_bytes must be >= 1");
  }
  if (!Positive(config.timeouts.header_read) ||
      !Positive(config.timeouts.body_read) ||
      !Positive(config.timeouts.keep_alive) ||
      !Positive(config.timeouts.request_total) ||
      !Positive(config.timeouts.queue_wait) ||
      !Positive(config.timeouts.engine_accept) ||
      !Positive(config.timeouts.stream_write) ||
      !Positive(config.timeouts.engine_cancel_ack) ||
      !Positive(config.timeouts.drain_grace) ||
      !Positive(config.timeouts.cancel_grace)) {
    return Invalid("all timeouts must be positive");
  }
  if (config.tls.enabled) {
    // TLS termination is a later phase; refuse to start with it enabled.
    return Invalid("tls.enabled is not supported by this build yet");
  }
  if (config.tls.cert_file.empty() != config.tls.key_file.empty()) {
    return Invalid("tls cert_file and key_file must be set together");
  }
  if (!config.tls.cert_file.empty() && !IsReadableFile(config.tls.cert_file)) {
    return Invalid("tls cert_file is not a readable file: " + config.tls.cert_file);
  }
  if (!config.tls.key_file.empty() && !IsReadableFile(config.tls.key_file)) {
    return Invalid("tls key_file is not a readable file: " + config.tls.key_file);
  }
  if (config.cors.enabled && config.cors.allow_credentials) {
    for (const auto& origin : config.cors.allow_origins) {
      if (origin == "*") {
        return Invalid("cors '*' origins cannot be used with credentials");
      }
    }
  }
  if (config.auth.enabled) {
    if (config.auth.api_keys.empty() && config.auth.api_keys_file.empty()) {
      return Invalid("auth.enabled requires api_keys or api_keys_file");
    }
    if (!config.auth.api_keys_file.empty() &&
        !IsReadableFile(config.auth.api_keys_file)) {
      return Invalid("auth api_keys_file is not readable: " + config.auth.api_keys_file);
    }
  }
  if (config.engine.max_inflight == 0) return Invalid("engine.max_inflight must be >= 1");
  if (config.observability.trace_sample_ratio < 0.0 ||
      config.observability.trace_sample_ratio > 1.0) {
    return Invalid("trace_sample_ratio must be within [0, 1]");
  }
  return Result<void>{};
}

Result<ServerConfig> LoadConfigFile(const std::string& path) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec)) {
    return Invalid("config file not found: " + path);
  }
  std::ifstream in(path);
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();

  HttpLimits config_limits;  // config files get a higher depth cap
  config_limits.max_json_depth = 128;
  config_limits.max_json_string_bytes = 8u * 1024u * 1024u;
  auto parsed = ParseJson(text, config_limits, 32u * 1024u * 1024u);
  if (!parsed) {
    return ApiError{400,                    ApiErrorKind::kInvalidRequest,
                    "invalid_request_error", "invalid_config",
                    "config file is invalid: " + parsed.error().message,
                    std::nullopt,           std::nullopt,
                    false,                  {}};
  }
  ServerConfig config;
  if (auto applied = ApplyJsonConfig(parsed.value(), config); !applied) {
    return applied.error();
  }
  return config;
}

Result<void> ApplyEnvOverrides(ServerConfig& config) {
  if (const char* host = std::getenv("INFERX_SERVER_HOST")) {
    config.bind_host = host;
  }
  if (const char* port = std::getenv("INFERX_SERVER_PORT")) {
    auto parsed = ParsePort(port);
    if (!parsed) return parsed.error();
    config.bind_port = parsed.value();
  }
  if (const char* threads = std::getenv("INFERX_SERVER_IO_THREADS")) {
    try {
      const unsigned long value = std::stoul(threads);
      if (value == 0) return Invalid("INFERX_SERVER_IO_THREADS must be >= 1");
      config.io_threads = static_cast<std::size_t>(value);
    } catch (const std::exception&) {
      return Invalid("INFERX_SERVER_IO_THREADS is not a number");
    }
  }
  if (const char* model = std::getenv("INFERX_SERVER_MODEL")) {
    config.model_id = model;
  }
  if (const char* names = std::getenv("INFERX_SERVED_MODEL_NAMES")) {
    config.served_model_names.clear();
    std::string current;
    for (const char c : std::string(names)) {
      if (c == ',') {
        if (!current.empty()) config.served_model_names.push_back(current);
        current.clear();
      } else {
        current.push_back(c);
      }
    }
    if (!current.empty()) config.served_model_names.push_back(current);
  }
  return Result<void>{};
}

Result<ServerConfig> LoadConfig(int argc, char** argv) {
  // Pass 1: parse the CLI layer (highest precedence) into override slots.
  std::string config_file;
  std::optional<std::string> cli_host;
  std::optional<std::uint16_t> cli_port;
  std::optional<std::string> cli_model;
  std::vector<std::string> served_names;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto next_value = [&](const char* name) -> Result<std::string> {
      if (i + 1 >= argc) return Invalid(std::string(name) + " requires a value");
      return std::string(argv[++i]);
    };
    if (flag == "--config") {
      auto value = next_value("--config");
      if (!value) return value.error();
      config_file = std::move(value.value());
    } else if (flag == "--host") {
      auto value = next_value("--host");
      if (!value) return value.error();
      cli_host = std::move(value.value());
    } else if (flag == "--port") {
      auto value = next_value("--port");
      if (!value) return value.error();
      auto port = ParsePort(value.value());
      if (!port) return port.error();
      cli_port = port.value();
    } else if (flag == "--model") {
      auto value = next_value("--model");
      if (!value) return value.error();
      cli_model = std::move(value.value());
    } else if (flag == "--served-model-name") {
      auto value = next_value("--served-model-name");
      if (!value) return value.error();
      served_names.push_back(std::move(value.value()));
    } else {
      return Invalid("unknown flag: " + flag);
    }
  }

  // Pass 2: defaults < config file < environment < CLI.
  ServerConfig config;
  if (!config_file.empty()) {
    auto loaded = LoadConfigFile(config_file);
    if (!loaded) return loaded.error();
    config = std::move(loaded.value());
  }
  if (auto env = ApplyEnvOverrides(config); !env) return env.error();
  if (cli_host) config.bind_host = *std::move(cli_host);
  if (cli_port) config.bind_port = *cli_port;
  if (cli_model) config.model_id = *std::move(cli_model);
  if (!served_names.empty()) config.served_model_names = std::move(served_names);
  return config;
}

}  // namespace inferx::server
