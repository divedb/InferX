#include "inferx/config/server_config.h"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "nlohmann/json.hpp"

namespace inferx {
namespace {

using json = nlohmann::json;

Status FieldError(const char* field, const std::string& detail) {
  return InvalidArgumentError("server config: ", field, ": ", detail);
}

/// Reads a size field: must be an unsigned integer when present.
Status ReadSize(const json& j, const char* field, std::size_t* out) {
  const auto it = j.find(field);
  if (it == j.end()) return OkStatus();
  if (!it->is_number_unsigned()) {
    return FieldError(field, "must be a non-negative integer");
  }
  *out = it->get<std::size_t>();
  return OkStatus();
}

Status ReadUint32(const json& j, const char* field, std::uint32_t* out) {
  const auto it = j.find(field);
  if (it == j.end()) return OkStatus();
  if (!it->is_number_unsigned() || it->get<std::uint64_t>() > UINT32_MAX) {
    return FieldError(field, "must be a non-negative 32-bit integer");
  }
  *out = it->get<std::uint32_t>();
  return OkStatus();
}

Status ReadSeconds(const json& j, const char* field, std::chrono::seconds* out) {
  const auto it = j.find(field);
  if (it == j.end()) return OkStatus();
  if (!it->is_number_unsigned()) {
    return FieldError(field, "must be a non-negative integer (seconds)");
  }
  *out = std::chrono::seconds(it->get<std::uint64_t>());
  return OkStatus();
}

Status ReadBool(const json& j, const char* field, bool* out) {
  const auto it = j.find(field);
  if (it == j.end()) return OkStatus();
  if (!it->is_boolean()) return FieldError(field, "must be a boolean");
  *out = it->get<bool>();
  return OkStatus();
}

/// The known top-level keys; anything else is a typo and rejected.
bool IsKnownKey(const std::string& key) {
  static const std::vector<std::string> kKeys = {
      "api",
      "io_threads",
      "tokenizer_workers",
      "tokenizer_queue_limit",
      "max_connections",
      "max_inflight_requests",
      "header_limit",
      "body_limit",
      "max_pending_tokens",
      "max_output_bytes",
      "max_write_batch_bytes",
      "max_read_ahead_bytes",
      "default_completion_max_tokens",
      "sse_early_headers",
      "header_read_timeout",
      "body_read_timeout",
      "keepalive_idle_timeout",
      "queue_wait_timeout",
      "request_timeout",
      "stream_write_timeout",
      "drain_timeout",
      "cancel_grace",
      "log_payloads",
  };
  for (const std::string& known : kKeys) {
    if (key == known) return true;
  }
  return false;
}

Status ReadApi(const json& j, ApiServerConfig* api) {
  const auto it = j.find("api");
  if (it == j.end()) return OkStatus();
  if (!it->is_object()) return FieldError("api", "must be an object");
  if (const auto host = it->find("host"); host != it->end()) {
    if (!host->is_string()) return FieldError("api.host", "must be a string");
    api->host = host->get<std::string>();
  }
  if (const auto port = it->find("port"); port != it->end()) {
    if (!port->is_number_unsigned() || port->get<std::uint64_t>() > 65535) {
      return FieldError("api.port", "must be an integer in [1, 65535]");
    }
    api->port = port->get<int>();
  }
  if (const auto keys = it->find("api_keys"); keys != it->end()) {
    if (!keys->is_array()) return FieldError("api.api_keys", "must be an array");
    api->api_keys.clear();
    for (const auto& key : *keys) {
      if (!key.is_string()) {
        return FieldError("api.api_keys", "entries must be strings");
      }
      api->api_keys.push_back(key.get<std::string>());
    }
  }
  return OkStatus();
}

Status CheckPositive(std::size_t value, const char* field) {
  if (value == 0) return FieldError(field, "must be at least 1");
  return OkStatus();
}

template <typename Rep>
Status CheckPositive(std::chrono::duration<Rep> value, const char* field) {
  if (value <= std::chrono::duration<Rep>::zero()) {
    return FieldError(field, "must be positive");
  }
  return OkStatus();
}

}  // namespace

Status ServerConfig::Validate() const {
  if (api.host.empty()) return FieldError("api.host", "must not be empty");
  if (api.port < 1 || api.port > 65535) {
    return FieldError("api.port", "must be in [1, 65535]");
  }
  INFERX_RETURN_IF_ERROR(CheckPositive(io_threads, "io_threads"));
  INFERX_RETURN_IF_ERROR(CheckPositive(tokenizer_workers, "tokenizer_workers"));
  INFERX_RETURN_IF_ERROR(CheckPositive(tokenizer_queue_limit, "tokenizer_queue_limit"));
  INFERX_RETURN_IF_ERROR(CheckPositive(max_connections, "max_connections"));
  INFERX_RETURN_IF_ERROR(CheckPositive(max_inflight_requests, "max_inflight_requests"));
  if (header_limit < 512) return FieldError("header_limit", "must be at least 512");
  if (body_limit < 1024) return FieldError("body_limit", "must be at least 1024");
  INFERX_RETURN_IF_ERROR(CheckPositive(max_pending_tokens, "max_pending_tokens"));
  INFERX_RETURN_IF_ERROR(CheckPositive(max_read_ahead_bytes, "max_read_ahead_bytes"));
  if (max_write_batch_bytes < 1 || max_output_bytes < max_write_batch_bytes) {
    return FieldError("max_output_bytes",
                      "must be at least max_write_batch_bytes");
  }
  if (default_completion_max_tokens < 1) {
    return FieldError("default_completion_max_tokens", "must be at least 1");
  }
  INFERX_RETURN_IF_ERROR(CheckPositive(header_read_timeout, "header_read_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(body_read_timeout, "body_read_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(keepalive_idle_timeout, "keepalive_idle_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(queue_wait_timeout, "queue_wait_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(request_timeout, "request_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(stream_write_timeout, "stream_write_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(drain_timeout, "drain_timeout"));
  INFERX_RETURN_IF_ERROR(CheckPositive(cancel_grace, "cancel_grace"));
  return OkStatus();
}

StatusOr<ServerConfig> ServerConfig::FromJson(const json& json_in) {
  if (!json_in.is_object()) {
    return InvalidArgumentError("server config: must be a JSON object");
  }
  for (auto it = json_in.begin(); it != json_in.end(); ++it) {
    if (!IsKnownKey(it.key())) {
      return FieldError(it.key().c_str(), "unknown field");
    }
  }

  ServerConfig config;
  INFERX_RETURN_IF_ERROR(ReadApi(json_in, &config.api));
  INFERX_RETURN_IF_ERROR(ReadSize(json_in, "io_threads", &config.io_threads));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "tokenizer_workers", &config.tokenizer_workers));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "tokenizer_queue_limit", &config.tokenizer_queue_limit));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "max_connections", &config.max_connections));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "max_inflight_requests", &config.max_inflight_requests));
  INFERX_RETURN_IF_ERROR(ReadSize(json_in, "header_limit", &config.header_limit));
  INFERX_RETURN_IF_ERROR(ReadSize(json_in, "body_limit", &config.body_limit));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "max_pending_tokens", &config.max_pending_tokens));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "max_output_bytes", &config.max_output_bytes));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "max_write_batch_bytes", &config.max_write_batch_bytes));
  INFERX_RETURN_IF_ERROR(
      ReadSize(json_in, "max_read_ahead_bytes", &config.max_read_ahead_bytes));
  INFERX_RETURN_IF_ERROR(ReadUint32(json_in, "default_completion_max_tokens",
                                    &config.default_completion_max_tokens));
  INFERX_RETURN_IF_ERROR(ReadBool(json_in, "sse_early_headers",
                                  &config.sse_early_headers));
  INFERX_RETURN_IF_ERROR(
      ReadSeconds(json_in, "header_read_timeout", &config.header_read_timeout));
  INFERX_RETURN_IF_ERROR(
      ReadSeconds(json_in, "body_read_timeout", &config.body_read_timeout));
  INFERX_RETURN_IF_ERROR(ReadSeconds(json_in, "keepalive_idle_timeout",
                                     &config.keepalive_idle_timeout));
  INFERX_RETURN_IF_ERROR(
      ReadSeconds(json_in, "queue_wait_timeout", &config.queue_wait_timeout));
  INFERX_RETURN_IF_ERROR(
      ReadSeconds(json_in, "request_timeout", &config.request_timeout));
  INFERX_RETURN_IF_ERROR(
      ReadSeconds(json_in, "stream_write_timeout", &config.stream_write_timeout));
  INFERX_RETURN_IF_ERROR(ReadSeconds(json_in, "drain_timeout", &config.drain_timeout));
  INFERX_RETURN_IF_ERROR(ReadSeconds(json_in, "cancel_grace", &config.cancel_grace));
  INFERX_RETURN_IF_ERROR(ReadBool(json_in, "log_payloads", &config.log_payloads));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

json ServerConfig::ToJson() const {
  json out;
  out["api"] = {{"host", api.host},
                // Unsigned so FromJson's strict type check round-trips.
                {"port", static_cast<std::uint64_t>(api.port)},
                {"api_keys", json(api.api_keys)}};
  out["io_threads"] = io_threads;
  out["tokenizer_workers"] = tokenizer_workers;
  out["tokenizer_queue_limit"] = tokenizer_queue_limit;
  out["max_connections"] = max_connections;
  out["max_inflight_requests"] = max_inflight_requests;
  out["header_limit"] = header_limit;
  out["body_limit"] = body_limit;
  out["max_pending_tokens"] = max_pending_tokens;
  out["max_output_bytes"] = max_output_bytes;
  out["max_write_batch_bytes"] = max_write_batch_bytes;
  out["max_read_ahead_bytes"] = max_read_ahead_bytes;
  out["default_completion_max_tokens"] = default_completion_max_tokens;
  out["sse_early_headers"] = sse_early_headers;
  // Durations are serialized unsigned so FromJson's strict type check
  // round-trips; Validate() has already ruled out negatives.
  out["header_read_timeout"] = static_cast<std::uint64_t>(header_read_timeout.count());
  out["body_read_timeout"] = static_cast<std::uint64_t>(body_read_timeout.count());
  out["keepalive_idle_timeout"] =
      static_cast<std::uint64_t>(keepalive_idle_timeout.count());
  out["queue_wait_timeout"] = static_cast<std::uint64_t>(queue_wait_timeout.count());
  out["request_timeout"] = static_cast<std::uint64_t>(request_timeout.count());
  out["stream_write_timeout"] = static_cast<std::uint64_t>(stream_write_timeout.count());
  out["drain_timeout"] = static_cast<std::uint64_t>(drain_timeout.count());
  out["cancel_grace"] = static_cast<std::uint64_t>(cancel_grace.count());
  out["log_payloads"] = log_payloads;
  return out;
}

void MergeEnvironmentOverrides(ServerConfig* config) {
  if (const char* env = std::getenv("INFERX_API_KEY"); env != nullptr && env[0] != '\0') {
    config->api.api_keys.push_back(env);
  }
}

}  // namespace inferx
