#include "inferx/server/config/server_config.h"

namespace inferx::server::config {

Status Validate(const ServerConfig& config) {
  if (config.listen_address.empty()) {
    return InvalidArgumentError("listen address must not be empty");
  }
  if (config.port == 0) return InvalidArgumentError("port must be nonzero");
  if (config.io_threads == 0 || config.coroutine_threads == 0) {
    return InvalidArgumentError("server thread counts must be positive");
  }
  if (config.max_request_body_bytes == 0) {
    return InvalidArgumentError("request body limit must be positive");
  }
  if (config.header_timeout <= std::chrono::seconds::zero() ||
      config.request_timeout <= std::chrono::seconds::zero() ||
      config.shutdown_grace < std::chrono::seconds::zero()) {
    return InvalidArgumentError("server timeouts are invalid");
  }
  if (config.admin_enabled) {
    if (config.admin_listen_address.empty() || config.admin_port == 0) {
      return InvalidArgumentError("enabled admin listener requires address and port");
    }
    if (config.admin_listen_address == config.listen_address &&
        config.admin_port == config.port) {
      return InvalidArgumentError("admin and public listeners must be distinct");
    }
  }
  if (config.development_auth_bypass &&
      config.listen_address != "127.0.0.1" && config.listen_address != "::1") {
    return InvalidArgumentError(
        "development authentication bypass requires loopback listener");
  }
  if (config.max_active_requests == 0 ||
      config.max_active_per_tenant == 0 ||
      config.max_active_per_tenant > config.max_active_requests ||
      config.max_reserved_tokens == 0) {
    return InvalidArgumentError("admission limits are invalid");
  }
  if (config.max_completed_snapshots == 0 ||
      config.completed_retention < std::chrono::seconds::zero()) {
    return InvalidArgumentError("completed request retention is invalid");
  }
  return OkStatus();
}

ConfigStore::ConfigStore(ServerConfig config)
    : config_(std::make_shared<const ServerConfig>(std::move(config))) {}

StatusOr<std::shared_ptr<ConfigStore>> ConfigStore::Create(ServerConfig config) {
  INFERX_RETURN_IF_ERROR(Validate(config));
  return std::shared_ptr<ConfigStore>(new ConfigStore(std::move(config)));
}

std::shared_ptr<const ServerConfig> ConfigStore::Snapshot() const {
  return config_.load();
}

Status ConfigStore::Reload(const ServerConfig& candidate) {
  INFERX_RETURN_IF_ERROR(Validate(candidate));
  const auto current = config_.load();
  if (candidate.listen_address != current->listen_address ||
      candidate.port != current->port ||
      candidate.io_threads != current->io_threads ||
      candidate.coroutine_threads != current->coroutine_threads ||
      candidate.max_request_body_bytes != current->max_request_body_bytes ||
      candidate.header_timeout != current->header_timeout ||
      candidate.request_timeout != current->request_timeout ||
      candidate.shutdown_grace != current->shutdown_grace ||
      candidate.admin_enabled != current->admin_enabled ||
      candidate.admin_listen_address != current->admin_listen_address ||
      candidate.admin_port != current->admin_port ||
      candidate.development_auth_bypass != current->development_auth_bypass) {
    return FailedPreconditionError(
        "reload attempted to change immutable server configuration");
  }
  config_.store(std::make_shared<const ServerConfig>(candidate));
  revision_.fetch_add(1);
  return OkStatus();
}

}  // namespace inferx::server::config
