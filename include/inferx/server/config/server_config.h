#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "inferx/core/status.h"

namespace inferx::server::config {

struct ServerConfig {
  std::string listen_address = "0.0.0.0";
  uint16_t port = 8000;
  size_t io_threads = 1;
  size_t coroutine_threads = 4;
  size_t max_request_body_bytes = 1u << 20;
  std::chrono::seconds header_timeout{10};
  std::chrono::seconds request_timeout{300};
  std::chrono::seconds shutdown_grace{30};

  bool admin_enabled = false;
  std::string admin_listen_address = "127.0.0.1";
  uint16_t admin_port = 8001;
  bool development_auth_bypass = false;

  // Reloadable policy fields.
  uint32_t max_active_requests = 1024;
  uint32_t max_active_per_tenant = 128;
  uint64_t max_reserved_tokens = 1u << 20;
  size_t max_completed_snapshots = 10000;
  std::chrono::seconds completed_retention{300};
};

Status Validate(const ServerConfig& config);

class ConfigStore {
 public:
  static StatusOr<std::shared_ptr<ConfigStore>> Create(ServerConfig config);

  std::shared_ptr<const ServerConfig> Snapshot() const;
  uint64_t revision() const { return revision_.load(); }

  /// Publishes reloadable policy fields atomically. Listener topology,
  /// executors, transport limits, and authentication mode remain immutable.
  Status Reload(const ServerConfig& candidate);

 private:
  explicit ConfigStore(ServerConfig config);

  std::atomic<std::shared_ptr<const ServerConfig>> config_;
  std::atomic<uint64_t> revision_{1};
};

}  // namespace inferx::server::config
