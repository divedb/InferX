#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::server::gateway {

struct GatewayServerConfig {
  std::string host = "127.0.0.1";
  int port = 8000;
  std::string scheduler_endpoint;
  std::string tokenizer_directory;
  std::string model_id;
  std::string model_version = "loaded";
  std::string tokenizer_revision;
  size_t max_request_bytes = 8u * 1024 * 1024;
  size_t max_active_requests = 1024;
  size_t io_threads = 0;
  size_t application_threads = 0;
  int read_timeout_seconds = 30;
  int write_timeout_seconds = 600;
  int request_timeout_seconds = 600;
  int scheduler_connect_timeout_seconds = 5;
  std::vector<std::string> api_key_sha256;
};

Status ValidateGatewayServerConfig(const GatewayServerConfig& config);

/// Host-only HTTP gateway. It owns no Engine, CUDA context, scheduler queue,
/// model weights, or KV cache; all inference work crosses the scheduler RPC.
class GatewayServer {
 public:
  static StatusOr<std::unique_ptr<GatewayServer>> Create(
      const GatewayServerConfig& config);
  ~GatewayServer();

  GatewayServer(const GatewayServer&) = delete;
  GatewayServer& operator=(const GatewayServer&) = delete;

  Status Listen();
  void Stop();
  bool WaitUntilReady();
  int port() const;

 private:
  struct Impl;
  explicit GatewayServer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server::gateway
