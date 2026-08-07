#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "inferx/core/status.h"
#include "inferx/server/transport/io_runtime.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::transport {

struct BeastListenerConfig {
  std::string host = "127.0.0.1";
  int port = 0;
  size_t max_request_bytes = 8u * 1024 * 1024;
  int header_read_timeout_seconds = 10;
  int body_read_timeout_seconds = 30;
  int write_timeout_seconds = 600;
  int keep_alive_timeout_seconds = 60;
};

/// Accepts connections and creates one serialized HTTP/1.1 session per socket.
class BeastListener {
 public:
  static StatusOr<std::unique_ptr<BeastListener>> Create(
      IoRuntime* runtime, BeastListenerConfig config,
      std::shared_ptr<RequestHandler> handler);
  ~BeastListener();

  BeastListener(const BeastListener&) = delete;
  BeastListener& operator=(const BeastListener&) = delete;

  Status Start();
  void Stop();
  int port() const;

 private:
  struct Impl;
  explicit BeastListener(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

}  // namespace inferx::server::transport
