#pragma once

#include <boost/asio/io_context.hpp>
#include <boost/asio/strand.hpp>
#include <folly/Executor.h>

#include <cstddef>
#include <memory>

#include "inferx/core/status.h"

namespace inferx::server::transport {

struct IoRuntimeConfig {
  size_t io_shards = 1;
  size_t threads_per_shard = 1;
  size_t coroutine_threads = 1;
};

/// Owns socket I/O shards and the executor on which application coroutines run.
class IoRuntime {
 public:
  static StatusOr<std::unique_ptr<IoRuntime>> Create(IoRuntimeConfig config);
  ~IoRuntime();

  IoRuntime(const IoRuntime&) = delete;
  IoRuntime& operator=(const IoRuntime&) = delete;

  Status Start();
  void Stop();
  void Join();

  boost::asio::io_context& accept_context();
  boost::asio::io_context& NextContext();
  folly::Executor::KeepAlive<> coroutine_executor() const;

 private:
  struct Impl;
  explicit IoRuntime(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server::transport
