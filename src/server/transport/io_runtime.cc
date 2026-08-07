#include "inferx/server/transport/io_runtime.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <atomic>
#include <thread>
#include <utility>
#include <vector>

namespace inferx::server::transport {

struct IoRuntime::Impl {
  struct Shard {
    boost::asio::io_context context;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type>
        work{context.get_executor()};
  };

  explicit Impl(IoRuntimeConfig value)
      : config(value), coroutine_pool(value.coroutine_threads) {
    shards.reserve(config.io_shards);
    for (size_t i = 0; i < config.io_shards; ++i) {
      shards.push_back(std::make_unique<Shard>());
    }
  }

  IoRuntimeConfig config;
  std::vector<std::unique_ptr<Shard>> shards;
  folly::CPUThreadPoolExecutor coroutine_pool;
  std::vector<std::thread> threads;
  std::atomic<size_t> next{0};
  std::atomic<bool> started{false};
  std::atomic<bool> stopped{false};
};

IoRuntime::IoRuntime(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

StatusOr<std::unique_ptr<IoRuntime>> IoRuntime::Create(IoRuntimeConfig config) {
  if (config.io_shards == 0) {
    return InvalidArgumentError("io_shards must be positive");
  }
  if (config.threads_per_shard == 0) {
    return InvalidArgumentError("threads_per_shard must be positive");
  }
  if (config.coroutine_threads == 0) {
    return InvalidArgumentError("coroutine_threads must be positive");
  }
  return std::unique_ptr<IoRuntime>(
      new IoRuntime(std::make_unique<Impl>(config)));
}

IoRuntime::~IoRuntime() {
  Stop();
  Join();
}

Status IoRuntime::Start() {
  if (impl_->stopped.load(std::memory_order_acquire)) {
    return FailedPreconditionError("I/O runtime has already stopped");
  }
  if (impl_->started.exchange(true, std::memory_order_acq_rel)) {
    return FailedPreconditionError("I/O runtime already started");
  }
  impl_->threads.reserve(impl_->config.io_shards *
                         impl_->config.threads_per_shard);
  for (const auto& shard : impl_->shards) {
    for (size_t i = 0; i < impl_->config.threads_per_shard; ++i) {
      impl_->threads.emplace_back([context = &shard->context] {
        context->run();
      });
    }
  }
  return OkStatus();
}

void IoRuntime::Stop() {
  if (impl_->stopped.exchange(true, std::memory_order_acq_rel)) return;
  for (const auto& shard : impl_->shards) {
    shard->work.reset();
  }
}

void IoRuntime::Join() {
  for (std::thread& thread : impl_->threads) {
    if (thread.joinable()) thread.join();
  }
  impl_->threads.clear();
  impl_->coroutine_pool.stop();
  impl_->coroutine_pool.join();
}

boost::asio::io_context& IoRuntime::accept_context() {
  return impl_->shards.front()->context;
}

boost::asio::io_context& IoRuntime::NextContext() {
  const size_t index =
      impl_->next.fetch_add(1, std::memory_order_relaxed) % impl_->shards.size();
  return impl_->shards[index]->context;
}

folly::Executor::KeepAlive<> IoRuntime::coroutine_executor() const {
  return folly::getKeepAliveToken(impl_->coroutine_pool);
}

}  // namespace inferx::server::transport
