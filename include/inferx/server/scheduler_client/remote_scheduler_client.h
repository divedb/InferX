#pragma once

#include <folly/Executor.h>

#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>

#include "inferx/server/scheduler_client/scheduler_client.h"

namespace inferx::server::scheduler_client {

struct RemoteRequestStatus {
  bool found = false;
  uint32_t attempt = 0;
  request::RequestSnapshot snapshot;
};

class RemoteEventStream {
 public:
  virtual ~RemoteEventStream() = default;
  virtual StatusOr<std::optional<GenerationEvent>> Next() = 0;
  virtual void Cancel() = 0;
};

/// Blocking RPC seam implemented by the generated gRPC transport. Keeping it
/// separate makes retry/idempotency policy independently testable.
class RemoteSchedulerTransport {
 public:
  virtual ~RemoteSchedulerTransport() = default;
  virtual StatusOr<SubmitResult> Submit(const ScheduledRequest& request) = 0;
  virtual StatusOr<RemoteRequestStatus> GetStatus(
      const request::RequestId& request_id, uint32_t attempt) = 0;
  virtual Status Cancel(const request::RequestId& request_id, uint32_t attempt,
                        request::CancellationReason reason) = 0;
  virtual Status UpdatePriority(const request::RequestId& request_id,
                                uint32_t attempt,
                                PriorityClass priority) = 0;
  virtual StatusOr<std::unique_ptr<RemoteEventStream>> Subscribe(
      const request::RequestId& request_id, uint32_t attempt,
      uint64_t after_sequence_number) = 0;
  virtual void Shutdown() = 0;
};

struct RemoteSchedulerClientConfig {
  uint32_t max_attempts = 4;
  std::chrono::milliseconds initial_backoff{25};
  std::chrono::milliseconds max_backoff{1000};
  std::function<void(std::chrono::milliseconds)> sleep =
      [](std::chrono::milliseconds delay) { std::this_thread::sleep_for(delay); };
};

class RemoteSchedulerClient final : public SchedulerClient {
 public:
  RemoteSchedulerClient(std::shared_ptr<RemoteSchedulerTransport> transport,
                        folly::Executor* blocking_executor,
                        RemoteSchedulerClientConfig config = {});
  ~RemoteSchedulerClient() override;

  folly::coro::Task<StatusOr<SubmitResult>> Submit(
      ScheduledRequest request, folly::CancellationToken cancellation) override;
  folly::coro::AsyncGenerator<GenerationEvent&&> Events(
      request::RequestId request_id,
      folly::CancellationToken cancellation) override;
  folly::coro::Task<Status> Cancel(
      request::RequestId request_id,
      request::CancellationReason reason) override;
  folly::coro::Task<StatusOr<request::RequestSnapshot>> GetStatus(
      request::RequestId request_id,
      folly::CancellationToken cancellation) override;
  folly::coro::Task<Status> UpdatePriority(
      request::RequestId request_id, PriorityClass priority,
      folly::CancellationToken cancellation) override;

 private:
  std::optional<uint32_t> AttemptFor(const request::RequestId& id) const;

  std::shared_ptr<RemoteSchedulerTransport> transport_;
  folly::Executor* blocking_executor_;
  RemoteSchedulerClientConfig config_;
  mutable std::mutex mutex_;
  std::unordered_map<request::RequestId, uint32_t> attempts_;
};

}  // namespace inferx::server::scheduler_client
