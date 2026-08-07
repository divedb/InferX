#pragma once
#include <folly/Executor.h>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "inferx/server/scheduler_client/scheduler_client.h"
namespace inferx::server::scheduler_client {
struct LegacyGenerationEvent {
  std::string text;
  bool terminal = false;
  FinishReason finish_reason = FinishReason::kNone;
  uint32_t generated_tokens = 0;
};
class LegacyGeneration {
 public:
  virtual ~LegacyGeneration() = default;
  virtual bool Next(LegacyGenerationEvent* event) = 0;
  virtual void Cancel() = 0;
  virtual uint32_t prompt_tokens() const = 0;
};
class LegacyEngineBackend {
 public:
  virtual ~LegacyEngineBackend() = default;
  virtual StatusOr<std::shared_ptr<LegacyGeneration>> Submit(
      const ScheduledRequest& request) = 0;
};
class InProcessSchedulerClient final : public SchedulerClient {
 public:
  InProcessSchedulerClient(LegacyEngineBackend* backend,
                           folly::Executor* blocking_executor)
      : backend_(backend), blocking_executor_(blocking_executor) {}
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
  LegacyEngineBackend* backend_;
  folly::Executor* blocking_executor_;
  std::mutex mutex_;
  std::unordered_map<request::RequestId, std::shared_ptr<LegacyGeneration>> generations_;
};
}  // namespace inferx::server::scheduler_client
