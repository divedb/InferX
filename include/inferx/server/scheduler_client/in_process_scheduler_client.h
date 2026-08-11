#pragma once
#include <folly/coro/Task.h>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "inferx/server/scheduler_client/scheduler_client.h"

namespace inferx::server::scheduler_client {
struct InProcessGenerationEvent {
  std::string text;
  bool terminal = false;
  FinishReason finish_reason = FinishReason::kNone;
  uint32_t generated_tokens = 0;
  /// The step's sampled token (-1 on the terminal event) and, when the
  /// request asked, its logprobs.
  int32_t token = -1;
  bool has_logprob = false;
  TokenLogprobs logprobs;
};
class InProcessGeneration {
 public:
  virtual ~InProcessGeneration() = default;
  virtual folly::coro::Task<StatusOr<std::optional<InProcessGenerationEvent>>>
  Next(folly::CancellationToken cancellation) = 0;
  virtual void Cancel() = 0;
  virtual uint32_t prompt_tokens() const = 0;
};
class InProcessEngineBackend {
 public:
  virtual ~InProcessEngineBackend() = default;
  virtual StatusOr<std::shared_ptr<InProcessGeneration>> Submit(
      const ScheduledRequest& request) = 0;
};
class InProcessSchedulerClient final : public SchedulerClient {
 public:
  explicit InProcessSchedulerClient(InProcessEngineBackend* backend)
      : backend_(backend) {}
  folly::coro::Task<StatusOr<SubmitResult>> Submit(
      ScheduledRequest request, folly::CancellationToken cancellation) override;
  folly::coro::AsyncGenerator<GenerationEvent&&> Events(
      request::RequestId request_id,
      folly::CancellationToken cancellation) override;
  folly::coro::Task<Status> Cancel(request::RequestId request_id,
                                   request::CancellationReason reason) override;
  folly::coro::Task<StatusOr<request::RequestSnapshot>> GetStatus(
      request::RequestId request_id,
      folly::CancellationToken cancellation) override;
  folly::coro::Task<Status> UpdatePriority(
      request::RequestId request_id, PriorityClass priority,
      folly::CancellationToken cancellation) override;

 private:
  struct ActiveGeneration {
    std::shared_ptr<InProcessGeneration> generation;
    uint32_t attempt = 0;
  };

  InProcessEngineBackend* backend_;
  std::mutex mutex_;
  std::unordered_map<request::RequestId, ActiveGeneration> generations_;
};
}  // namespace inferx::server::scheduler_client
