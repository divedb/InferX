#include "inferx/server/scheduler_client/in_process_scheduler_client.h"
#include <folly/coro/Baton.h>
#include <folly/coro/CurrentExecutor.h>
#include <folly/coro/ViaIfAsync.h>
#include <exception>
#include <optional>
namespace inferx::server::scheduler_client {
namespace {
struct NextState {
  folly::coro::Baton ready;
  LegacyGenerationEvent event;
  bool has_event = false;
  std::exception_ptr exception;
};
folly::coro::Task<StatusOr<std::optional<LegacyGenerationEvent>>> NextOn(
    folly::Executor* blocking_executor,
    std::shared_ptr<LegacyGeneration> generation) {
  if (blocking_executor == nullptr) {
    co_return FailedPreconditionError("blocking executor is not configured");
  }
  auto state = std::make_shared<NextState>();
  auto* resume_executor = co_await folly::coro::co_current_executor;
  blocking_executor->add([state, generation = std::move(generation)] {
    try { state->has_event = generation->Next(&state->event); }
    catch (...) { state->exception = std::current_exception(); }
    state->ready.post();
  });
  co_await folly::coro::co_viaIfAsync(resume_executor, state->ready);
  if (state->exception) {
    try { std::rethrow_exception(state->exception); }
    catch (const std::exception& error) {
      co_return InternalError("legacy generation failed: ", error.what());
    } catch (...) { co_return InternalError("legacy generation failed"); }
  }
  if (!state->has_event) co_return std::optional<LegacyGenerationEvent>{};
  co_return std::optional<LegacyGenerationEvent>{std::move(state->event)};
}
}  // namespace
folly::coro::Task<StatusOr<SubmitResult>> InProcessSchedulerClient::Submit(
    ScheduledRequest request, folly::CancellationToken cancellation) {
  if (backend_ == nullptr) co_return InternalError("engine backend is null");
  if (request.workload != WorkloadClass::kGeneration) {
    co_return UnimplementedError("in-process backend does not support embeddings");
  }
  if (cancellation.isCancellationRequested()) {
    co_return absl::CancelledError("submission cancelled");
  }
  auto generation = backend_->Submit(request);
  if (!generation.ok()) co_return generation.status();
  {
    std::lock_guard lock(mutex_);
    if (generations_.contains(request.request_id)) {
      co_return FailedPreconditionError("request already submitted");
    }
    generations_.emplace(
        request.request_id,
        ActiveGeneration{.generation = *generation, .attempt = request.attempt});
  }
  co_return SubmitResult{request.request_id, request.attempt};
}
folly::coro::AsyncGenerator<GenerationEvent&&> InProcessSchedulerClient::Events(
    request::RequestId request_id, folly::CancellationToken cancellation) {
  std::shared_ptr<LegacyGeneration> generation;
  uint32_t attempt = 0;
  {
    std::lock_guard lock(mutex_);
    const auto found = generations_.find(request_id);
    if (found == generations_.end()) co_return;
    generation = found->second.generation;
    attempt = found->second.attempt;
  }
  uint64_t sequence = 1;
  while (!cancellation.isCancellationRequested()) {
    auto next = co_await NextOn(blocking_executor_, generation);
    if (!next.ok()) {
      GenerationEvent failed;
      failed.request_id = request_id;
      failed.attempt = attempt;
      failed.sequence_number = sequence;
      failed.terminal = true;
      failed.finish_reason = FinishReason::kFailed;
      failed.error = next.status();
      co_yield std::move(failed);
      break;
    }
    if (!next->has_value()) break;
    GenerationEvent event;
    event.request_id = request_id;
    event.attempt = attempt;
    event.sequence_number = sequence++;
    event.text_delta = std::move((*next)->text);
    event.generated_tokens = (*next)->generated_tokens;
    event.terminal = (*next)->terminal;
    event.finish_reason = (*next)->finish_reason;
    event.usage = {.prompt_tokens = generation->prompt_tokens(),
                   .completion_tokens = (*next)->generated_tokens};
    const bool terminal = event.terminal;
    co_yield std::move(event);
    if (terminal) break;
  }
  std::lock_guard lock(mutex_);
  generations_.erase(request_id);
}
folly::coro::Task<Status> InProcessSchedulerClient::Cancel(
    request::RequestId request_id, request::CancellationReason reason) {
  (void)reason;
  std::shared_ptr<LegacyGeneration> generation;
  {
    std::lock_guard lock(mutex_);
    const auto found = generations_.find(request_id);
    if (found == generations_.end()) co_return OkStatus();
    generation = found->second.generation;
  }
  generation->Cancel();
  co_return OkStatus();
}
folly::coro::Task<StatusOr<request::RequestSnapshot>>
InProcessSchedulerClient::GetStatus(request::RequestId, folly::CancellationToken) {
  co_return UnimplementedError("status is owned by RequestManager");
}
folly::coro::Task<Status> InProcessSchedulerClient::UpdatePriority(
    request::RequestId, PriorityClass, folly::CancellationToken) {
  co_return UnimplementedError("in-process priority updates are unsupported");
}
}  // namespace inferx::server::scheduler_client
