#include "inferx/server/scheduler_client/remote_scheduler_client.h"

#include <folly/coro/Baton.h>
#include <folly/coro/CurrentExecutor.h>
#include <folly/coro/ViaIfAsync.h>

#include <algorithm>
#include <exception>
#include <type_traits>
#include <utility>

namespace inferx::server::scheduler_client {
namespace {

bool Retryable(const Status& status) {
  return status.code() == absl::StatusCode::kUnavailable ||
         status.code() == absl::StatusCode::kDeadlineExceeded ||
         status.code() == absl::StatusCode::kUnknown ||
         status.code() == absl::StatusCode::kAborted;
}

std::chrono::milliseconds NextBackoff(std::chrono::milliseconds current,
                                      std::chrono::milliseconds maximum) {
  return std::min(maximum, current * 2);
}

template <typename Result>
struct BlockingState {
  folly::coro::Baton ready;
  std::optional<Result> result;
  std::exception_ptr exception;
};

template <typename Function>
folly::coro::Task<std::invoke_result_t<Function>> RunBlocking(
    folly::Executor* executor, Function function) {
  using Result = std::invoke_result_t<Function>;
  if (executor == nullptr) {
    if constexpr (std::is_same_v<Result, Status>) {
      co_return FailedPreconditionError("blocking executor is not configured");
    } else {
      co_return FailedPreconditionError("blocking executor is not configured");
    }
  }
  auto state = std::make_shared<BlockingState<Result>>();
  auto* resume_executor = co_await folly::coro::co_current_executor;
  executor->add([state, function = std::move(function)]() mutable {
    try {
      state->result.emplace(function());
    } catch (...) {
      state->exception = std::current_exception();
    }
    state->ready.post();
  });
  co_await folly::coro::co_viaIfAsync(resume_executor, state->ready);
  if (state->exception) {
    try {
      std::rethrow_exception(state->exception);
    } catch (const std::exception& error) {
      if constexpr (std::is_same_v<Result, Status>) {
        co_return InternalError("remote scheduler transport failed: ",
                                error.what());
      } else {
        co_return InternalError("remote scheduler transport failed: ",
                                error.what());
      }
    } catch (...) {
      if constexpr (std::is_same_v<Result, Status>) {
        co_return InternalError("remote scheduler transport failed");
      } else {
        co_return InternalError("remote scheduler transport failed");
      }
    }
  }
  co_return std::move(*state->result);
}

}  // namespace

RemoteSchedulerClient::RemoteSchedulerClient(
    std::shared_ptr<RemoteSchedulerTransport> transport,
    folly::Executor* blocking_executor, RemoteSchedulerClientConfig config)
    : transport_(std::move(transport)),
      blocking_executor_(blocking_executor),
      config_(std::move(config)) {
  config_.max_attempts = std::max<uint32_t>(1, config_.max_attempts);
  config_.initial_backoff =
      std::max(config_.initial_backoff, std::chrono::milliseconds::zero());
  config_.max_backoff = std::max(config_.initial_backoff, config_.max_backoff);
}

RemoteSchedulerClient::~RemoteSchedulerClient() {
  if (transport_ != nullptr) transport_->Shutdown();
}

folly::coro::Task<StatusOr<SubmitResult>> RemoteSchedulerClient::Submit(
    ScheduledRequest request, folly::CancellationToken cancellation) {
  if (transport_ == nullptr) {
    co_return FailedPreconditionError("remote transport is not configured");
  }
  if (request.request_id.empty()) {
    co_return InvalidArgumentError("request ID is required");
  }
  if (cancellation.isCancellationRequested()) {
    co_return absl::CancelledError("submission cancelled");
  }
  auto transport = transport_;
  const auto config = config_;
  auto submitted = co_await RunBlocking(
      blocking_executor_, [transport, request, config]() mutable
          -> StatusOr<SubmitResult> {
        Status last = absl::UnavailableError("scheduler unavailable");
        auto backoff = config.initial_backoff;
        for (uint32_t try_number = 0; try_number < config.max_attempts;
             ++try_number) {
          if (request.deadline != std::chrono::steady_clock::time_point{} &&
              std::chrono::steady_clock::now() >= request.deadline) {
            return absl::DeadlineExceededError(
                "scheduler submission deadline expired");
          }
          auto result = transport->Submit(request);
          if (result.ok()) return *result;
          last = result.status();
          if (!Retryable(last)) return last;

          auto status = transport->GetStatus(request.request_id,
                                             request.attempt);
          if (status.ok() && status->found &&
              status->attempt == request.attempt) {
            return SubmitResult{request.request_id, request.attempt};
          }
          if (try_number + 1 < config.max_attempts && config.sleep) {
            config.sleep(backoff);
            backoff = NextBackoff(backoff, config.max_backoff);
          }
        }
        return last;
      });
  if (!submitted.ok()) co_return submitted.status();
  if (cancellation.isCancellationRequested()) {
    (void)co_await RunBlocking(
        blocking_executor_, [transport = transport_, id = request.request_id,
                             attempt = request.attempt] {
          return transport->Cancel(
              id, attempt,
              request::CancellationReason::kClientDisconnected);
        });
    co_return absl::CancelledError("submission cancelled");
  }
  {
    std::lock_guard lock(mutex_);
    attempts_[submitted->request_id] = submitted->attempt;
  }
  co_return *submitted;
}

folly::coro::AsyncGenerator<GenerationEvent&&>
RemoteSchedulerClient::Events(request::RequestId request_id,
                              folly::CancellationToken cancellation) {
  const auto attempt = AttemptFor(request_id);
  if (!attempt.has_value() || transport_ == nullptr) co_return;
  uint64_t sequence = 0;
  uint32_t reconnects = 0;
  auto backoff = config_.initial_backoff;
  std::unique_ptr<RemoteEventStream> stream;
  while (!cancellation.isCancellationRequested()) {
    if (stream == nullptr) {
      auto opened = co_await RunBlocking(
          blocking_executor_, [transport = transport_, request_id, attempt,
                               sequence] {
            return transport->Subscribe(request_id, *attempt, sequence);
          });
      if (!opened.ok()) {
        if (!Retryable(opened.status()) ||
            ++reconnects >= config_.max_attempts) {
          GenerationEvent failed{.request_id = request_id,
                                 .attempt = *attempt,
                                 .sequence_number = sequence + 1,
                                 .terminal = true,
                                 .finish_reason = FinishReason::kFailed,
                                 .error = opened.status()};
          co_yield std::move(failed);
          co_return;
        }
        if (config_.sleep) {
          const auto delay = backoff;
          const Status slept = co_await RunBlocking(
              blocking_executor_, [sleep = config_.sleep, delay] {
                sleep(delay);
                return OkStatus();
              });
          if (!slept.ok()) co_return;
        }
        backoff = NextBackoff(backoff, config_.max_backoff);
        continue;
      }
      stream = std::move(*opened);
    }

    auto next = co_await RunBlocking(blocking_executor_,
                                     [current = stream.get()] {
                                       return current->Next();
                                     });
    if (!next.ok()) {
      stream->Cancel();
      stream.reset();
      if (Retryable(next.status()) &&
          ++reconnects < config_.max_attempts) {
        if (config_.sleep) {
          const auto delay = backoff;
          const Status slept = co_await RunBlocking(
              blocking_executor_, [sleep = config_.sleep, delay] {
                sleep(delay);
                return OkStatus();
              });
          if (!slept.ok()) co_return;
        }
        backoff = NextBackoff(backoff, config_.max_backoff);
        continue;
      }
      GenerationEvent failed{.request_id = request_id,
                             .attempt = *attempt,
                             .sequence_number = sequence + 1,
                             .terminal = true,
                             .finish_reason = FinishReason::kFailed,
                             .error = next.status()};
      co_yield std::move(failed);
      co_return;
    }
    if (!next->has_value()) co_return;
    GenerationEvent event = std::move(**next);
    if (event.request_id != request_id || event.attempt != *attempt ||
        event.sequence_number != sequence + 1) {
      event.error =
          absl::DataLossError("scheduler event identity or sequence gap");
      event.terminal = true;
      event.finish_reason = FinishReason::kFailed;
      co_yield std::move(event);
      co_return;
    }
    sequence = event.sequence_number;
    const bool terminal = event.terminal;
    co_yield std::move(event);
    if (terminal) {
      std::lock_guard lock(mutex_);
      attempts_.erase(request_id);
      co_return;
    }
  }
  if (stream != nullptr) stream->Cancel();
}

folly::coro::Task<Status> RemoteSchedulerClient::Cancel(
    request::RequestId request_id, request::CancellationReason reason) {
  const auto attempt = AttemptFor(request_id);
  if (!attempt.has_value()) co_return OkStatus();
  auto result = co_await RunBlocking(
      blocking_executor_, [transport = transport_, request_id, attempt, reason] {
        return transport->Cancel(request_id, *attempt, reason);
      });
  if (result.ok()) {
    std::lock_guard lock(mutex_);
    attempts_.erase(request_id);
  }
  co_return result;
}

folly::coro::Task<StatusOr<request::RequestSnapshot>>
RemoteSchedulerClient::GetStatus(request::RequestId request_id,
                                 folly::CancellationToken cancellation) {
  if (cancellation.isCancellationRequested()) {
    co_return absl::CancelledError("status query cancelled");
  }
  const auto attempt = AttemptFor(request_id);
  if (!attempt.has_value()) co_return NotFoundError("request is not tracked");
  auto result = co_await RunBlocking(
      blocking_executor_, [transport = transport_, request_id, attempt] {
        return transport->GetStatus(request_id, *attempt);
      });
  if (!result.ok()) co_return result.status();
  if (!result->found || result->attempt != *attempt) {
    co_return NotFoundError("remote request is not found");
  }
  co_return result->snapshot;
}

folly::coro::Task<Status> RemoteSchedulerClient::UpdatePriority(
    request::RequestId request_id, PriorityClass priority,
    folly::CancellationToken cancellation) {
  if (cancellation.isCancellationRequested()) {
    co_return absl::CancelledError("priority update cancelled");
  }
  const auto attempt = AttemptFor(request_id);
  if (!attempt.has_value()) co_return NotFoundError("request is not tracked");
  co_return co_await RunBlocking(
      blocking_executor_, [transport = transport_, request_id, attempt,
                           priority] {
        return transport->UpdatePriority(request_id, *attempt, priority);
      });
}

std::optional<uint32_t> RemoteSchedulerClient::AttemptFor(
    const request::RequestId& id) const {
  std::lock_guard lock(mutex_);
  const auto found = attempts_.find(id);
  if (found == attempts_.end()) return std::nullopt;
  return found->second;
}

}  // namespace inferx::server::scheduler_client
