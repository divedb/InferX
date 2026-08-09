#include "inferx/server/request/managed_request_service.h"

#include <algorithm>
#include <limits>

#include "inferx/support/log.h"

namespace inferx::server::request {

folly::coro::Task<StatusOr<scheduler_client::SubmitResult>>
ManagedRequestService::Submit(scheduler_client::ScheduledRequest request,
                              folly::CancellationToken cancellation) {
  if (manager_ == nullptr || scheduler_ == nullptr) {
    co_return InternalError("managed request service is not configured");
  }
  if (cancellation.isCancellationRequested()) {
    co_return absl::CancelledError("request cancelled before submission");
  }
  const uint64_t reserved_tokens = request.prompt_tokens.size() +
                                   std::max(request.sampling.max_tokens, 0);
  if (reserved_tokens > std::numeric_limits<uint32_t>::max()) {
    co_return InvalidArgumentError("token reservation exceeds supported range");
  }
  uint64_t reservation = 0;
  if (admission_ != nullptr) {
    auto decision = admission_->TryAdmit(
        {.tenant_id = request.tenant_id,
         .reserved_tokens = static_cast<uint32_t>(reserved_tokens)});
    if (!decision.admitted()) {
      LOG(WARNING) << Rid(request.request_id) << "admission rejected: "
                   << (decision.message.empty() ? "unspecified reason"
                                                : decision.message);
      co_return ResourceExhaustedError(
          decision.message.empty() ? "request admission rejected"
                                   : decision.message);
    }
    reservation = decision.reservation_id;
  }

  RequestContext initial;
  initial.request_id = request.request_id;
  initial.tenant_id = request.tenant_id;
  initial.resolved_version = request.model_version;
  initial.received_at = std::chrono::steady_clock::now();
  initial.deadline = request.deadline;
  if (!initial.TransitionTo(RequestState::kValidating) ||
      !initial.TransitionTo(RequestState::kAuthenticated) ||
      !initial.TransitionTo(RequestState::kAdmissionPending)) {
    if (admission_ != nullptr) admission_->Release(reservation);
    co_return InternalError("failed to initialize request lifecycle");
  }
  auto created = manager_->Create(std::move(initial));
  if (!created.ok()) {
    if (admission_ != nullptr) admission_->Release(reservation);
    co_return created.status();
  }
  {
    std::lock_guard lock(mutex_);
    active_.emplace(request.request_id,
                    ActiveRequest{*created, reservation});
  }
  auto submitted = co_await scheduler_->Submit(std::move(request), cancellation);
  if (!submitted.ok()) {
    LOG(WARNING) << Rid((*created)->request_id)
                 << "scheduler submission failed: " << submitted.status();
    (void)Finish((*created)->request_id, RequestState::kRejected);
    co_return submitted.status();
  }
  if (!(*created)->TransitionTo(RequestState::kQueued)) {
    (void)co_await scheduler_->Cancel(
        submitted->request_id, CancellationReason::kInternal);
    (void)Finish(submitted->request_id, RequestState::kFailed);
    co_return InternalError("failed to queue request lifecycle");
  }
  VLOG(1) << Rid(submitted->request_id) << "request queued";
  co_return *submitted;
}

folly::coro::AsyncGenerator<scheduler_client::GenerationEvent&&>
ManagedRequestService::Events(RequestId request_id,
                              folly::CancellationToken cancellation) {
  if (scheduler_ == nullptr) co_return;
  auto context = FindContext(request_id);
  if (context == nullptr) co_return;
  auto upstream = scheduler_->Events(request_id, cancellation);
  bool started = false;
  while (auto next = co_await upstream.next()) {
    auto event = std::move(*next);
    if (!started) {
      if (context->state == RequestState::kQueued) {
        (void)context->TransitionTo(RequestState::kPrefilling);
      }
      if (context->state == RequestState::kPrefilling) {
        (void)context->TransitionTo(RequestState::kDecoding);
      }
      started = true;
      VLOG(1) << Rid(request_id) << "request decoding started";
    }
    if (!event.error.ok()) {
      LOG(ERROR) << Rid(request_id) << "generation failed: " << event.error;
      (void)Finish(request_id, RequestState::kFailed);
      co_yield std::move(event);
      co_return;
    }
    if (event.terminal) {
      if (context->state == RequestState::kDecoding) {
        (void)context->TransitionTo(RequestState::kFinishing);
      }
      (void)Finish(request_id, RequestState::kCompleted);
      VLOG(1) << Rid(request_id) << "request completed";
      co_yield std::move(event);
      co_return;
    }
    co_yield std::move(event);
  }
  (void)Finish(request_id, RequestState::kFailed);
}

folly::coro::Task<Status> ManagedRequestService::Cancel(
    RequestId request_id, CancellationReason reason) {
  if (manager_ == nullptr || scheduler_ == nullptr) {
    co_return InternalError("managed request service is not configured");
  }
  const Status local = manager_->Cancel(request_id, reason);
  if (!local.ok() && local.code() != absl::StatusCode::kNotFound) {
    co_return local;
  }
  auto context = FindContext(request_id);
  if (context == nullptr) co_return OkStatus();
  const Status remote = co_await scheduler_->Cancel(request_id, reason);
  (void)Finish(request_id, context->state);
  co_return remote;
}

Status ManagedRequestService::Finish(const RequestId& request_id,
                                     RequestState terminal) {
  ActiveRequest active;
  {
    std::lock_guard lock(mutex_);
    const auto found = active_.find(request_id);
    if (found == active_.end()) return OkStatus();
    active = std::move(found->second);
    active_.erase(found);
  }
  if (!IsTerminal(active.context->state)) {
    (void)active.context->TransitionTo(terminal);
  }
  if (admission_ != nullptr && active.admission_reservation != 0) {
    admission_->Release(active.admission_reservation);
  }
  return manager_->Finalize(request_id);
}

std::shared_ptr<RequestContext> ManagedRequestService::FindContext(
    const RequestId& request_id) {
  std::lock_guard lock(mutex_);
  const auto found = active_.find(request_id);
  return found == active_.end() ? nullptr : found->second.context;
}

}  // namespace inferx::server::request
