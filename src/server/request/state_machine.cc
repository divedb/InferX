#include "inferx/server/request/request_context.h"
#include "inferx/server/request/request_manager.h"

namespace inferx::server::request {

const char* RequestStateName(RequestState state) {
  switch (state) {
    case RequestState::kReceived: return "received";
    case RequestState::kValidating: return "validating";
    case RequestState::kAuthenticated: return "authenticated";
    case RequestState::kAdmissionPending: return "admission_pending";
    case RequestState::kQueued: return "queued";
    case RequestState::kPrefilling: return "prefilling";
    case RequestState::kDecoding: return "decoding";
    case RequestState::kFinishing: return "finishing";
    case RequestState::kCompleted: return "completed";
    case RequestState::kRejected: return "rejected";
    case RequestState::kTimedOut: return "timed_out";
    case RequestState::kCancelled: return "cancelled";
    case RequestState::kFailed: return "failed";
  }
  return "unknown";
}

bool IsTerminal(RequestState state) {
  return state == RequestState::kCompleted || state == RequestState::kRejected ||
         state == RequestState::kTimedOut || state == RequestState::kCancelled ||
         state == RequestState::kFailed;
}

bool CanTransition(RequestState from, RequestState to) {
  if (IsTerminal(from)) return false;
  if (to == RequestState::kRejected || to == RequestState::kTimedOut ||
      to == RequestState::kCancelled || to == RequestState::kFailed) {
    return true;
  }
  switch (from) {
    case RequestState::kReceived: return to == RequestState::kValidating;
    case RequestState::kValidating: return to == RequestState::kAuthenticated;
    case RequestState::kAuthenticated:
      return to == RequestState::kAdmissionPending;
    case RequestState::kAdmissionPending: return to == RequestState::kQueued;
    case RequestState::kQueued: return to == RequestState::kPrefilling;
    case RequestState::kPrefilling: return to == RequestState::kDecoding;
    case RequestState::kDecoding:
      return to == RequestState::kDecoding || to == RequestState::kFinishing;
    case RequestState::kFinishing: return to == RequestState::kCompleted;
    default: return false;
  }
}

bool RequestContext::TransitionTo(RequestState next) {
  if (!CanTransition(state, next)) return false;
  state = next;
  if (next == RequestState::kCompleted || IsTerminal(next)) {
    completed_at = std::chrono::steady_clock::now();
  }
  return true;
}

void RequestContext::Cancel(CancellationReason reason) {
  if (IsTerminal(state)) return;
  reason_ = reason;
  (void)TransitionTo(reason == CancellationReason::kDeadline
                         ? RequestState::kTimedOut
                         : RequestState::kCancelled);
  cancellation.requestCancellation();
}

StatusOr<std::shared_ptr<RequestContext>> RequestManager::Create(
    RequestContext context) {
  if (context.request_id.empty()) {
    return InvalidArgumentError("request ID must not be empty");
  }
  auto owned = std::make_shared<RequestContext>(std::move(context));
  std::lock_guard lock(mutex_);
  if (active_.contains(owned->request_id)) {
    return FailedPreconditionError("request ID already exists: ",
                                   owned->request_id);
  }
  active_.emplace(owned->request_id, owned);
  return owned;
}

Status RequestManager::Cancel(const RequestId& id, CancellationReason reason) {
  std::shared_ptr<RequestContext> context;
  {
    std::lock_guard lock(mutex_);
    const auto it = active_.find(id);
    if (it == active_.end()) return NotFoundError("request not found: ", id);
    context = it->second;
  }
  context->Cancel(reason);
  return OkStatus();
}

RequestSnapshot RequestManager::Snapshot(const RequestContext& context) {
  return RequestSnapshot{context.request_id, context.state,
                         context.cancellation_reason(), context.received_at,
                         context.deadline, context.completed_at};
}

StatusOr<RequestSnapshot> RequestManager::GetStatus(const RequestId& id) const {
  std::lock_guard lock(mutex_);
  const auto it = active_.find(id);
  if (it == active_.end()) return NotFoundError("request not found: ", id);
  return Snapshot(*it->second);
}

Status RequestManager::Finalize(const RequestId& id) {
  std::lock_guard lock(mutex_);
  const auto it = active_.find(id);
  if (it == active_.end()) return OkStatus();
  if (!IsTerminal(it->second->state)) {
    return FailedPreconditionError("request is not terminal: ", id);
  }
  active_.erase(it);
  return OkStatus();
}

size_t RequestManager::active_count() const {
  std::lock_guard lock(mutex_);
  return active_.size();
}

}  // namespace inferx::server::request
