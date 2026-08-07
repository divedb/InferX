#include "inferx/server/request/request_context.h"
#include "inferx/server/request/request_manager.h"

namespace inferx::server::request {

RequestManager::RequestManager(RequestManagerConfig config)
    : config_(std::move(config)) {
  if (!config_.now) {
    config_.now = [] { return std::chrono::steady_clock::now(); };
  }
}

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
  PruneCompletedLocked(config_.now());
  const auto it = active_.find(id);
  if (it != active_.end()) return Snapshot(*it->second);
  const auto completed = completed_.find(id);
  if (completed != completed_.end()) return completed->second;
  return NotFoundError("request not found: ", id);
}

Status RequestManager::Finalize(const RequestId& id) {
  std::lock_guard lock(mutex_);
  const auto it = active_.find(id);
  if (it == active_.end()) return OkStatus();
  if (!IsTerminal(it->second->state)) {
    return FailedPreconditionError("request is not terminal: ", id);
  }
  const auto now = config_.now();
  RequestSnapshot snapshot = Snapshot(*it->second);
  if (snapshot.completed_at == std::chrono::steady_clock::time_point{}) {
    snapshot.completed_at = now;
  }
  completed_.insert_or_assign(id, std::move(snapshot));
  completion_order_.emplace_back(id, now);
  active_.erase(it);
  PruneCompletedLocked(now);
  return OkStatus();
}

void RequestManager::PruneCompletedLocked(
    std::chrono::steady_clock::time_point now) const {
  while (!completion_order_.empty()) {
    const auto found = completed_.find(completion_order_.front().first);
    if (found == completed_.end()) {
      completion_order_.pop_front();
      continue;
    }
    const bool over_count = completed_.size() > config_.max_completed_snapshots;
    const bool expired = config_.completed_retention.count() >= 0 &&
                         now - completion_order_.front().second >=
                             config_.completed_retention;
    if (!over_count && !expired) break;
    completed_.erase(found);
    completion_order_.pop_front();
  }
}

size_t RequestManager::active_count() const {
  std::lock_guard lock(mutex_);
  return active_.size();
}

size_t RequestManager::completed_count() const {
  std::lock_guard lock(mutex_);
  PruneCompletedLocked(config_.now());
  return completed_.size();
}

}  // namespace inferx::server::request
