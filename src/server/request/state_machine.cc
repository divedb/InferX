#include "inferx/server/request/request_context.h"

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

}  // namespace inferx::server::request
