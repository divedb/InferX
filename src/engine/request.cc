#include "inferx/engine/request.h"

#include <utility>

namespace inferx {

Status Request::AppendOutput(absl::Span<const TokenId> token_ids) {
  if (status_ != RequestStatus::kRunning) {
    return FailedPreconditionError("Request ", id_,
                                   " is not running; cannot append output tokens");
  }

  all_token_ids_.insert(all_token_ids_.end(), token_ids.begin(), token_ids.end());

  return OkStatus();
}

Status Request::MarkRunning() {
  if (status_ != RequestStatus::kWaiting) {
    return FailedPreconditionError("Request ", id_, " is not waiting");
  }

  status_ = RequestStatus::kRunning;

  return OkStatus();
}

Status Request::MarkWaiting() {
  if (status_ != RequestStatus::kPreempted) {
    return FailedPreconditionError("Request ", id_, " is not preempted");
  }

  status_ = RequestStatus::kWaiting;

  return OkStatus();
}

Status Request::Preempt() {
  if (status_ != RequestStatus::kRunning) {
    return FailedPreconditionError("Request ", id_, " is not running");
  }

  status_ = RequestStatus::kPreempted;
  num_computed_tokens_ = 0;
  ++num_preemptions_;

  return OkStatus();
}

Status Request::Finish(FinishReason reason, StopReason stop_reason) {
  if (IsFinished()) {
    return FailedPreconditionError("Request ", id_, " is already finished");
  }

  status_ = RequestStatus::kFinished;
  finish_reason_ = reason;
  stop_reason_ = std::move(stop_reason);

  return OkStatus();
}

}  // namespace inferx
