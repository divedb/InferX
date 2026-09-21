#include "inferx/engine/request.h"

#include <utility>

namespace inferx {

Status Request::AppendOutput(absl::Span<const TokenId> token_ids) {
  if (status_ == RequestStatus::kFinished) {
    return FailedPreconditionError("request ", id_, " is finished");
  }
  output_.insert(output_.end(), token_ids.begin(), token_ids.end());
  return OkStatus();
}

Status Request::Finish(FinishReason reason) {
  if (status_ == RequestStatus::kFinished) {
    return FailedPreconditionError("request ", id_, " is already finished");
  }
  status_ = RequestStatus::kFinished;
  finish_reason_ = reason;
  block_table_.reset();
  return OkStatus();
}

}  // namespace inferx
