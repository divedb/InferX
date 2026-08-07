#include "inferx/server/streaming/event_router.h"

#include <utility>

namespace inferx::server::streaming {

EventRouter::EventRouter(EventBuffer* buffer,
                         scheduler_client::RequestId request_id)
    : buffer_(buffer), request_id_(std::move(request_id)) {}

Status EventRouter::Route(scheduler_client::GenerationEvent event) {
  if (buffer_ == nullptr) return FailedPreconditionError("event buffer is null");
  if (event.request_id != request_id_) {
    return InvalidArgumentError("generation event request ID mismatch");
  }
  if (event.sequence_number != next_sequence_) {
    return InvalidArgumentError("generation event sequence mismatch: expected ",
                                next_sequence_, ", received ",
                                event.sequence_number);
  }
  const PushResult result = buffer_->TryPush(std::move(event));
  if (result == PushResult::kFull) {
    return ResourceExhaustedError("generation event buffer is full");
  }
  if (result == PushResult::kClosed) {
    return FailedPreconditionError("generation event buffer is closed");
  }
  ++next_sequence_;
  return OkStatus();
}

}  // namespace inferx::server::streaming
