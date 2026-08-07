#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/server/streaming/event_buffer.h"

namespace inferx::server::streaming {

class EventRouter {
 public:
  EventRouter(EventBuffer* buffer, scheduler_client::RequestId request_id);

  /// Validates identity/order, then enqueues exactly one event.
  Status Route(scheduler_client::GenerationEvent event);
  uint64_t next_sequence() const { return next_sequence_; }

 private:
  EventBuffer* buffer_;
  scheduler_client::RequestId request_id_;
  uint64_t next_sequence_ = 1;
};

}  // namespace inferx::server::streaming
