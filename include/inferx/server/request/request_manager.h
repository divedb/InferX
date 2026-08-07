#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "inferx/core/status.h"
#include "inferx/server/request/request_context.h"

namespace inferx::server::request {

struct RequestSnapshot {
  RequestId request_id;
  RequestState state = RequestState::kReceived;
  CancellationReason cancellation_reason = CancellationReason::kInternal;
  std::chrono::steady_clock::time_point received_at;
  std::chrono::steady_clock::time_point deadline;
  std::chrono::steady_clock::time_point completed_at;
};

class RequestManager {
 public:
  StatusOr<std::shared_ptr<RequestContext>> Create(RequestContext context);
  Status Cancel(const RequestId& id, CancellationReason reason);
  StatusOr<RequestSnapshot> GetStatus(const RequestId& id) const;
  Status Finalize(const RequestId& id);
  size_t active_count() const;

 private:
  static RequestSnapshot Snapshot(const RequestContext& context);

  mutable std::mutex mutex_;
  std::unordered_map<RequestId, std::shared_ptr<RequestContext>> active_;
};

}  // namespace inferx::server::request
