#pragma once

#include <chrono>
#include <deque>
#include <functional>
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

struct RequestManagerConfig {
  std::chrono::seconds completed_retention{300};
  size_t max_completed_snapshots = 10000;
  std::function<std::chrono::steady_clock::time_point()> now =
      [] { return std::chrono::steady_clock::now(); };
};

class RequestManager {
 public:
  explicit RequestManager(RequestManagerConfig config = {});
  StatusOr<std::shared_ptr<RequestContext>> Create(RequestContext context);
  Status Cancel(const RequestId& id, CancellationReason reason);
  StatusOr<RequestSnapshot> GetStatus(const RequestId& id) const;
  Status Finalize(const RequestId& id);
  size_t active_count() const;
  size_t completed_count() const;

 private:
  static RequestSnapshot Snapshot(const RequestContext& context);
  void PruneCompletedLocked(std::chrono::steady_clock::time_point now) const;

  RequestManagerConfig config_;
  mutable std::mutex mutex_;
  std::unordered_map<RequestId, std::shared_ptr<RequestContext>> active_;
  mutable std::unordered_map<RequestId, RequestSnapshot> completed_;
  mutable std::deque<
      std::pair<RequestId, std::chrono::steady_clock::time_point>>
      completion_order_;
};

}  // namespace inferx::server::request
