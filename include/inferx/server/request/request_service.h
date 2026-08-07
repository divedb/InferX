#pragma once

#include <folly/CancellationToken.h>
#include <folly/coro/AsyncGenerator.h>
#include <folly/coro/Task.h>

#include "inferx/server/scheduler_client/scheduler_client.h"

namespace inferx::server::request {

/// Lifecycle-aware execution seam consumed by HTTP handlers. Implementations
/// own admission, RequestManager transitions, deadlines, scheduler submission,
/// cancellation, and accounting reconciliation.
class RequestService {
 public:
  virtual ~RequestService() = default;
  virtual folly::coro::Task<StatusOr<scheduler_client::SubmitResult>> Submit(
      scheduler_client::ScheduledRequest request,
      folly::CancellationToken cancellation) = 0;
  virtual folly::coro::AsyncGenerator<
      scheduler_client::GenerationEvent&&>
  Events(RequestId request_id, folly::CancellationToken cancellation) = 0;
  virtual folly::coro::Task<Status> Cancel(
      RequestId request_id, CancellationReason reason) = 0;
};

}  // namespace inferx::server::request
