#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "inferx/server/admission/admission_controller.h"
#include "inferx/server/request/request_manager.h"
#include "inferx/server/request/request_service.h"

namespace inferx::server::request {

class ManagedRequestService final : public RequestService {
 public:
  ManagedRequestService(RequestManager* manager,
                        scheduler_client::SchedulerClient* scheduler,
                        admission::AdmissionController* admission = nullptr)
      : manager_(manager), scheduler_(scheduler), admission_(admission) {}

  folly::coro::Task<StatusOr<scheduler_client::SubmitResult>> Submit(
      scheduler_client::ScheduledRequest request,
      folly::CancellationToken cancellation) override;
  folly::coro::AsyncGenerator<scheduler_client::GenerationEvent&&> Events(
      RequestId request_id,
      folly::CancellationToken cancellation) override;
  folly::coro::Task<Status> Cancel(
      RequestId request_id, CancellationReason reason) override;

 private:
  struct ActiveRequest {
    std::shared_ptr<RequestContext> context;
    uint64_t admission_reservation = 0;
  };

  Status Finish(const RequestId& request_id, RequestState terminal);
  std::shared_ptr<RequestContext> FindContext(const RequestId& request_id);

  RequestManager* manager_;
  scheduler_client::SchedulerClient* scheduler_;
  admission::AdmissionController* admission_;
  std::mutex mutex_;
  std::unordered_map<RequestId, ActiveRequest> active_;
};

}  // namespace inferx::server::request
