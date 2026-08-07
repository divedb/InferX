#pragma once

#include <grpcpp/channel.h>

#include <memory>

#include "inferx/server/scheduler_client/remote_scheduler_client.h"

namespace inferx::server::scheduler_client {

/// Blocking gRPC implementation used behind RemoteSchedulerClient's dedicated
/// executor. RPC retry and event-resume policy remain in RemoteSchedulerClient.
class GrpcSchedulerTransport final : public RemoteSchedulerTransport {
 public:
  explicit GrpcSchedulerTransport(std::shared_ptr<grpc::Channel> channel);
  ~GrpcSchedulerTransport() override;

  StatusOr<SubmitResult> Submit(const ScheduledRequest& request) override;
  StatusOr<RemoteRequestStatus> GetStatus(
      const request::RequestId& request_id, uint32_t attempt) override;
  Status Cancel(const request::RequestId& request_id, uint32_t attempt,
                request::CancellationReason reason) override;
  Status UpdatePriority(const request::RequestId& request_id, uint32_t attempt,
                        PriorityClass priority) override;
  StatusOr<std::unique_ptr<RemoteEventStream>> Subscribe(
      const request::RequestId& request_id, uint32_t attempt,
      uint64_t after_sequence_number) override;
  void Shutdown() override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server::scheduler_client
