#include "inferx/server/scheduler_client/grpc_scheduler_transport.h"

#include <grpcpp/client_context.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "inference/scheduler/v1/scheduler.grpc.pb.h"

namespace inferx::server::scheduler_client {
namespace {

namespace wire = ::inference::scheduler::v1;

Status FromGrpc(const grpc::Status& status) {
  if (status.ok()) return OkStatus();
  const auto code = static_cast<absl::StatusCode>(status.error_code());
  return Status(code, status.error_message());
}

wire::WorkloadClass ToWire(WorkloadClass workload) {
  return workload == WorkloadClass::kEmbedding
             ? wire::WORKLOAD_CLASS_EMBEDDING
             : wire::WORKLOAD_CLASS_GENERATION;
}

wire::PriorityClass ToWire(PriorityClass priority) {
  switch (priority) {
    case PriorityClass::kInteractive:
      return wire::PRIORITY_CLASS_INTERACTIVE;
    case PriorityClass::kBatch:
      return wire::PRIORITY_CLASS_BATCH;
    case PriorityClass::kBackground:
      return wire::PRIORITY_CLASS_BACKGROUND;
  }
  return wire::PRIORITY_CLASS_UNSPECIFIED;
}

FinishReason FromWire(wire::FinishReason reason) {
  switch (reason) {
    case wire::FINISH_REASON_STOP:
      return FinishReason::kStop;
    case wire::FINISH_REASON_LENGTH:
      return FinishReason::kLength;
    case wire::FINISH_REASON_CANCELLED:
      return FinishReason::kCancelled;
    case wire::FINISH_REASON_FAILED:
      return FinishReason::kFailed;
    case wire::FINISH_REASON_UNSPECIFIED:
      return FinishReason::kNone;
    default:
      return FinishReason::kFailed;
  }
}

int64_t DeadlineUnixMillis(std::chrono::steady_clock::time_point deadline) {
  if (deadline == std::chrono::steady_clock::time_point{}) return 0;
  const auto remaining = deadline - std::chrono::steady_clock::now();
  const auto wall_deadline = std::chrono::system_clock::now() + remaining;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             wall_deadline.time_since_epoch())
      .count();
}

std::chrono::steady_clock::time_point FromUnixMillis(int64_t value) {
  if (value == 0) return {};
  const auto wall =
      std::chrono::system_clock::time_point(std::chrono::milliseconds(value));
  return std::chrono::steady_clock::now() +
         (wall - std::chrono::system_clock::now());
}

void Populate(const ScheduledRequest& request, wire::SubmitRequest* output) {
  output->set_request_id(request.request_id);
  output->set_attempt(request.attempt);
  output->set_tenant_id(request.tenant_id);
  output->set_model_version(request.model_version);
  output->set_workload(ToWire(request.workload));
  output->set_priority(ToWire(request.priority));
  for (int32_t token : request.prompt_tokens) output->add_prompt_tokens(token);
  auto* sampling = output->mutable_sampling();
  sampling->set_max_tokens(request.sampling.max_tokens);
  sampling->set_temperature(request.sampling.temperature);
  sampling->set_top_p(request.sampling.top_p);
  sampling->set_seed(request.sampling.seed);
  for (const std::string& stop : request.sampling.stop) {
    sampling->add_stop(stop);
  }
  output->set_tokenizer_revision(request.tokenizer_revision);
  output->set_deadline_unix_millis(DeadlineUnixMillis(request.deadline));
  output->mutable_trace()->set_traceparent(request.traceparent);
  output->mutable_trace()->set_tracestate(request.tracestate);
}

GenerationEvent FromWire(const wire::GenerationEvent& input) {
  GenerationEvent result;
  result.request_id = input.request_id();
  result.attempt = input.attempt();
  result.sequence_number = input.sequence_number();
  result.text_delta = input.text_delta();
  result.token_ids.assign(input.token_ids().begin(), input.token_ids().end());
  result.embeddings.reserve(input.embeddings_size());
  for (const auto& embedding : input.embeddings()) {
    EmbeddingResult mapped;
    mapped.index = embedding.index();
    mapped.values.assign(embedding.values().begin(), embedding.values().end());
    result.embeddings.push_back(std::move(mapped));
  }
  result.generated_tokens = input.generated_tokens();
  result.terminal = input.terminal();
  result.finish_reason = FromWire(input.finish_reason());
  result.usage = {.prompt_tokens = input.usage().prompt_tokens(),
                  .completion_tokens = input.usage().completion_tokens()};
  if (input.has_error() && input.error().code() != 0) {
    result.error = Status(static_cast<absl::StatusCode>(input.error().code()),
                          input.error().message());
  }
  return result;
}

class GrpcEventStream final : public RemoteEventStream {
 public:
  GrpcEventStream(
      std::shared_ptr<grpc::ClientContext> context,
      std::unique_ptr<grpc::ClientReader<wire::GenerationEvent>> reader)
      : context_(std::move(context)), reader_(std::move(reader)) {}

  StatusOr<std::optional<GenerationEvent>> Next() override {
    wire::GenerationEvent event;
    if (reader_->Read(&event)) {
      return std::optional<GenerationEvent>(FromWire(event));
    }
    const Status finished = FromGrpc(reader_->Finish());
    if (!finished.ok()) return finished;
    return std::optional<GenerationEvent>{};
  }

  void Cancel() override { context_->TryCancel(); }

 private:
  std::shared_ptr<grpc::ClientContext> context_;
  std::unique_ptr<grpc::ClientReader<wire::GenerationEvent>> reader_;
};

}  // namespace

class GrpcSchedulerTransport::Impl {
 public:
  explicit Impl(std::shared_ptr<grpc::Channel> channel)
      : stub(wire::Scheduler::NewStub(std::move(channel))) {}

  std::unique_ptr<wire::Scheduler::Stub> stub;
  std::atomic<bool> shutdown{false};
  std::mutex contexts_mutex;
  std::vector<std::weak_ptr<grpc::ClientContext>> contexts;

  std::shared_ptr<grpc::ClientContext> NewContext() {
    auto context = std::make_shared<grpc::ClientContext>();
    std::lock_guard lock(contexts_mutex);
    contexts.erase(
        std::remove_if(contexts.begin(), contexts.end(),
                       [](const auto& item) { return item.expired(); }),
        contexts.end());
    contexts.push_back(context);
    if (shutdown.load()) context->TryCancel();
    return context;
  }

  void Shutdown() {
    if (shutdown.exchange(true)) return;
    std::lock_guard lock(contexts_mutex);
    for (const auto& item : contexts) {
      if (auto context = item.lock()) context->TryCancel();
    }
    contexts.clear();
  }
};

GrpcSchedulerTransport::GrpcSchedulerTransport(
    std::shared_ptr<grpc::Channel> channel)
    : impl_(std::make_unique<Impl>(std::move(channel))) {}

GrpcSchedulerTransport::~GrpcSchedulerTransport() = default;

StatusOr<SubmitResult> GrpcSchedulerTransport::Submit(
    const ScheduledRequest& request) {
  if (impl_->shutdown.load()) return absl::CancelledError("transport stopped");
  auto context = impl_->NewContext();
  wire::SubmitRequest command;
  wire::SubmitResponse response;
  Populate(request, &command);
  if (request.deadline != std::chrono::steady_clock::time_point{}) {
    context->set_deadline(
        std::chrono::system_clock::now() +
        (request.deadline - std::chrono::steady_clock::now()));
  }
  const grpc::Status status =
      impl_->stub->Submit(context.get(), command, &response);
  if (!status.ok()) return FromGrpc(status);
  return SubmitResult{response.request_id(), response.attempt()};
}

StatusOr<RemoteRequestStatus> GrpcSchedulerTransport::GetStatus(
    const request::RequestId& request_id, uint32_t attempt) {
  if (impl_->shutdown.load()) return absl::CancelledError("transport stopped");
  auto context = impl_->NewContext();
  wire::GetStatusRequest command;
  wire::StatusResponse response;
  command.set_request_id(request_id);
  command.set_attempt(attempt);
  const grpc::Status status =
      impl_->stub->GetStatus(context.get(), command, &response);
  if (!status.ok()) return FromGrpc(status);
  request::RequestSnapshot snapshot;
  snapshot.request_id = response.request_id();
  snapshot.state = static_cast<request::RequestState>(response.state());
  snapshot.received_at = FromUnixMillis(response.received_unix_millis());
  snapshot.deadline = FromUnixMillis(response.deadline_unix_millis());
  snapshot.completed_at = FromUnixMillis(response.completed_unix_millis());
  return RemoteRequestStatus{response.found(), response.attempt(), snapshot};
}

Status GrpcSchedulerTransport::Cancel(const request::RequestId& request_id,
                                      uint32_t attempt,
                                      request::CancellationReason reason) {
  if (impl_->shutdown.load()) return absl::CancelledError("transport stopped");
  auto context = impl_->NewContext();
  wire::CancelRequest command;
  wire::OperationResponse response;
  command.set_request_id(request_id);
  command.set_attempt(attempt);
  command.set_reason(static_cast<int32_t>(reason));
  return FromGrpc(impl_->stub->Cancel(context.get(), command, &response));
}

Status GrpcSchedulerTransport::UpdatePriority(
    const request::RequestId& request_id, uint32_t attempt,
    PriorityClass priority) {
  if (impl_->shutdown.load()) return absl::CancelledError("transport stopped");
  auto context = impl_->NewContext();
  wire::UpdatePriorityRequest command;
  wire::OperationResponse response;
  command.set_request_id(request_id);
  command.set_attempt(attempt);
  command.set_priority(ToWire(priority));
  return FromGrpc(
      impl_->stub->UpdatePriority(context.get(), command, &response));
}

StatusOr<std::unique_ptr<RemoteEventStream>> GrpcSchedulerTransport::Subscribe(
    const request::RequestId& request_id, uint32_t attempt,
    uint64_t after_sequence_number) {
  if (impl_->shutdown.load()) return absl::CancelledError("transport stopped");
  auto context = impl_->NewContext();
  wire::SubscribeRequest command;
  command.set_request_id(request_id);
  command.set_attempt(attempt);
  command.set_after_sequence_number(after_sequence_number);
  auto reader = impl_->stub->Subscribe(context.get(), command);
  if (reader == nullptr) return absl::UnavailableError("stream did not open");
  return std::unique_ptr<RemoteEventStream>(
      std::make_unique<GrpcEventStream>(std::move(context), std::move(reader)));
}

void GrpcSchedulerTransport::Shutdown() { impl_->Shutdown(); }

}  // namespace inferx::server::scheduler_client
