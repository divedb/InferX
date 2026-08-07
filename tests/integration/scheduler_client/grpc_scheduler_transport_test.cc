#include "inferx/server/scheduler_client/grpc_scheduler_transport.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "inference/scheduler/v1/scheduler.grpc.pb.h"

namespace inferx::server::scheduler_client {
namespace {

namespace wire = ::inference::scheduler::v1;

class FakeSchedulerService final : public wire::Scheduler::Service {
 public:
  grpc::Status Submit(grpc::ServerContext*, const wire::SubmitRequest* request,
                      wire::SubmitResponse* response) override {
    last_submit = *request;
    ++submit_calls;
    if (request->request_id() == "slow") {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    response->set_request_id(request->request_id());
    response->set_attempt(request->attempt());
    return grpc::Status::OK;
  }

  grpc::Status Subscribe(
      grpc::ServerContext*, const wire::SubscribeRequest* request,
      grpc::ServerWriter<wire::GenerationEvent>* writer) override {
    last_subscribe = *request;
    wire::GenerationEvent delta;
    delta.set_request_id(request->request_id());
    delta.set_attempt(request->attempt());
    delta.set_sequence_number(request->after_sequence_number() + 1);
    delta.set_text_delta("hello");
    delta.add_token_ids(17);
    delta.set_generated_tokens(1);
    auto* embedding = delta.add_embeddings();
    embedding->set_index(3);
    embedding->add_values(0.25F);
    embedding->add_values(0.75F);
    writer->Write(delta);

    wire::GenerationEvent terminal;
    terminal.set_request_id(request->request_id());
    terminal.set_attempt(request->attempt());
    terminal.set_sequence_number(request->after_sequence_number() + 2);
    terminal.set_terminal(true);
    terminal.set_finish_reason(wire::FINISH_REASON_STOP);
    terminal.mutable_usage()->set_prompt_tokens(4);
    terminal.mutable_usage()->set_completion_tokens(1);
    writer->Write(terminal);
    return grpc::Status::OK;
  }

  grpc::Status Cancel(grpc::ServerContext*, const wire::CancelRequest* request,
                      wire::OperationResponse*) override {
    last_cancel = *request;
    ++cancel_calls;
    return grpc::Status::OK;
  }

  grpc::Status GetStatus(grpc::ServerContext*,
                         const wire::GetStatusRequest* request,
                         wire::StatusResponse* response) override {
    response->set_found(true);
    response->set_request_id(request->request_id());
    response->set_attempt(request->attempt());
    response->set_state(static_cast<int32_t>(request::RequestState::kDecoding));
    response->set_received_unix_millis(1700000000000);
    response->set_deadline_unix_millis(1700000001000);
    return grpc::Status::OK;
  }

  grpc::Status UpdatePriority(grpc::ServerContext*,
                              const wire::UpdatePriorityRequest* request,
                              wire::OperationResponse*) override {
    last_priority = *request;
    ++priority_calls;
    return grpc::Status::OK;
  }

  wire::SubmitRequest last_submit;
  wire::SubscribeRequest last_subscribe;
  wire::CancelRequest last_cancel;
  wire::UpdatePriorityRequest last_priority;
  int submit_calls = 0;
  int cancel_calls = 0;
  int priority_calls = 0;
};

class GrpcSchedulerTransportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    grpc::ServerBuilder builder;
    builder.RegisterService(&service);
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(),
                             &port);
    server = builder.BuildAndStart();
    ASSERT_NE(server, nullptr);
    ASSERT_GT(port, 0);
    transport = std::make_unique<GrpcSchedulerTransport>(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(port),
                            grpc::InsecureChannelCredentials()));
  }

  void TearDown() override {
    transport.reset();
    if (server != nullptr) {
      server->Shutdown(std::chrono::system_clock::now());
      server->Wait();
    }
  }

  FakeSchedulerService service;
  int port = 0;
  std::unique_ptr<grpc::Server> server;
  std::unique_ptr<GrpcSchedulerTransport> transport;
};

TEST_F(GrpcSchedulerTransportTest, MapsSubmissionAndControlCalls) {
  ScheduledRequest request;
  request.request_id = "req-1";
  request.attempt = 2;
  request.tenant_id = "tenant-a";
  request.model_version = "model@v3";
  request.workload = WorkloadClass::kEmbedding;
  request.priority = PriorityClass::kBatch;
  request.prompt_tokens = {11, 22};
  request.sampling = {.max_tokens = 9,
                      .temperature = 0.5F,
                      .top_p = 0.8F,
                      .seed = 42,
                      .stop = {"END"}};
  request.tokenizer_revision = "tok-v3";
  request.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  request.traceparent = "00-trace-parent";
  request.tracestate = "vendor=value";

  auto submitted = transport->Submit(request);
  ASSERT_TRUE(submitted.ok()) << submitted.status();
  EXPECT_EQ(submitted->request_id, request.request_id);
  EXPECT_EQ(submitted->attempt, request.attempt);
  EXPECT_EQ(service.last_submit.tenant_id(), request.tenant_id);
  EXPECT_EQ(service.last_submit.model_version(), request.model_version);
  EXPECT_EQ(service.last_submit.workload(), wire::WORKLOAD_CLASS_EMBEDDING);
  EXPECT_EQ(service.last_submit.priority(), wire::PRIORITY_CLASS_BATCH);
  EXPECT_EQ(service.last_submit.prompt_tokens_size(), 2);
  EXPECT_EQ(service.last_submit.sampling().stop(0), "END");
  EXPECT_EQ(service.last_submit.tokenizer_revision(), "tok-v3");
  EXPECT_GT(service.last_submit.deadline_unix_millis(), 0);
  EXPECT_EQ(service.last_submit.trace().traceparent(), "00-trace-parent");

  auto duplicate = transport->Submit(request);
  ASSERT_TRUE(duplicate.ok()) << duplicate.status();
  EXPECT_EQ(duplicate->attempt, request.attempt);
  EXPECT_EQ(service.submit_calls, 2);

  auto status = transport->GetStatus(request.request_id, request.attempt);
  ASSERT_TRUE(status.ok()) << status.status();
  EXPECT_TRUE(status->found);
  EXPECT_EQ(status->attempt, 2);
  EXPECT_EQ(status->snapshot.state, request::RequestState::kDecoding);

  EXPECT_TRUE(transport
                  ->UpdatePriority(request.request_id, request.attempt,
                                   PriorityClass::kBackground)
                  .ok());
  EXPECT_EQ(service.last_priority.priority(), wire::PRIORITY_CLASS_BACKGROUND);
  EXPECT_TRUE(transport
                  ->Cancel(request.request_id, request.attempt,
                           request::CancellationReason::kClientDisconnected)
                  .ok());
  EXPECT_EQ(service.last_cancel.request_id(), request.request_id);
  EXPECT_TRUE(transport
                  ->Cancel(request.request_id, request.attempt,
                           request::CancellationReason::kClientDisconnected)
                  .ok());
  EXPECT_EQ(service.cancel_calls, 2);
}

TEST_F(GrpcSchedulerTransportTest, MapsStreamedEventsAndResumeCursor) {
  auto opened = transport->Subscribe("req-stream", 4, 9);
  ASSERT_TRUE(opened.ok()) << opened.status();
  auto first = (*opened)->Next();
  ASSERT_TRUE(first.ok()) << first.status();
  ASSERT_TRUE(first->has_value());
  EXPECT_EQ((*first)->sequence_number, 10);
  EXPECT_EQ((*first)->text_delta, "hello");
  EXPECT_EQ((*first)->token_ids, (std::vector<int32_t>{17}));
  ASSERT_EQ((*first)->embeddings.size(), 1);
  EXPECT_EQ((*first)->embeddings[0].index, 3);
  EXPECT_EQ((*first)->embeddings[0].values, (std::vector<float>{0.25F, 0.75F}));

  auto terminal = (*opened)->Next();
  ASSERT_TRUE(terminal.ok()) << terminal.status();
  ASSERT_TRUE(terminal->has_value());
  EXPECT_TRUE((*terminal)->terminal);
  EXPECT_EQ((*terminal)->finish_reason, FinishReason::kStop);
  EXPECT_EQ((*terminal)->usage.prompt_tokens, 4);
  EXPECT_EQ(service.last_subscribe.after_sequence_number(), 9);

  auto finished = (*opened)->Next();
  ASSERT_TRUE(finished.ok()) << finished.status();
  EXPECT_FALSE(finished->has_value());
}

TEST_F(GrpcSchedulerTransportTest, EnforcesDeadlineAndShutdown) {
  ScheduledRequest request;
  request.request_id = "slow";
  request.attempt = 1;
  request.deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
  auto result = transport->Submit(request);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kDeadlineExceeded);

  transport->Shutdown();
  result = transport->Submit(request);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kCancelled);
}

}  // namespace
}  // namespace inferx::server::scheduler_client
