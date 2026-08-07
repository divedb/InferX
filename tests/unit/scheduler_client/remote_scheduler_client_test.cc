#include <gtest/gtest.h>

#include <folly/coro/BlockingWait.h>
#include <folly/coro/Task.h>
#include <folly/executors/CPUThreadPoolExecutor.h>

#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "inferx/server/scheduler_client/remote_scheduler_client.h"

namespace inferx::server::scheduler_client {
namespace {

class ScriptedStream final : public RemoteEventStream {
 public:
  explicit ScriptedStream(
      std::deque<StatusOr<std::optional<GenerationEvent>>> script)
      : script_(std::move(script)) {}

  StatusOr<std::optional<GenerationEvent>> Next() override {
    if (script_.empty()) return std::optional<GenerationEvent>{};
    auto next = std::move(script_.front());
    script_.pop_front();
    return next;
  }
  void Cancel() override { cancelled = true; }

  bool cancelled = false;

 private:
  std::deque<StatusOr<std::optional<GenerationEvent>>> script_;
};

class FakeRemoteTransport final : public RemoteSchedulerTransport {
 public:
  StatusOr<SubmitResult> Submit(const ScheduledRequest& request) override {
    ++submit_calls;
    seen = request;
    if (!submit_results.empty()) {
      auto result = std::move(submit_results.front());
      submit_results.pop_front();
      return result;
    }
    return SubmitResult{request.request_id, request.attempt};
  }

  StatusOr<RemoteRequestStatus> GetStatus(
      const request::RequestId&, uint32_t) override {
    ++status_calls;
    return remote_status;
  }

  Status Cancel(const request::RequestId&, uint32_t,
                request::CancellationReason) override {
    ++cancel_calls;
    return OkStatus();
  }

  Status UpdatePriority(const request::RequestId&, uint32_t,
                        PriorityClass) override {
    ++priority_calls;
    return OkStatus();
  }

  StatusOr<std::unique_ptr<RemoteEventStream>> Subscribe(
      const request::RequestId&, uint32_t, uint64_t after) override {
    subscribe_after.push_back(after);
    if (streams.empty()) return absl::UnavailableError("no stream");
    auto stream = std::move(streams.front());
    streams.pop_front();
    return std::unique_ptr<RemoteEventStream>(std::move(stream));
  }

  void Shutdown() override { shutdown = true; }

  ScheduledRequest seen;
  RemoteRequestStatus remote_status;
  std::deque<StatusOr<SubmitResult>> submit_results;
  std::deque<std::unique_ptr<ScriptedStream>> streams;
  std::vector<uint64_t> subscribe_after;
  int submit_calls = 0;
  int status_calls = 0;
  int cancel_calls = 0;
  int priority_calls = 0;
  bool shutdown = false;
};

RemoteSchedulerClientConfig NoDelayConfig() {
  return {.max_attempts = 3,
          .initial_backoff = std::chrono::milliseconds(0),
          .max_backoff = std::chrono::milliseconds(0),
          .sleep = [](std::chrono::milliseconds) {}};
}

TEST(RemoteSchedulerClientTest, ResolvesAmbiguousSubmitByStatus) {
  auto transport = std::make_shared<FakeRemoteTransport>();
  transport->submit_results.push_back(
      absl::UnavailableError("reply lost after acceptance"));
  transport->remote_status = {.found = true, .attempt = 7};
  folly::CPUThreadPoolExecutor blocking(1);
  folly::CPUThreadPoolExecutor coroutine(1);
  RemoteSchedulerClient client(transport, &blocking, NoDelayConfig());
  ScheduledRequest request;
  request.request_id = "req_ambiguous";
  request.attempt = 7;
  request.model_version = "model@v1";
  request.traceparent = "00-trace-parent";

  auto future = folly::coro::co_withExecutor(
                    &coroutine, client.Submit(request, {}))
                    .start();
  auto result = std::move(future).get();
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->attempt, 7);
  EXPECT_EQ(transport->submit_calls, 1);
  EXPECT_EQ(transport->status_calls, 1);
  EXPECT_EQ(transport->seen.traceparent, "00-trace-parent");
}

TEST(RemoteSchedulerClientTest, ResumesStreamWithoutSplicingAttempts) {
  auto transport = std::make_shared<FakeRemoteTransport>();
  GenerationEvent first{.request_id = "req_stream",
                        .attempt = 2,
                        .sequence_number = 1,
                        .text_delta = "a"};
  transport->streams.push_back(std::make_unique<ScriptedStream>(
      std::deque<StatusOr<std::optional<GenerationEvent>>>{
          std::optional<GenerationEvent>(first),
          absl::UnavailableError("scheduler restarted")}));
  GenerationEvent terminal{.request_id = "req_stream",
                           .attempt = 2,
                           .sequence_number = 2,
                           .terminal = true,
                           .finish_reason = FinishReason::kStop};
  transport->streams.push_back(std::make_unique<ScriptedStream>(
      std::deque<StatusOr<std::optional<GenerationEvent>>>{
          std::optional<GenerationEvent>(terminal)}));
  folly::CPUThreadPoolExecutor blocking(1);
  folly::CPUThreadPoolExecutor coroutine(1);
  RemoteSchedulerClient client(transport, &blocking, NoDelayConfig());

  auto task = [&]() -> folly::coro::Task<std::vector<GenerationEvent>> {
    ScheduledRequest request;
    request.request_id = "req_stream";
    request.attempt = 2;
    auto submitted = co_await client.Submit(request, {});
    EXPECT_TRUE(submitted.ok()) << submitted.status();
    if (!submitted.ok()) co_return std::vector<GenerationEvent>{};
    std::vector<GenerationEvent> events;
    auto stream = client.Events(request.request_id, {});
    while (auto next = co_await stream.next()) {
      events.push_back(std::move(*next));
    }
    co_return events;
  };
  auto future = folly::coro::co_withExecutor(&coroutine, task()).start();
  auto events = std::move(future).get();
  ASSERT_EQ(events.size(), 2);
  EXPECT_EQ(events[0].text_delta, "a");
  EXPECT_TRUE(events[1].terminal);
  EXPECT_EQ(transport->subscribe_after, (std::vector<uint64_t>{0, 1}));
}

TEST(RemoteSchedulerClientTest, RejectsExpiredSubmissionAndShutsDown) {
  auto transport = std::make_shared<FakeRemoteTransport>();
  folly::CPUThreadPoolExecutor blocking(1);
  folly::CPUThreadPoolExecutor coroutine(1);
  {
    RemoteSchedulerClient client(transport, &blocking, NoDelayConfig());
    ScheduledRequest request;
    request.request_id = "req_expired";
    request.deadline = std::chrono::steady_clock::now() -
                       std::chrono::milliseconds(1);
    auto future = folly::coro::co_withExecutor(
                      &coroutine, client.Submit(request, {}))
                      .start();
    auto result = std::move(future).get();
    EXPECT_EQ(result.status().code(), absl::StatusCode::kDeadlineExceeded);
    EXPECT_EQ(transport->submit_calls, 0);
  }
  EXPECT_TRUE(transport->shutdown);
}

}  // namespace
}  // namespace inferx::server::scheduler_client
