// RequestContext unit tests: request-ID resolution/sanitization (UUIDv7
// shape), cancel-token propagation into the engine handle, usage counters,
// and exactly-once terminal state.
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>
#include <boost/asio.hpp>

#include "inferx/server/engine_client.h"
#include "inferx/server/request_context.h"
#include "inferx/server/sse_writer.h"

namespace inferx::server {
namespace {

class NullWriter final : public IStreamWriter {
 public:
  boost::asio::awaitable<Result<void>> Begin(int, HeaderMap) override {
    co_return Result<void>{};
  }
  boost::asio::awaitable<Result<void>> WriteEvent(std::string_view) override {
    co_return Result<void>{};
  }
  boost::asio::awaitable<Result<void>> WriteComment(std::string_view) override {
    co_return Result<void>{};
  }
  boost::asio::awaitable<Result<void>> WriteDone() override {
    co_return Result<void>{};
  }
  void Abort() noexcept override {}
  bool committed() const noexcept override { return false; }
  std::size_t bytes_written() const noexcept override { return 0; }
};

class RecordingEngine final : public IEngineClient {
 public:
  boost::asio::awaitable<Result<EngineSubmission>> SubmitGenerate(
      EngineGenerateRequest, const EngineCallContext&) override {
    co_return ApiErrorMapper_InternalStub();
  }
  boost::asio::awaitable<Result<EngineEmbedSubmission>> SubmitEmbed(
      EngineEmbedRequest, const EngineCallContext&) override {
    co_return ApiErrorMapper_InternalStub();
  }
  void Abort(EngineRequestId id, CancelReason reason) noexcept override {
    last_aborted_id = id;
    last_abort_reason = reason;
    abort_count.fetch_add(1);
  }
  boost::asio::awaitable<Result<std::vector<EngineModelInfo>>> ListModels()
      override {
    co_return std::vector<EngineModelInfo>{};
  }
  boost::asio::awaitable<EngineHealth> CheckHealth() override {
    co_return EngineHealth{};
  }
  boost::asio::awaitable<Result<void>> Shutdown(std::chrono::milliseconds)
      override {
    co_return Result<void>{};
  }

  static ApiError ApiErrorMapper_InternalStub() {
    return ApiError{500, ApiErrorKind::kInternal, "internal_error",
                    "internal_error", "not implemented", std::nullopt,
                    std::nullopt, false, {}};
  }

  std::optional<EngineRequestId> last_aborted_id;
  std::optional<CancelReason> last_abort_reason;
  std::atomic<int> abort_count{0};
};

class ClosedEventStream final : public IEngineEventStream {
 public:
  boost::asio::awaitable<Result<EngineGenerateEvent>> Next() override {
    co_return ApiError{500, ApiErrorKind::kEngine, "engine_error",
                       "stream_closed", "closed", std::nullopt, std::nullopt,
                       false, {}};
  }
  void Close() noexcept override { closed = true; }
  std::size_t buffered_events() const noexcept override { return 0; }
  bool closed = false;
};

/// Builds a fields container from pairs; http::fields has no
/// initializer-list constructor.
HeaderMap Headers(
    std::initializer_list<std::pair<std::string_view, std::string_view>> init) {
  HeaderMap fields;
  for (const auto& [name, value] : init) fields.set(name, value);
  return fields;
}

std::unique_ptr<RequestContext> MakeContext(const HeaderMap& headers = {}) {
  static NullWriter writer;
  HttpRequest request;
  request.headers = headers;
  return std::make_unique<RequestContext>(
      ClientRequestId("req-fixed"), ConnectionId("conn-1"), TenantId{},
      std::move(request), boost::asio::system_executor{}, Clock::now(),
      Clock::now() + std::chrono::seconds(30), &writer);
}

TEST(RequestIdTest, GeneratedIdIsUuidv7Shaped) {
  const std::string id = GenerateRequestId();
  ASSERT_EQ(id.size(), 36u);
  EXPECT_EQ(id[8], '-');
  EXPECT_EQ(id[13], '-');
  EXPECT_EQ(id[14], '7');  // version nibble
  EXPECT_EQ(id[18], '-');
  EXPECT_NE(id[19], '0');  // variant 10xx: first hex digit in {8,9,a,b}
  EXPECT_NE(id[19], '1');
  EXPECT_NE(id[19], '2');
  EXPECT_NE(id[19], '3');
  EXPECT_NE(id[19], '4');
  EXPECT_NE(id[19], '5');
  EXPECT_NE(id[19], '6');
  EXPECT_NE(id[19], '7');
  EXPECT_EQ(id[23], '-');
}

TEST(RequestIdTest, ValidHeaderEchoedSpoofedReplaced) {
  const ClientRequestId echoed = ResolveClientRequestId(
      Headers({{"X-Request-Id", "client-abc123"}}));
  EXPECT_EQ(echoed.str(), "client-abc123");

  const ClientRequestId replaced = ResolveClientRequestId(
      Headers({{"X-Request-Id", "bad id with spaces!!"}}));
  EXPECT_NE(replaced.str(), "bad id with spaces!!");
  EXPECT_EQ(replaced.str().size(), 36u);  // fresh UUIDv7

  const ClientRequestId too_short = ResolveClientRequestId(
      Headers({{"X-Request-Id", "abc"}}));
  EXPECT_EQ(too_short.str().size(), 36u);
}

TEST(RequestContextTest, CancelTokenPropagatesToEngineHandle) {
  auto engine = std::make_shared<RecordingEngine>();
  auto stream = std::make_unique<ClosedEventStream>();
  ClosedEventStream* stream_ptr = stream.get();
  auto ctx = MakeContext();
  ctx->set_engine_handle(EngineRequestHandle(engine, EngineRequestId("er-1"),
                                             std::move(stream)));
  ASSERT_TRUE(ctx->engine_handle() != nullptr);
  EXPECT_FALSE(ctx->cancelled());

  ctx->RequestCancel(CancelReason::kClientDisconnected);
  EXPECT_TRUE(ctx->cancelled());
  EXPECT_EQ(ctx->cancel_reason(), CancelReason::kClientDisconnected);
  EXPECT_EQ(engine->abort_count.load(), 1);
  EXPECT_EQ(engine->last_aborted_id.value().str(), "er-1");
  EXPECT_EQ(engine->last_abort_reason, CancelReason::kClientDisconnected);
  EXPECT_TRUE(stream_ptr->closed);

  // Second cancel is a no-op (exactly-once).
  ctx->RequestCancel(CancelReason::kInternal);
  EXPECT_EQ(engine->abort_count.load(), 1);
}

TEST(RequestContextTest, HandleWithoutCancelAbortsOnDestruction) {
  auto engine = std::make_shared<RecordingEngine>();
  {
    auto ctx = MakeContext();
    ctx->set_engine_handle(EngineRequestHandle(engine, EngineRequestId("er-2"),
                                               std::make_unique<ClosedEventStream>()));
  }
  EXPECT_EQ(engine->abort_count.load(), 1);
}

TEST(RequestContextTest, FinalizeIsExactlyOnce) {
  auto ctx = MakeContext();
  EXPECT_EQ(ctx->phase(), RequestPhase::kCreated);
  EXPECT_TRUE(ctx->finalize(RequestPhase::kCompleted));
  EXPECT_EQ(ctx->phase(), RequestPhase::kCompleted);
  EXPECT_FALSE(ctx->finalize(RequestPhase::kCancelled));
  EXPECT_FALSE(ctx->finalize(RequestPhase::kFailed));
  EXPECT_EQ(ctx->phase(), RequestPhase::kCompleted);
}

TEST(RequestContextTest, UsageCountersAndCommitFlag) {
  auto ctx = MakeContext();
  EXPECT_FALSE(ctx->response_committed());
  ctx->set_response_committed();
  EXPECT_TRUE(ctx->response_committed());
  ctx->set_usage_prompt_tokens(11);
  ctx->set_usage_completion_tokens(7);
  EXPECT_EQ(ctx->prompt_tokens(), 11u);
  EXPECT_EQ(ctx->completion_tokens(), 7u);
  EXPECT_FALSE(ctx->expired());
}

}  // namespace
}  // namespace inferx::server
