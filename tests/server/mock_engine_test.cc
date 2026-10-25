// MockEngine contract conformance (docs/server_design.md §9.2 E1–E6,
// §14.1 step 6). These tests pin the double itself: every server test that
// uses MockEngine relies on exactly this behavior.
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "gtest/gtest.h"

#include "mock_engine.h"

namespace inferx::server {
namespace {

// Boundary types moved to namespace inferx (inferx/engine/engine_client.h);
// from inferx::server they resolve unqualified.

/// Records sink calls; OnTokens/OnTerminal arrive on the engine thread.
class CountingSink final : public EventSink {
 public:
  EventSink::Ack OnTokens(std::span<const TokenId> ids) override {
    std::lock_guard<std::mutex> lock(mu_);
    batches_.emplace_back(ids.begin(), ids.end());
    total_tokens_ += static_cast<std::uint32_t>(ids.size());
    return EventSink::Ack::kOk;
  }

  void OnTerminal(TerminalEvent event) noexcept override {
    std::lock_guard<std::mutex> lock(mu_);
    terminal_ = event;
  }

  [[nodiscard]] std::vector<std::vector<TokenId>> batches() const {
    std::lock_guard<std::mutex> lock(mu_);
    return batches_;
  }
  [[nodiscard]] std::uint32_t total_tokens() const {
    std::lock_guard<std::mutex> lock(mu_);
    return total_tokens_;
  }
  [[nodiscard]] std::optional<TerminalEvent> terminal() const {
    std::lock_guard<std::mutex> lock(mu_);
    return terminal_;
  }

 private:
  mutable std::mutex mu_;
  std::vector<std::vector<TokenId>> batches_;
  std::uint32_t total_tokens_ = 0;
  std::optional<TerminalEvent> terminal_;
};

GenerationRequest MakeRequest(std::vector<TokenId> prompt = {7, 8, 9}) {
  GenerationRequest request;
  request.prompt_tokens = std::move(prompt);
  request.sampling.max_tokens = 16;
  return request;
}

/// Polls until `predicate` is true or a generous deadline passes (timing
/// tests must survive slow CI and sanitizer instrumentation).
template <typename Predicate>
bool Eventually(Predicate predicate,
                std::chrono::milliseconds timeout = std::chrono::milliseconds{5000}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  return predicate();
}

MockEngine::Script SimpleScript() {
  MockEngine::Script script;
  script.tokens_per_step = {{1, 2}, {3}};
  return script;
}

TEST(MockEngineTest, AcceptedRequestReceivesTokensThenFinishedTerminal) {
  MockEngine engine;
  engine.SetScript(SimpleScript());
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  auto handle = engine.Submit(MakeRequest(), sink);
  ASSERT_TRUE(handle.ok()) << handle.status();

  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));
  const TerminalEvent terminal = *sink->terminal();
  EXPECT_EQ(terminal.reason, FinishReason::kStopped);
  // Authoritative count covers every emitted token (§6.5).
  EXPECT_EQ(terminal.completion_tokens, 3u);
  EXPECT_EQ(sink->batches(), (std::vector<std::vector<TokenId>>{{1, 2}, {3}}));

  EXPECT_TRUE(Eventually([&] { return engine.all_released(); }));
  EXPECT_EQ(engine.live_requests(), 0u);
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{2000}).ok());
}

TEST(MockEngineTest, SubmitIsNonBlockingWhileEngineSteps) {  // E1
  MockEngine engine;
  MockEngine::Script script = SimpleScript();
  script.tokens_per_step = {{1}, {2}, {3}, {4}, {5}, {6}};
  script.step_delay = std::chrono::milliseconds{100};
  engine.SetScript(script);
  engine.Start();

  auto first = std::make_shared<CountingSink>();
  ASSERT_TRUE(engine.Submit(MakeRequest(), first).ok());
  ASSERT_TRUE(Eventually([&] { return !first->batches().empty(); }));

  // The engine thread sleeps between steps; Submit must still return fast.
  const auto start = std::chrono::steady_clock::now();
  auto second = std::make_shared<CountingSink>();
  ASSERT_TRUE(engine.Submit(MakeRequest(), second).ok());
  const auto elapsed = std::chrono::steady_clock::now() - start;
  EXPECT_LT(elapsed, std::chrono::milliseconds{50});

  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{5000}).ok());
}

TEST(MockEngineTest, SubmitRejectionUsesClosedSetAndRetainsNothing) {  // E6
  {
    MockEngine engine;
    MockEngine::Script script;
    script.reject_submit = absl::ResourceExhaustedError("inbox full");
    engine.SetScript(script);
    engine.Start();
    auto sink = std::make_shared<CountingSink>();
    auto handle = engine.Submit(MakeRequest(), sink);
    EXPECT_FALSE(handle.ok());
    EXPECT_TRUE(absl::IsResourceExhausted(handle.status()));
    // Rejection means no callbacks and no retained sink (E2).
    EXPECT_EQ(sink->batches().size(), 0u);
    EXPECT_FALSE(sink->terminal().has_value());
    EXPECT_EQ(engine.live_requests(), 0u);
    EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
  }
  {
    MockEngine engine;
    engine.SetScript(SimpleScript());
    engine.Start();
    auto sink = std::make_shared<CountingSink>();
    // Empty prompt and zero sampling.max_tokens are well-formedness rejections.
    EXPECT_TRUE(absl::IsInvalidArgument(
        engine.Submit(MakeRequest({}), sink).status()));
    GenerationRequest zero_tokens = MakeRequest();
    zero_tokens.sampling.max_tokens = 0;
    EXPECT_TRUE(absl::IsInvalidArgument(
        engine.Submit(std::move(zero_tokens), sink).status()));
    EXPECT_EQ(engine.live_requests(), 0u);
    EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
  }
}

TEST(MockEngineTest, CancelHonoredAtStepBoundaryFirstReasonWins) {  // E4
  MockEngine engine;
  MockEngine::Script script;
  script.tokens_per_step = {{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}};
  script.ignore_cancel_steps = 2;
  // A step delay keeps the "cancel after batch 1, exactly 2 more batches"
  // observation deterministic instead of racing the poll loop.
  script.step_delay = std::chrono::milliseconds{20};
  engine.SetScript(script);
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  auto handle = engine.Submit(MakeRequest(), sink);
  ASSERT_TRUE(handle.ok());
  ASSERT_TRUE(Eventually([&] { return sink->total_tokens() >= 1; }));

  // Foreign-thread cancel; a racing second reason must not win (§7.5).
  handle->Cancel(CancelReason::kClientDisconnect);
  handle->Cancel(CancelReason::kDeadline);

  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));
  EXPECT_EQ(sink->terminal()->reason, FinishReason::kAborted);
  // Exactly ignore_cancel_steps = 2 batches after the cancel was observed.
  EXPECT_EQ(sink->batches().size(), 3u);
  EXPECT_EQ(engine.cancel_calls(CancelReason::kClientDisconnect), 1u);
  EXPECT_EQ(engine.cancel_calls(CancelReason::kDeadline), 0u);

  EXPECT_TRUE(Eventually([&] { return engine.all_released(); }));
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

TEST(MockEngineTest, CancelAfterTerminalIsNoop) {
  MockEngine engine;
  engine.SetScript(SimpleScript());
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  auto handle = engine.Submit(MakeRequest(), sink);
  ASSERT_TRUE(handle.ok());
  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));

  handle->Cancel(CancelReason::kShutdown);
  EXPECT_TRUE(Eventually([&] { return engine.all_released(); }));
  EXPECT_EQ(engine.cancel_calls(CancelReason::kShutdown), 0u);
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

TEST(MockEngineTest, ImmediateCancelWithZeroIgnoreSteps) {
  MockEngine engine;
  MockEngine::Script script;
  script.tokens_per_step = {{1}, {2}};
  engine.SetScript(script);
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  auto handle = engine.Submit(MakeRequest(), sink);
  ASSERT_TRUE(handle.ok());
  handle->Cancel(CancelReason::kStopString);

  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));
  EXPECT_EQ(sink->terminal()->reason, FinishReason::kAborted);
  EXPECT_EQ(sink->terminal()->completion_tokens, sink->total_tokens());
  EXPECT_EQ(engine.cancel_calls(CancelReason::kStopString), 1u);
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

TEST(MockEngineTest, PerRequestFailureDeliversFailedTerminal) {
  MockEngine engine;
  MockEngine::Script script;
  script.tokens_per_step = {{1}, {2}, {3}};
  script.fail_at_step = 1;
  engine.SetScript(script);
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  ASSERT_TRUE(engine.Submit(MakeRequest(), sink).ok());
  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));
  EXPECT_EQ(sink->terminal()->reason, FinishReason::kError);
  EXPECT_FALSE(sink->terminal()->failure.ok());
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

TEST(MockEngineTest, FatalFailureFansOutAndRejectsSubmits) {  // E5
  MockEngine engine;
  MockEngine::Script script;
  script.tokens_per_step = {{1}, {2}, {3}};
  script.fatal_fail_at_step = 1;
  engine.SetScript(script);
  engine.Start();

  auto first = std::make_shared<CountingSink>();
  auto second = std::make_shared<CountingSink>();
  ASSERT_TRUE(engine.Submit(MakeRequest(), first).ok());
  ASSERT_TRUE(engine.Submit(MakeRequest(), second).ok());

  EXPECT_TRUE(Eventually([&] { return first->terminal().has_value() &&
                                      second->terminal().has_value(); }));
  EXPECT_EQ(first->terminal()->reason, FinishReason::kError);
  EXPECT_EQ(second->terminal()->reason, FinishReason::kError);
  // Further Submits reject with kUnavailable.
  EXPECT_TRUE(absl::IsUnavailable(
      engine.Submit(MakeRequest(), std::make_shared<CountingSink>()).status()));
  EXPECT_TRUE(Eventually([&] { return engine.all_released(); }));
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

TEST(MockEngineTest, WithheldTerminalHoldsUntilReleased) {  // R-2, §10.7
  MockEngine engine;
  MockEngine::Script script;
  script.tokens_per_step = {{1}};
  script.withhold_terminal = true;
  engine.SetScript(script);
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  auto handle = engine.Submit(MakeRequest(), sink);
  ASSERT_TRUE(handle.ok());
  const RequestId id = handle->request_id();

  // Tokens flow, but no terminal ever arrives on its own.
  EXPECT_TRUE(Eventually([&] { return sink->total_tokens() == 1; }));
  std::this_thread::sleep_for(std::chrono::milliseconds{100});
  EXPECT_FALSE(sink->terminal().has_value());
  EXPECT_EQ(engine.live_requests(), 1u);  // Permit must stay retained.

  engine.ReleaseWithheldTerminal(id);
  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));
  EXPECT_EQ(sink->terminal()->reason, FinishReason::kStopped);
  EXPECT_TRUE(Eventually([&] { return engine.all_released(); }));
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

TEST(MockEngineTest, ShutdownDeliversTerminalsToEverythingLive) {  // §10.7
  MockEngine engine;
  MockEngine::Script script;
  script.tokens_per_step = {{1}, {2}, {3}};
  script.step_delay = std::chrono::milliseconds{50};
  engine.SetScript(script);
  engine.Start();

  auto running = std::make_shared<CountingSink>();
  ASSERT_TRUE(engine.Submit(MakeRequest(), running).ok());

  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{5000}).ok());
  EXPECT_TRUE(running->terminal().has_value());
  EXPECT_EQ(running->terminal()->reason, FinishReason::kAborted);
  EXPECT_EQ(running->terminal()->completion_tokens, running->total_tokens());
  EXPECT_EQ(engine.live_requests(), 0u);
  // Intake closes with Shutdown: late Submits reject, health is kDraining.
  EXPECT_TRUE(absl::IsUnavailable(
      engine.Submit(MakeRequest(), std::make_shared<CountingSink>()).status()));

  // Shutdown is idempotent.
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());

  // A withheld request is force-released by Shutdown too.
  MockEngine withheld_engine;
  MockEngine::Script wscript;
  wscript.tokens_per_step = {{1}};
  wscript.withhold_terminal = true;
  withheld_engine.SetScript(wscript);
  withheld_engine.Start();
  auto withheld_sink = std::make_shared<CountingSink>();
  auto withheld_handle = withheld_engine.Submit(MakeRequest(), withheld_sink);
  ASSERT_TRUE(withheld_handle.ok());
  EXPECT_TRUE(Eventually([&] { return withheld_sink->total_tokens() == 1; }));
  EXPECT_TRUE(withheld_engine.Shutdown(std::chrono::milliseconds{1000}).ok());
  EXPECT_TRUE(withheld_sink->terminal().has_value());
  EXPECT_EQ(withheld_sink->terminal()->reason, FinishReason::kAborted);
}

TEST(MockEngineTest, ZeroTokenFinishDeliversFinishedWithNoTokens) {  // §6.4
  MockEngine engine;
  MockEngine::Script script;
  script.zero_token_finish = true;  // tokens_per_step stays empty.
  engine.SetScript(script);
  engine.Start();

  auto sink = std::make_shared<CountingSink>();
  ASSERT_TRUE(engine.Submit(MakeRequest(), sink).ok());
  EXPECT_TRUE(Eventually([&] { return sink->terminal().has_value(); }));
  EXPECT_EQ(sink->terminal()->reason, FinishReason::kStopped);
  EXPECT_EQ(sink->terminal()->completion_tokens, 0u);
  EXPECT_EQ(sink->total_tokens(), 0u);
  EXPECT_TRUE(engine.Shutdown(std::chrono::milliseconds{1000}).ok());
}

}  // namespace
}  // namespace inferx::server
