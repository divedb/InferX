#include <folly/coro/BlockingWait.h>
#include <gtest/gtest.h>

#include <string>

#include "inferx/engine/engine.h"

namespace inferx::engine {
namespace {

TEST(GenerationTest, DeliversEventsAndTerminalStateInOrder) {
  Generation generation;
  generation.Emit({.text = "token", .generated = 1});
  generation.Finish(scheduler::FinishReason::kMaxTokens, 1);

  auto token = folly::coro::blockingWait(generation.Next());
  ASSERT_TRUE(token.ok());
  ASSERT_TRUE(token->has_value());
  EXPECT_EQ((*token)->text, "token");
  EXPECT_FALSE((*token)->done);

  auto terminal = folly::coro::blockingWait(generation.Next());
  ASSERT_TRUE(terminal.ok());
  ASSERT_TRUE(terminal->has_value());
  EXPECT_TRUE((*terminal)->done);
  EXPECT_EQ((*terminal)->reason, scheduler::FinishReason::kMaxTokens);
  EXPECT_EQ((*terminal)->generated, 1);

  auto closed = folly::coro::blockingWait(generation.Next());
  ASSERT_TRUE(closed.ok());
  EXPECT_FALSE(closed->has_value());
}

TEST(GenerationTest, CancelClosesStreamAndIsIdempotent) {
  Generation generation;

  generation.Cancel();
  generation.Cancel();
  generation.Emit({.text = "ignored"});
  generation.Finish(scheduler::FinishReason::kCancelled, 0);

  EXPECT_TRUE(generation.cancelled());
  auto closed = folly::coro::blockingWait(generation.Next());
  ASSERT_TRUE(closed.ok());
  EXPECT_FALSE(closed->has_value());
}

TEST(GenerationTest, ReportsBoundedBufferOverflow) {
  Generation generation;
  for (int i = 0; i < 257; ++i) {
    generation.Emit({.text = std::to_string(i), .generated = i + 1});
  }

  for (int i = 0; i < 256; ++i) {
    auto event = folly::coro::blockingWait(generation.Next());
    ASSERT_TRUE(event.ok());
    ASSERT_TRUE(event->has_value());
    EXPECT_EQ((*event)->generated, i + 1);
  }

  auto overflow = folly::coro::blockingWait(generation.Next());
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace inferx::engine
