#include "inferx/support/scope_exit.h"

#include <gtest/gtest.h>

#include <utility>

namespace inferx {
namespace {

TEST(ScopeExitTest, ExecutesWhenLeavingScope) {
  bool executed = false;
  {
    const auto guard = MakeScopeExit([&] { executed = true; });
    EXPECT_FALSE(executed);
  }
  EXPECT_TRUE(executed);
}

TEST(ScopeExitTest, ReleaseDisengagesCleanup) {
  bool executed = false;
  {
    auto guard = MakeScopeExit([&] { executed = true; });
    guard.Release();
  }
  EXPECT_FALSE(executed);
}

TEST(ScopeExitTest, MovingTransfersCleanup) {
  int executions = 0;
  {
    auto first = MakeScopeExit([&] { ++executions; });
    const auto second = std::move(first);
  }
  EXPECT_EQ(executions, 1);
}

}  // namespace
}  // namespace inferx
