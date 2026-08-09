#include "inferx/support/crash.h"

#include <csignal>

#include "gtest/gtest.h"

namespace inferx {
namespace {

TEST(CrashTest, ReportsFatalSignal) {
  ASSERT_DEATH(
      {
        InstallCrashHandler("crash_test");
        std::raise(SIGABRT);
      },
      "SIGABRT");
}

}  // namespace
}  // namespace inferx
