#include <cstdio>

#include "inferx/support/crash.h"

namespace inferx::example {

[[gnu::noinline, gnu::visibility("default")]] int TriggerSegmentationFault() {
  std::fputs("Triggering the example SIGSEGV...\n", stderr);
  volatile int* invalid_address = nullptr;
  *invalid_address = 1;

  return 0;
}

[[gnu::noinline, gnu::visibility("default")]] int RunWorker() {
  return TriggerSegmentationFault() + 1;
}

[[gnu::noinline, gnu::visibility("default")]] int RunRequest() {
  return RunWorker() + 1;
}

}  // namespace inferx::example

int main(int argc, char** argv) {
  inferx::InstallCrashHandler(argv[0]);

  return inferx::example::RunRequest() + argc;
}
