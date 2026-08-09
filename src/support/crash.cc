#include "inferx/support/crash.h"

#include "absl/debugging/failure_signal_handler.h"
#include "absl/debugging/symbolize.h"

namespace inferx {

void InstallCrashHandler(const char* argv0) {
  absl::InitializeSymbolizer(argv0);
  absl::FailureSignalHandlerOptions options;
  absl::InstallFailureSignalHandler(options);
}

}  // namespace inferx
