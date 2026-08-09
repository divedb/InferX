#include "inferx/support/shutdown.h"

#include <signal.h>
#include <unistd.h>

#include <utility>

namespace inferx {
namespace {

sigset_t ShutdownSet() {
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGINT);
  sigaddset(&set, SIGTERM);
  return set;
}

}  // namespace

void BlockShutdownSignals() {
  sigset_t set = ShutdownSet();
  pthread_sigmask(SIG_BLOCK, &set, nullptr);
}

ShutdownSignalWaiter::ShutdownSignalWaiter(std::function<void(int)> on_signal)
    : thread_([this, on_signal = std::move(on_signal)] {
        sigset_t set = ShutdownSet();
        int signum = 0;
        if (sigwait(&set, &signum) != 0) return;
        if (!disarmed_.load()) on_signal(signum);
      }) {}

ShutdownSignalWaiter::~ShutdownSignalWaiter() {
  disarmed_.store(true);
  // Process-directed so a thread still parked in sigwait consumes it. A
  // thread-directed raise() would not work: it pends on the calling thread,
  // which keeps it blocked forever.
  kill(getpid(), SIGTERM);
  thread_.join();
  // If the thread had already left sigwait (a real signal arrived first), the
  // unblock signal above is still pending -- as is any repeated Ctrl-C. Drain
  // them so they cannot leak into a later waiter or an exec'd child.
  sigset_t set = ShutdownSet();
  const timespec zero{0, 0};
  while (sigtimedwait(&set, nullptr, &zero) > 0) {
  }
}

}  // namespace inferx
