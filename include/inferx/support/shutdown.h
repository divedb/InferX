#pragma once

#include <atomic>
#include <functional>
#include <thread>

namespace inferx {

// Blocks SIGINT and SIGTERM for the calling thread. Call from main() before
// spawning any threads (in particular before engine or server creation) so
// every thread inherits the mask and delivery is confined to the
// ShutdownSignalWaiter's sigwait.
void BlockShutdownSignals();

// A thread that waits for a blocked SIGINT/SIGTERM and then invokes
// `on_signal` as ordinary code, free of async-signal-safety restrictions --
// the callback may log, lock, and call Stop() on a server. Requires
// BlockShutdownSignals() to have run first.
//
// The destructor disarms the callback and unblocks the wait, so destroying
// the waiter is safe whether or not a signal ever arrived.
class ShutdownSignalWaiter {
 public:
  explicit ShutdownSignalWaiter(std::function<void(int)> on_signal);
  ~ShutdownSignalWaiter();

  ShutdownSignalWaiter(const ShutdownSignalWaiter&) = delete;
  ShutdownSignalWaiter& operator=(const ShutdownSignalWaiter&) = delete;

 private:
  std::atomic<bool> disarmed_{false};
  std::thread thread_;
};

}  // namespace inferx
