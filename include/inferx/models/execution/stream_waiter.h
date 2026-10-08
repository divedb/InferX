#ifndef INFERX_MODELS_EXECUTION_STREAM_WAITER_H_
#define INFERX_MODELS_EXECUTION_STREAM_WAITER_H_

#include <functional>

#include "inferx/models/model_runner.h"  // Status, DeviceRuntime, Stream, DeviceEvent

namespace inferx::models::execution {

/// \brief Waits for a stream to drain, optionally polling a health check so a
///        failed peer rank cannot leave this rank blocked forever.
///
/// Owns the completion event used for polling. Not thread safe.
class StreamWaiter {
 public:
  StreamWaiter() = default;
  StreamWaiter(const StreamWaiter&) = delete;
  StreamWaiter& operator=(const StreamWaiter&) = delete;
  ~StreamWaiter() { Reset(); }

  /// Binds the runtime and stream. Both must outlive this object.
  void Bind(DeviceRuntime* runtime, const Stream* stream) {
    runtime_ = runtime;
    stream_ = stream;
  }

  /// Installs (or clears) the health check. Creates the polling event once.
  /// If the event cannot be created, waiting falls back to a plain sync.
  void SetHealthCheck(std::function<Status()> check);

  /// Blocks until all work queued on the stream has completed.
  Status Wait();

  /// Destroys the event. The runtime's device must be active. Idempotent.
  void Reset();

 private:
  DeviceRuntime* runtime_ = nullptr;
  const Stream* stream_ = nullptr;
  DeviceEvent event_;
  std::function<Status()> check_health_;
};

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_STREAM_WAITER_H_
