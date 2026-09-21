#pragma once

#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/stream.h"

namespace inferx::ops {

/// \brief Where and how an op executes: one device runtime plus the stream
///        that orders its work.
///
/// A context is a borrowed handle, not an owner: the caller guarantees the
/// runtime and stream outlive every call made with the context. Ops select
/// the context's device on the calling thread, enqueue their work on the
/// context's stream, and hold no state between calls.
class ExecutionContext {
 public:
  /// \brief Constructs a context over an existing runtime and stream.
  ///
  /// \param runtime The runtime bound to the execution device.
  /// \param stream  A stream created by that runtime.
  ExecutionContext(DeviceRuntime& runtime, Stream stream)
      : runtime_(runtime), stream_(stream) {}

  /// \brief Returns the runtime of the execution device.
  DeviceRuntime& runtime() const { return runtime_; }
  /// \brief Returns the stream ops enqueue their work on.
  Stream stream() const { return stream_; }
  /// \brief Returns the device ops execute on.
  DeviceId device() const { return runtime_.device(); }

 private:
  DeviceRuntime& runtime_;
  Stream stream_;
};

}  // namespace inferx::ops
