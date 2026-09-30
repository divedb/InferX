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
class OpContext {
 public:
  /// \brief Constructs a context over an existing runtime and stream.
  ///
  /// \param runtime The runtime bound to the execution device.
  /// \param stream  A stream created by that runtime.
  OpContext(DeviceRuntime& runtime, Stream stream) : runtime_(runtime), stream_(stream) {}

  /// \brief Returns the runtime ops execute on, which is bound to a device.
  ///
  /// \return The device runtime.
  DeviceRuntime& Runtime() const noexcept { return runtime_; }

  /// \brief Returns the stream ops enqueue work on, which is bound to a device.
  ///
  /// \return The stream.
  Stream GetStream() const noexcept { return stream_; }

  /// \brief Returns the device ops execute on.
  ///
  /// \return The device.
  DeviceId Device() const noexcept { return runtime_.device(); }

 private:
  DeviceRuntime& runtime_;
  Stream stream_;
};

}  // namespace inferx::ops
