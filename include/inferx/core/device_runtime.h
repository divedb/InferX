#pragma once

#include <cstddef>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"

namespace inferx {

struct DeviceMemoryInfo {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
};

enum class CopyKind { kHostToDevice, kDeviceToHost, kDeviceToDevice };

struct RuntimeCapabilities {
  bool graph_capture = false;
  bool pinned_host_memory = false;
  bool device_sampling = false;
};

// Mechanical device services. Compute remains behind the ops API; this
// interface exists so orchestration never needs a vendor runtime header.
class DeviceRuntime {
 public:
  virtual ~DeviceRuntime() = default;

  virtual DeviceKind kind() const = 0;
  virtual int DeviceCount() const = 0;
  virtual Status SetDevice(DeviceId device) = 0;
  virtual StatusOr<DeviceMemoryInfo> GetMemoryInfo(DeviceId device) = 0;

  virtual StatusOr<void*> Allocate(DeviceId device, size_t bytes) = 0;
  virtual Status Free(DeviceId device, void* ptr) = 0;
  virtual StatusOr<void*> AllocatePinnedHost(size_t bytes) = 0;
  virtual Status FreePinnedHost(void* ptr) = 0;

  virtual Status Copy(void* dst, const void* src, size_t bytes,
                      CopyKind kind) = 0;
  virtual Status CopyAsync(void* dst, const void* src, size_t bytes,
                           CopyKind kind, Stream stream) = 0;

  virtual StatusOr<Stream> CreateStream(DeviceId device) = 0;
  virtual Status DestroyStream(Stream stream) = 0;
  virtual Status SynchronizeStream(Stream stream) = 0;
  virtual StatusOr<DeviceEvent> CreateEvent(bool timing) = 0;
  virtual Status DestroyEvent(DeviceEvent event) = 0;
  virtual Status RecordEvent(DeviceEvent event, Stream stream) = 0;
  virtual Status SynchronizeEvent(DeviceEvent event) = 0;
  virtual StatusOr<bool> QueryEvent(DeviceEvent event) = 0;
  virtual StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) = 0;
  virtual StatusOr<bool> IsCapturing(Stream stream) = 0;

  virtual Status BeginCapture(Stream stream) = 0;
  virtual StatusOr<GraphExec> EndCaptureAndInstantiate(Stream stream) = 0;
  virtual Status LaunchGraph(GraphExec graph, Stream stream) = 0;
  virtual Status DestroyGraph(GraphExec graph) = 0;

  virtual const RuntimeCapabilities& capabilities() const = 0;
};

// Returns the process-wide runtime for `device.kind`. The returned object has
// static lifetime. Unsupported accelerator kinds fail instead of silently
// falling back to host memory.
StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device);

}  // namespace inferx
