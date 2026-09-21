#pragma once

#include <cstddef>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"

namespace inferx {

/// \brief Free and total device memory, in bytes.
struct DeviceMemoryInfo {
  size_t free_bytes = 0;   ///< Bytes currently unallocated.
  size_t total_bytes = 0;  ///< Total capacity of the device.
};

/// \brief The direction of a copy.
enum class CopyKind {
  kHostToDevice,    ///< From host memory to device memory.
  kDeviceToHost,    ///< From device memory to host memory.
  kDeviceToDevice,  ///< Between two device allocations.
};

/// \brief Optional services a runtime may provide.
struct RuntimeCapabilities {
  bool graph_capture = false;       ///< CUDA-graph capture is available.
  bool pinned_host_memory = false;  ///< Page-locked host allocation is available.
  bool device_sampling = false;     ///< Hardware counters are available.
};

/// \brief Mechanical services bound to one device.
///
/// Compute remains behind the ops API; this interface exists so orchestration
/// never needs a vendor runtime header. Operations select the bound device on
/// the calling thread as needed. Streams, events, and graphs passed to this
/// runtime must belong to its device.
class DeviceRuntime {
 public:
  virtual ~DeviceRuntime() = default;

  DeviceRuntime(const DeviceRuntime&) = delete;
  DeviceRuntime& operator=(const DeviceRuntime&) = delete;

  /// \brief Returns the device this runtime is permanently bound to.
  DeviceId device() const { return device_; }
  /// \brief Returns the kind of the bound device.
  DeviceKind kind() const { return device_.kind; }
  /// \brief Returns the number of devices of this runtime's kind.
  virtual int DeviceCount() const = 0;
  /// \brief Makes the bound device current on the calling thread.
  ///
  /// Used before compute or vendor calls outside this interface. Runtime
  /// operations activate their device themselves when needed.
  /// \return       OK, or an error status.
  virtual Status Activate() = 0;
  /// \brief Returns the free/total memory of the bound device.
  ///
  /// \return       The memory info, or an error status.
  virtual StatusOr<DeviceMemoryInfo> GetMemoryInfo() = 0;

  /// \brief Allocates `bytes` of memory on the bound device.
  ///
  /// \param bytes  Number of bytes to allocate.
  /// \return       The allocation, or an error status.
  virtual StatusOr<void*> Allocate(size_t bytes) = 0;
  /// \brief Frees device memory returned by Allocate().
  ///
  /// \param ptr    The block to free.
  /// \return       OK, or an error status.
  virtual Status Free(void* ptr) = 0;
  /// \brief Allocates page-locked host memory of `bytes`.
  ///
  /// \param bytes Number of bytes to allocate.
  /// \return      The allocation, or an error status.
  virtual StatusOr<void*> AllocatePinnedHost(size_t bytes) = 0;
  /// \brief Frees page-locked host memory.
  ///
  /// \param ptr The block to free.
  /// \return    OK, or an error status.
  virtual Status FreePinnedHost(void* ptr) = 0;

  /// \brief Copies `bytes` synchronously.
  ///
  /// \param dst   Destination address.
  /// \param src   Source address.
  /// \param bytes Number of bytes to copy.
  /// \param kind  Direction of the copy.
  /// \return      OK, or an error status.
  virtual Status Copy(void* dst, const void* src, size_t bytes, CopyKind kind) = 0;
  /// \brief Copies `bytes` asynchronously on `stream`.
  ///
  /// \param dst    Destination address.
  /// \param src    Source address.
  /// \param bytes  Number of bytes to copy.
  /// \param kind   Direction of the copy.
  /// \param stream Stream to enqueue the copy on.
  /// \return       OK, or an error status.
  virtual Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                           Stream stream) = 0;

  /// \brief Creates a stream on the bound device.
  ///
  /// \return       The stream, or an error status.
  virtual StatusOr<Stream> CreateStream() = 0;
  /// \brief Destroys a stream.
  ///
  /// \param stream The stream to destroy.
  /// \return       OK, or an error status.
  virtual Status DestroyStream(Stream stream) = 0;
  /// \brief Blocks until all work enqueued on `stream` completes.
  ///
  /// \param stream The stream to wait on.
  /// \return       OK, or an error status.
  virtual Status SynchronizeStream(Stream stream) = 0;
  /// \brief Creates an event, optionally enabled for timing.
  ///
  /// \param timing True to record elapsed time between events.
  /// \return       The event, or an error status.
  virtual StatusOr<DeviceEvent> CreateEvent(bool timing) = 0;
  /// \brief Destroys an event.
  ///
  /// \param event The event to destroy.
  /// \return      OK, or an error status.
  virtual Status DestroyEvent(DeviceEvent event) = 0;
  /// \brief Records `event` on `stream`.
  ///
  /// \param event  The event to record.
  /// \param stream The stream to record it on.
  /// \return       OK, or an error status.
  virtual Status RecordEvent(DeviceEvent event, Stream stream) = 0;
  /// \brief Blocks the host until `event` completes.
  ///
  /// \param event The event to wait on.
  /// \return      OK, or an error status.
  virtual Status SynchronizeEvent(DeviceEvent event) = 0;
  /// \brief Checks whether `event` has completed.
  ///
  /// \param event The event to query.
  /// \return      True when complete, or an error status.
  virtual StatusOr<bool> QueryEvent(DeviceEvent event) = 0;
  /// \brief Returns the elapsed time between two recorded events.
  ///
  /// \param start The earlier event; both must be timing events.
  /// \param end   The later event.
  /// \return      Elapsed milliseconds, or an error status.
  virtual StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) = 0;
  /// \brief Checks whether `stream` is currently capturing a graph.
  ///
  /// \param stream The stream to query.
  /// \return      True when capturing, or an error status.
  virtual StatusOr<bool> IsCapturing(Stream stream) = 0;

  /// \brief Begins graph capture on `stream`.
  ///
  /// \param stream The stream to capture.
  /// \return       OK, or an error status.
  virtual Status BeginCapture(Stream stream) = 0;
  /// \brief Ends capture and instantiates the captured graph.
  ///
  /// \param stream The stream whose capture ends.
  /// \return       The instantiated graph, or an error status.
  virtual StatusOr<GraphExec> EndCaptureAndInstantiate(Stream stream) = 0;
  /// \brief Enqueues the instantiated graph on `stream`.
  ///
  /// \param graph  The graph to launch.
  /// \param stream The stream to launch it on.
  /// \return       OK, or an error status.
  virtual Status LaunchGraph(GraphExec graph, Stream stream) = 0;
  /// \brief Destroys an instantiated graph.
  ///
  /// \param graph The graph to destroy.
  /// \return      OK, or an error status.
  virtual Status DestroyGraph(GraphExec graph) = 0;

  /// \brief Returns the services this runtime provides.
  virtual const RuntimeCapabilities& capabilities() const = 0;

 protected:
  explicit DeviceRuntime(DeviceId device) : device_(device) {}

 private:
  const DeviceId device_;
};

/// \brief Returns the process-wide runtime bound to `device`.
///
/// Repeated lookups of the same device return the same object; distinct devices
/// return distinct objects. Lookup is thread-safe and the returned object has
/// static lifetime. Invalid ordinals and unsupported accelerator kinds fail
/// instead of silently falling back to another device or host memory.
///
/// \param device The device whose runtime to return.
/// \return       The runtime, or an error status for invalid/unsupported devices.
StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device);

}  // namespace inferx
