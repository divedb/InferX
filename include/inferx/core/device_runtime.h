#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"

namespace inferx {

/// \brief Free and total device memory, in bytes.
struct DeviceMemoryInfo {
  size_t free_bytes = 0;   ///< Bytes currently unallocated.
  size_t total_bytes = 0;  ///< Total capacity of the device.
};

/// \brief Static properties of a device, for kernel and tile selection.
struct DeviceProperties {
  std::string name;               ///< Driver-reported device name.
  int arch_major = 0;             ///< Architecture major (CUDA: compute major).
  int arch_minor = 0;             ///< Architecture minor (CUDA: compute minor).
  int multiprocessor_count = 0;   ///< SMs / CUs / AI cores.
  size_t total_memory_bytes = 0;  ///< Total device memory.
};

/// \brief The direction of a copy.
enum class CopyKind {
  kHostToHost,      ///< Between two host allocations.
  kHostToDevice,    ///< From host memory to device memory.
  kDeviceToHost,    ///< From device memory to host memory.
  kDeviceToDevice,  ///< Between two allocations on the same device.
};

/// \brief Scheduling priority of a stream.
enum class StreamPriority {
  kNormal,
  kHigh,  ///< May run ahead of normal-priority work on the same device.
};

/// \brief How strictly graph capture restricts other activity.
enum class CaptureMode {
  kGlobal,       ///< Unsafe runtime calls on any thread are illegal during capture.
  kThreadLocal,  ///< Only calls on the capturing thread are restricted.
  kRelaxed,      ///< No restrictions; the caller guarantees safety.
};

/// \brief Move-only owner of a runtime handle.
///
/// Destroys the handle through the API that created it, so callers cannot leak
/// or double-free. The creating API (and so its DeviceRuntime) must outlive
/// the owner. Use `Get()` to pass the non-owning handle to runtime calls.
///
/// \tparam Handle The non-owning handle type, e.g. `DeviceStream`.
/// \tparam Api    The API class that created and can destroy `Handle`.
template <typename Handle, typename Api>
class Owned {
 public:
  Owned() = default;

  /// \brief Adopts `handle`, to be destroyed through `api`.
  Owned(Handle handle, Api* api) : handle_(handle), api_(api) {}

  Owned(Owned&& other) noexcept
      : handle_(std::exchange(other.handle_, Handle{})),
        api_(std::exchange(other.api_, nullptr)) {}

  Owned& operator=(Owned&& other) noexcept {
    if (this != &other) {
      Reset();
      handle_ = std::exchange(other.handle_, Handle{});
      api_ = std::exchange(other.api_, nullptr);
    }

    return *this;
  }

  Owned(const Owned&) = delete;
  Owned& operator=(const Owned&) = delete;

  ~Owned() { Reset(); }

  /// \brief The non-owning handle. Valid until this owner is reset or destroyed.
  Handle Get() const { return handle_; }

  /// \brief True if this owner holds a handle.
  explicit operator bool() const { return handle_ != Handle{}; }

  /// \brief Gives up ownership and returns the handle; the caller must destroy it.
  Handle Release() {
    api_ = nullptr;
    return std::exchange(handle_, Handle{});
  }

  /// \brief Destroys the held handle, if any.
  void Reset() {
    if (handle_ != Handle{} && api_ != nullptr) api_->Destroy(handle_);

    handle_ = Handle{};
    api_ = nullptr;
  }

 private:
  Handle handle_{};
  Api* api_ = nullptr;
};

class StreamApi;
class GraphApi;

using OwnedStream = Owned<DeviceStream, StreamApi>;
using OwnedEvent = Owned<DeviceEvent, StreamApi>;
using OwnedGraphExec = Owned<DeviceGraphExec, GraphApi>;

/// \brief Memory allocation, host pinning, and copies. Always available.
///
/// This is the raw, mechanical layer (`cudaMalloc` and friends). Caching and
/// pooling belong in an allocator on top of it. Runtimes without page-locked
/// memory (the CPU) treat pinned allocations as plain aligned allocations and
/// registration as a successful no-op, so callers need not branch.
class MemoryApi {
 public:
  virtual ~MemoryApi() = default;

  /// \brief Returns the free and total memory of the bound device.
  ///
  /// \return The memory info, or an error status.
  virtual StatusOr<DeviceMemoryInfo> GetMemoryInfo() = 0;

  /// \brief Allocates device memory.
  ///
  /// `Allocate(0, ...)` returns a null pointer and OK. Illegal while a stream
  /// of this device is capturing a graph.
  ///
  /// \param bytes     Number of bytes to allocate.
  /// \param alignment Required alignment in bytes; a power of two. Use
  ///                  `kTensorAlignment` unless the caller knows better.
  /// \return          The allocation, or an error status.
  virtual StatusOr<void*> Allocate(size_t bytes, size_t alignment) = 0;

  /// \brief Frees device memory returned by Allocate().
  ///
  /// `Free(nullptr)` is a no-op. Does not wait for work still using `ptr`;
  /// synchronize first. Illegal while a stream of this device is capturing.
  ///
  /// \param ptr The block to free.
  /// \return    OK, or an error status.
  virtual Status Free(void* ptr) = 0;

  /// \brief Allocates page-locked host memory.
  ///
  /// Asynchronous copies overlap with compute only when the host side is
  /// pinned; a copy from pageable memory may block the host.
  ///
  /// \param bytes     Number of bytes to allocate.
  /// \param alignment Required alignment in bytes; a power of two.
  /// \return          The allocation, or an error status.
  virtual StatusOr<void*> AllocatePinnedHost(size_t bytes, size_t alignment) = 0;

  /// \brief Frees memory returned by AllocatePinnedHost().
  ///
  /// \param ptr The block to free.
  /// \return    OK, or an error status.
  virtual Status FreePinnedHost(void* ptr) = 0;

  /// \brief Page-locks an existing host range, such as mmap'd weights.
  ///
  /// Registration works on whole pages: `ptr` and `bytes` should be multiples
  /// of `kPageAlignment`, or neighbouring memory is pinned too.
  ///
  /// \param ptr   Start of the host range.
  /// \param bytes Length of the range.
  /// \return      OK, or an error status.
  virtual Status RegisterHost(void* ptr, size_t bytes) = 0;

  /// \brief Releases a registration made by RegisterHost().
  ///
  /// \param ptr Start of the registered range.
  /// \return    OK, or an error status.
  virtual Status UnregisterHost(void* ptr) = 0;

  /// \brief Copies `bytes` synchronously; returns when the copy is complete.
  ///
  /// Illegal while a stream of this device is capturing.
  ///
  /// \param dst   Destination address.
  /// \param src   Source address.
  /// \param bytes Number of bytes to copy.
  /// \param kind  Direction of the copy.
  /// \return      OK, or an error status.
  virtual Status Copy(void* dst, const void* src, size_t bytes, CopyKind kind) = 0;

  /// \brief Copies `bytes` asynchronously, ordered on `stream`.
  ///
  /// \param dst    Destination address.
  /// \param src    Source address.
  /// \param bytes  Number of bytes to copy.
  /// \param kind   Direction of the copy.
  /// \param stream The stream to enqueue the copy on.
  /// \return       OK, or an error status.
  virtual Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                           DeviceStream stream) = 0;

  /// \brief Fills `bytes` bytes at `dst` with `value`, ordered on `stream`.
  ///
  /// \param dst    Destination address.
  /// \param value  The byte value to store.
  /// \param bytes  Number of bytes to fill.
  /// \param stream The stream to enqueue the fill on.
  /// \return       OK, or an error status.
  virtual Status MemsetAsync(void* dst, uint8_t value, size_t bytes, DeviceStream stream) = 0;
};

/// \brief Streams, events, and synchronization. Always available.
///
/// Runtimes without real asynchrony (the CPU) may complete work eagerly and
/// hand out inert handles; the ordering contracts below still hold.
class StreamApi {
 public:
  virtual ~StreamApi() = default;

  /// \brief Creates a stream on the bound device.
  ///
  /// \param priority Scheduling priority of the new stream.
  /// \return         The owning stream, or an error status.
  virtual StatusOr<OwnedStream> CreateStream(StreamPriority priority) = 0;

  /// \brief Blocks the host until all work enqueued on `stream` completes.
  ///
  /// \param stream The stream to wait on.
  /// \return       OK, or the first error from work enqueued on it.
  virtual Status SynchronizeStream(DeviceStream stream) = 0;

  /// \brief Checks, without blocking, whether `stream` has drained.
  ///
  /// \param stream The stream to query.
  /// \return       True when all enqueued work completed, or an error status.
  virtual StatusOr<bool> QueryStream(DeviceStream stream) = 0;

  /// \brief Makes later work on `stream` wait until `event` completes.
  ///
  /// The host does not block. This is the primitive for ordering work across
  /// streams, such as a copy stream feeding a compute stream.
  ///
  /// \param stream The stream that waits.
  /// \param event  The event to wait for; must have been recorded.
  /// \return       OK, or an error status.
  virtual Status WaitEvent(DeviceStream stream, DeviceEvent event) = 0;

  /// \brief Blocks the host until all work on the bound device completes.
  ///
  /// \return OK, or the first error from any stream of this device.
  virtual Status SynchronizeDevice() = 0;

  /// \brief Creates an event, optionally enabled for timing.
  ///
  /// Timing events are slower than non-timing ones; request timing only when
  /// ElapsedMs() will be used.
  ///
  /// \param timing True to allow measuring elapsed time between events.
  /// \return       The owning event, or an error status.
  virtual StatusOr<OwnedEvent> CreateEvent(bool timing) = 0;

  /// \brief Records `event` on `stream`.
  ///
  /// The event completes when all work enqueued on `stream` before this call
  /// completes.
  ///
  /// \param event  The event to record.
  /// \param stream The stream to record it on.
  /// \return       OK, or an error status.
  virtual Status RecordEvent(DeviceEvent event, DeviceStream stream) = 0;

  /// \brief Blocks the host until `event` completes.
  ///
  /// \param event The event to wait on.
  /// \return      OK, or an error status.
  virtual Status SynchronizeEvent(DeviceEvent event) = 0;

  /// \brief Checks, without blocking, whether `event` has completed.
  ///
  /// \param event The event to query.
  /// \return      True when complete, or an error status.
  virtual StatusOr<bool> QueryEvent(DeviceEvent event) = 0;

  /// \brief Returns the elapsed time between two recorded events.
  ///
  /// \param start The earlier event; both must be timing events and recorded.
  /// \param end   The later event.
  /// \return      Elapsed milliseconds, or an error status.
  virtual StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) = 0;

 protected:
  /// \brief Destroys a stream. Called only by `OwnedStream`; never throws.
  ///        Failures are the implementation's to log, since a destructor cannot
  ///        report them.
  ///
  /// \param stream The stream to destroy.
  virtual void Destroy(DeviceStream stream) noexcept = 0;

  /// \brief Destroys an event. Called only by `OwnedEvent`; never throws.
  ///
  /// \param event The event to destroy.
  virtual void Destroy(DeviceEvent event) noexcept = 0;

 private:
  friend class Owned<DeviceStream, StreamApi>;
  friend class Owned<DeviceEvent, StreamApi>;
};

/// \brief Graph capture and replay. Optional: see DeviceRuntime::graphs().
///
/// While a stream is capturing, do not call MemoryApi::Allocate/Free/Copy, any
/// Synchronize*, or any query that waits on the device: they are illegal or
/// invalidate the capture. Use a pool and the asynchronous variants instead.
class GraphApi {
 public:
  virtual ~GraphApi() = default;

  /// \brief Checks whether `stream` is currently capturing a graph.
  ///
  /// \param stream The stream to query.
  /// \return       True when capturing, or an error status.
  virtual StatusOr<bool> IsCapturing(DeviceStream stream) = 0;

  /// \brief Begins graph capture on `stream`.
  ///
  /// \param stream The stream to capture.
  /// \param mode   How strictly other calls are restricted during capture.
  /// \return       OK, or an error status.
  virtual Status BeginCapture(DeviceStream stream, CaptureMode mode) = 0;

  /// \brief Ends capture and instantiates the captured graph.
  ///
  /// Capture always ends, even on failure; after an error the stream is no
  /// longer capturing and no graph is returned.
  ///
  /// \param stream The stream whose capture ends.
  /// \return       The owning instantiated graph, or an error status.
  virtual StatusOr<OwnedGraphExec> EndCaptureAndInstantiate(DeviceStream stream) = 0;

  /// \brief Ends capture and updates `graph` in place with the new capture.
  ///
  /// Cheaper than re-instantiating when only kernel arguments or pointers
  /// changed. Capture always ends. Fails if the topology differs; the caller
  /// should then fall back to EndCaptureAndInstantiate(). Backends that cannot
  /// update return an "unimplemented" error status.
  ///
  /// \param stream The stream whose capture ends.
  /// \param graph  The instantiated graph to update.
  /// \return       OK, or an error status.
  virtual Status EndCaptureAndUpdate(DeviceStream stream, DeviceGraphExec graph) = 0;

  /// \brief Enqueues the instantiated graph on `stream`.
  ///
  /// \param graph  The graph to launch.
  /// \param stream The stream to launch it on.
  /// \return       OK, or an error status.
  virtual Status LaunchGraph(DeviceGraphExec graph, DeviceStream stream) = 0;

 protected:
  /// \brief Destroys an instantiated graph. Called only by `OwnedGraphExec`.
  ///
  /// \param graph The graph to destroy.
  virtual void Destroy(DeviceGraphExec graph) noexcept = 0;

 private:
  friend class Owned<DeviceGraphExec, GraphApi>;
};

/// \brief Direct transfers to other devices of the same type. Optional: see
///        DeviceRuntime::peer().
class PeerApi {
 public:
  virtual ~PeerApi() = default;

  /// \brief Checks whether the bound device can reach `peer` directly.
  ///
  /// \param peer A device of this runtime's type.
  /// \return     True if direct access works, or an error status.
  virtual StatusOr<bool> CanAccess(Device peer) = 0;

  /// \brief Copies between two devices of this runtime's type, ordered on `stream`.
  ///
  /// `stream` must belong to this runtime's device, which must be one of the
  /// two endpoints. Whether the transfer is direct or staged is up to the
  /// implementation; CanAccess() says which to expect.
  ///
  /// \param dst        Destination address on `dst_device`.
  /// \param dst_device The destination device.
  /// \param src        Source address on `src_device`.
  /// \param src_device The source device.
  /// \param bytes      Number of bytes to copy.
  /// \param stream     The stream to enqueue the copy on.
  /// \return           OK, or an error status.
  virtual Status CopyAsync(void* dst, Device dst_device, const void* src, Device src_device,
                           size_t bytes, DeviceStream stream) = 0;
};

/// \brief Mechanical services bound to one device, grouped by capability.
///
/// Compute remains behind the ops API; this interface exists so orchestration
/// never needs a vendor runtime header. The optional groups are nullable
/// accessors, so the presence of a capability is visible in the type:
///
/// \code
///   if (GraphApi* g = rt->graphs()) { /* capture path */ } else { /* eager path */ }
/// \endcode
///
/// The returned API objects are owned by the runtime and live as long as it.
class DeviceRuntime {
 public:
  virtual ~DeviceRuntime() = default;

  DeviceRuntime(const DeviceRuntime&) = delete;
  DeviceRuntime& operator=(const DeviceRuntime&) = delete;

  /// \brief The device this runtime is permanently bound to.
  Device device() const { return device_; }

  /// \brief Returns static properties of the bound device.
  ///
  /// \return The properties, or an error status.
  virtual StatusOr<DeviceProperties> GetProperties() = 0;

  /// \brief Makes the bound device current on the calling thread.
  ///
  /// Needed only before compute or vendor calls made outside this interface.
  ///
  /// \return OK, or an error status.
  virtual Status Activate() = 0;

  /// \brief Memory, pinning and copies. Never null.
  ///
  /// \return The memory API.
  virtual MemoryApi& Memory() = 0;

  /// \brief Streams, events and synchronization. Never null.
  ///
  /// \return The stream API.
  virtual StreamApi& Streams() = 0;

  /// \brief Graph capture and replay, or null if unsupported.
  ///
  /// \return The graph API, or null if the backend does not support graphs.
  virtual GraphApi* Graphs() { return nullptr; }

  /// \brief Peer-to-peer transfers, or null if unsupported.
  ///
  /// \return The peer API, or null if the backend does not support direct transfers.
  virtual PeerApi* Peer() { return nullptr; }

 protected:
  explicit DeviceRuntime(Device device) : device_(device) {}

 private:
  const Device device_;
};

/// \brief The process-wide entry point for one backend (device type).
///
/// Answers questions about the backend as a whole, which no single
/// DeviceRuntime can: how many devices exist, and the runtime for ordinal N.
/// A provider exists only if its backend was built; there are no `#ifdef`s at
/// call sites. See RegisterProvider().
class RuntimeProvider {
 public:
  virtual ~RuntimeProvider() = default;

  /// \brief The device type this provider serves.
  ///
  /// \return The device type.
  virtual DeviceType GetDeviceType() const = 0;

  /// \brief Returns the number of devices of this type attached right now.
  ///
  /// 0 means the backend is linked but no hardware is present.
  /// An error status means the driver or runtime itself failed.
  ///
  /// \return The device count, or an error status.
  virtual StatusOr<int> DeviceCount() = 0;

  /// \brief Returns the runtime bound to the device at `ordinal`.
  ///
  /// Repeated calls with the same ordinal return the same object; the object
  /// has static lifetime. Fails for ordinals outside [0, DeviceCount()).
  ///
  /// \param ordinal The device ordinal.
  /// \return        The runtime, or an error status.
  virtual StatusOr<DeviceRuntime*> GetRuntime(int ordinal) = 0;
};

/// \brief Registers a backend's provider. Called once per backend at startup.
///
/// The registry is a function-local static, so registration from static
/// initializers in any translation unit is safe. A second provider for the
/// same type is an error.
///
/// \param provider The provider to register; the registry takes ownership.
/// \return         OK, or an error status.
Status RegisterProvider(std::unique_ptr<RuntimeProvider> provider);

/// \brief Returns the provider for the given `type`.
///
/// \param type The device type.
/// \return     The provider, or a "not built" error status if none registered.
StatusOr<RuntimeProvider*> ProviderFor(DeviceType type);

}  // namespace inferx