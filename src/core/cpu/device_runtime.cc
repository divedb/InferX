// CPU DeviceRuntime: host allocation, memcpy, and no-op synchronization.
//
// The CPU runtime exists so that the scheduler, KV-cache bookkeeping, and the
// tensor layer are unit-testable with no device attached. Its streams and
// events are dummies -- host execution is synchronous -- and graph capture is
// unsupported, which `capabilities()` reports.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "../device_runtime_internal.h"
#include "absl/status/status.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"

namespace inferx::internal {
namespace {

// Rounds a byte count up to a multiple of `alignment`.
size_t RoundUp(size_t bytes, size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

// Allocates host memory aligned to at least `alignment`, or nullptr.
//
// std::aligned_alloc requires the size to be a multiple of the alignment, so
// the request is rounded up before the call.
void* AlignedHostAlloc(size_t bytes, size_t alignment) {
  if (bytes == 0) {
    return nullptr;
  }
  return std::aligned_alloc(alignment, RoundUp(bytes, alignment));
}

class CpuRuntime final : public DeviceRuntime {
 public:
  CpuRuntime() : DeviceRuntime(DeviceId::Cpu()) {}
  int DeviceCount() const override { return 1; }

  Status Activate() override { return OkStatus(); }

  StatusOr<DeviceMemoryInfo> GetMemoryInfo() override {
#if defined(_SC_PHYS_PAGES) && defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    const long page_size = sysconf(_SC_PAGESIZE);
    const long total_pages = sysconf(_SC_PHYS_PAGES);
    const long free_pages = sysconf(_SC_AVPHYS_PAGES);
    DeviceMemoryInfo info;
    if (page_size > 0) {
      if (total_pages > 0) {
        info.total_bytes = static_cast<size_t>(total_pages) * static_cast<size_t>(page_size);
      }
      if (free_pages > 0) {
        info.free_bytes = static_cast<size_t>(free_pages) * static_cast<size_t>(page_size);
      }
    }
    return info;
#else
    return DeviceMemoryInfo{};
#endif
  }

  StatusOr<void*> Allocate(size_t bytes) override {
    void* p = AlignedHostAlloc(bytes, kTensorAlignment);
    if (p == nullptr && bytes != 0) {
      return ResourceExhaustedError("failed to allocate ", bytes, " bytes of host memory");
    }
    return p;
  }

  Status Free(void* ptr) override {
    std::free(ptr);
    return OkStatus();
  }

  StatusOr<void*> AllocatePinnedHost(size_t bytes) override {
    // Host memory is not page-locked here; the allocation is still valid, but
    // capabilities().pinned_host_memory reports false so callers do not assume
    // DMA-safe memory.
    void* p = AlignedHostAlloc(bytes, kPageAlignment);
    if (p == nullptr && bytes != 0) {
      return ResourceExhaustedError("failed to allocate ", bytes, " bytes of host memory");
    }
    return p;
  }

  Status FreePinnedHost(void* ptr) override {
    std::free(ptr);
    return OkStatus();
  }

  Status Copy(void* dst, const void* src, size_t bytes, CopyKind /*kind*/) override {
    if (bytes != 0 && dst != nullptr && src != nullptr) {
      std::memcpy(dst, src, bytes);
    }
    return OkStatus();
  }

  Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                   Stream /*stream*/) override {
    return Copy(dst, src, bytes, kind);
  }

  StatusOr<Stream> CreateStream() override { return Stream(nullptr); }

  Status DestroyStream(Stream /*stream*/) override { return OkStatus(); }

  Status SynchronizeStream(Stream /*stream*/) override { return OkStatus(); }

  StatusOr<DeviceEvent> CreateEvent(bool /*timing*/) override { return DeviceEvent{nullptr}; }

  Status DestroyEvent(DeviceEvent /*event*/) override { return OkStatus(); }

  Status RecordEvent(DeviceEvent /*event*/, Stream /*stream*/) override { return OkStatus(); }

  Status SynchronizeEvent(DeviceEvent /*event*/) override { return OkStatus(); }

  StatusOr<bool> QueryEvent(DeviceEvent /*event*/) override { return true; }

  StatusOr<float> ElapsedMs(DeviceEvent /*start*/, DeviceEvent /*end*/) override {
    return 0.0f;
  }

  StatusOr<bool> IsCapturing(Stream /*stream*/) override { return false; }

  Status BeginCapture(Stream /*stream*/) override {
    return UnimplementedError("CPU graph capture is not supported");
  }

  StatusOr<GraphExec> EndCaptureAndInstantiate(Stream /*stream*/) override {
    return UnimplementedError("CPU graph capture is not supported");
  }

  Status LaunchGraph(GraphExec /*graph*/, Stream /*stream*/) override {
    return UnimplementedError("CPU graph launch is not supported");
  }

  Status DestroyGraph(GraphExec /*graph*/) override {
    return UnimplementedError("CPU graph capture is not supported");
  }

  const RuntimeCapabilities& capabilities() const override { return capabilities_; }

 private:
  RuntimeCapabilities capabilities_{};
};

}  // namespace

DeviceRuntime* CpuDeviceRuntime() {
  static CpuRuntime runtime;
  return &runtime;
}

}  // namespace inferx::internal
