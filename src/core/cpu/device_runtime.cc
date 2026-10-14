// CPU backend of the device runtime layer.
//
// Host memory is the device memory; streams and events are inert because
// host compute is synchronous. Pinned host memory is a plain aligned
// allocation (nothing to page-lock), and host registration is a no-op, so
// callers need not branch on the backend.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#include "absl/status/status.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"

namespace inferx {
namespace {

size_t RoundUp(size_t bytes, size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

// std::aligned_alloc requires the size to be a multiple of the alignment.
void* AlignedHostAlloc(size_t bytes, size_t alignment) {
  if (bytes == 0) return nullptr;
  return std::aligned_alloc(alignment, RoundUp(bytes, alignment));
}

class CpuMemoryApi final : public MemoryApi {
 public:
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

  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override {
    void* p = AlignedHostAlloc(bytes, alignment);
    if (p == nullptr && bytes != 0) {
      return ResourceExhaustedError("failed to allocate ", bytes, " bytes of host memory");
    }
    return p;
  }

  Status Free(void* ptr) override {
    std::free(ptr);
    return OkStatus();
  }

  StatusOr<void*> AllocatePinnedHost(size_t bytes, size_t alignment) override {
    return Allocate(bytes, alignment);
  }

  Status FreePinnedHost(void* ptr) override { return Free(ptr); }

  Status RegisterHost(void* /*ptr*/, size_t /*bytes*/) override { return OkStatus(); }

  Status UnregisterHost(void* /*ptr*/) override { return OkStatus(); }

  Status Copy(void* dst, const void* src, size_t bytes, CopyKind /*kind*/) override {
    if (bytes != 0 && dst != nullptr && src != nullptr) {
      std::memcpy(dst, src, bytes);
    }
    return OkStatus();
  }

  Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                   DeviceStream /*stream*/) override {
    return Copy(dst, src, bytes, kind);
  }

  Status MemsetAsync(void* dst, uint8_t value, size_t bytes,
                     DeviceStream /*stream*/) override {
    if (bytes != 0 && dst != nullptr) {
      std::memset(dst, value, bytes);
    }
    return OkStatus();
  }
};

class CpuStreamApi final : public StreamApi {
 public:
  StatusOr<OwnedStream> CreateStream(StreamPriority /*priority*/) override {
    // The handle stays default (null): there is no queue to represent. The
    // owner keeps the API pointer so destruction routes back here.
    return OwnedStream(DeviceStream{}, this);
  }

  Status SynchronizeStream(DeviceStream /*stream*/) override { return OkStatus(); }

  StatusOr<bool> QueryStream(DeviceStream /*stream*/) override { return true; }

  Status WaitEvent(DeviceStream /*stream*/, DeviceEvent /*event*/) override {
    return OkStatus();
  }

  Status SynchronizeDevice() override { return OkStatus(); }

  StatusOr<OwnedEvent> CreateEvent(bool /*timing*/) override {
    return OwnedEvent(DeviceEvent{}, this);
  }

  Status RecordEvent(DeviceEvent /*event*/, DeviceStream /*stream*/) override {
    return OkStatus();
  }

  Status SynchronizeEvent(DeviceEvent /*event*/) override { return OkStatus(); }

  StatusOr<bool> QueryEvent(DeviceEvent /*event*/) override { return true; }

  StatusOr<float> ElapsedMs(DeviceEvent /*start*/, DeviceEvent /*end*/) override {
    return UnimplementedError("CPU events carry no timing");
  }

 protected:
  void Destroy(DeviceStream /*stream*/) noexcept override {}
  void Destroy(DeviceEvent /*event*/) noexcept override {}
};

class CpuRuntime final : public DeviceRuntime {
 public:
  CpuRuntime() : DeviceRuntime(Device::Cpu()) {}

  StatusOr<DeviceProperties> GetProperties() override {
    DeviceProperties properties;
    properties.name = "cpu";
    return properties;
  }

  Status Activate() override { return OkStatus(); }

  MemoryApi& Memory() override { return memory_; }
  StreamApi& Streams() override { return streams_; }

 private:
  CpuMemoryApi memory_;
  CpuStreamApi streams_;
};

class CpuProvider final : public RuntimeProvider {
 public:
  DeviceType GetDeviceType() const override { return DeviceType::kCpu; }

  StatusOr<int> DeviceCount() override { return 1; }

  StatusOr<DeviceRuntime*> GetRuntime(int ordinal) override {
    if (ordinal != 0) {
      return NotFoundError("the CPU backend has device 0 only, requested ", ordinal);
    }
    static CpuRuntime runtime;
    return &runtime;
  }
};

}  // namespace

std::unique_ptr<RuntimeProvider> MakeCpuProvider() {
  return std::make_unique<CpuProvider>();
}

}  // namespace inferx
