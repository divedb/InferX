#include "inferx/core/allocator.h"

#include <mimalloc.h>

#include <array>
#include <bit>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "inferx/core/device_runtime.h"

namespace inferx {
namespace {

Status CheckAlignment(size_t alignment) {
  if (alignment == 0 || !std::has_single_bit(alignment)) {
    return InvalidArgumentError("alignment ", alignment,
                                " is not a non-zero power of two");
  }

  return OkStatus();
}

class HostAllocatorImpl final : public Allocator {
 public:
  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0) return static_cast<void*>(nullptr);

    INFERX_RETURN_IF_ERROR(CheckAlignment(alignment));

    // mimalloc honours the requested alignment itself, so -- unlike
    // std::aligned_alloc -- the size needs no rounding to a multiple of it. It
    // also remembers the block, so mi_free is O(1) and takes no size.
    void* p = mi_malloc_aligned(bytes, alignment);

    if (p == nullptr) {
      return ResourceExhaustedError("failed to allocate ", bytes,
                                    " bytes of host memory aligned to ",
                                    alignment);
    }

    return p;
  }

  Status Deallocate(void* ptr) override {
    mi_free(ptr);

    return OkStatus();
  }

  DeviceId Device() const override { return DeviceId::Cpu(); }
  std::string_view Name() const override { return "host"; }
};

class DeviceAllocatorImpl final : public Allocator {
 public:
  explicit DeviceAllocatorImpl(DeviceId device) : device_(device) {}

  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0) return static_cast<void*>(nullptr);

    INFERX_RETURN_IF_ERROR(CheckAlignment(alignment));

    INFERX_ASSIGN_OR_RETURN(DeviceRuntime * runtime, RuntimeFor(device_));
    INFERX_ASSIGN_OR_RETURN(void* p, runtime->Allocate(device_, bytes));

    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);

    if (addr % alignment != 0) {
      (void)runtime->Free(device_, p);

      return InternalError("cudaMalloc returned 0x", absl::Hex(addr),
                           " which is not aligned to ", alignment, " bytes");
    }

    return p;
  }

  Status Deallocate(void* ptr) override {
    if (ptr == nullptr) return OkStatus();

    INFERX_ASSIGN_OR_RETURN(DeviceRuntime * runtime, RuntimeFor(device_));
    return runtime->Free(device_, ptr);
  }

  DeviceId Device() const override { return device_; }
  std::string_view Name() const override { return "accelerator"; }

 private:
  DeviceId device_;
};

constexpr int kMaxDevices = 16;

StatusOr<Allocator*> DeviceAllocatorFor(DeviceId device) {
  const int8_t ordinal = device.index;
  if (ordinal < 0 || ordinal >= kMaxDevices) {
    return InvalidArgumentError("device ordinal ", static_cast<int>(ordinal),
                                " is outside [0, ", kMaxDevices, ")");
  }
  // One allocator per ordinal, created on first use and intentionally leaked
  // for the same reason as the host allocator.
  static std::mutex mu;
  static std::array<Allocator*, kMaxDevices> table{};

  std::lock_guard<std::mutex> lock(mu);
  if (table[ordinal] == nullptr) {
    table[ordinal] = new DeviceAllocatorImpl(device);
  }
  return table[ordinal];
}

}  // namespace

StatusOr<Allocator*> AllocatorFor(DeviceId device) {
  if (device.IsCpu()) {
    static Allocator* const kInstance = new HostAllocatorImpl();

    return kInstance;
  }

  INFERX_RETURN_IF_ERROR(RuntimeFor(device).status());
  return DeviceAllocatorFor(device);
}

}  // namespace inferx
