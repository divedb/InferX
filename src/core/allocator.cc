#include "inferx/core/allocator.h"

#include <mimalloc.h>

#include <array>
#include <bit>
#include <mutex>

#include "absl/strings/str_cat.h"
#include "inferx/core/cuda_utils.h"

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

#ifdef INFERX_WITH_CUDA
class CudaAllocatorImpl final : public Allocator {
 public:
  explicit CudaAllocatorImpl(int8_t ordinal) : ordinal_(ordinal) {}

  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override {
    if (bytes == 0) return static_cast<void*>(nullptr);

    INFERX_RETURN_IF_ERROR(CheckAlignment(alignment));

    int prev = 0;
    INFERX_CUDA_RETURN_IF_ERROR(cudaGetDevice(&prev));
    const bool switch_device = prev != ordinal_;

    if (switch_device) {
      INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(ordinal_));
    }

    // Unpadded. cudaMalloc suballocates from large VA reservations and applies
    // its own size granularity, so rounding here would only inflate the
    // request.
    void* p = nullptr;
    const cudaError_t err = cudaMalloc(&p, bytes);

    if (switch_device) {
      // Restore before surfacing any error, so a failed allocation does not
      // also leave the calling thread pointed at the wrong device.
      INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(prev));
    }

    if (err != cudaSuccess) {
      return CudaErrorToStatus(err, "cudaMalloc", __FILE__, __LINE__);
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(p);

    if (addr % alignment != 0) {
      cudaFree(p);

      return InternalError("cudaMalloc returned 0x", absl::Hex(addr),
                           " which is not aligned to ", alignment, " bytes");
    }

    return p;
  }

  Status Deallocate(void* ptr) override {
    if (ptr == nullptr) return OkStatus();

    const cudaError_t err = cudaFree(ptr);

    if (err != cudaSuccess) {
      return CudaErrorToStatus(err, "cudaFree", __FILE__, __LINE__);
    }

    return OkStatus();
  }

  DeviceId Device() const override { return DeviceId::Cuda(ordinal_); }
  std::string_view Name() const override { return "cuda"; }

 private:
  int8_t ordinal_;
};

constexpr int kMaxDevices = 16;

StatusOr<Allocator*> CudaAllocatorFor(int8_t ordinal) {
  if (ordinal < 0 || ordinal >= kMaxDevices) {
    return InvalidArgumentError("CUDA ordinal ", static_cast<int>(ordinal),
                                " is outside [0, ", kMaxDevices, ")");
  }
  // One allocator per ordinal, created on first use and intentionally leaked
  // for the same reason as the host allocator.
  static std::mutex mu;
  static std::array<Allocator*, kMaxDevices> table{};

  std::lock_guard<std::mutex> lock(mu);
  if (table[ordinal] == nullptr) {
    table[ordinal] = new CudaAllocatorImpl(ordinal);
  }
  return table[ordinal];
}
#endif  // INFERX_WITH_CUDA

}  // namespace

StatusOr<Allocator*> AllocatorFor(DeviceId device) {
  if (device.IsCpu()) {
    static Allocator* const kInstance = new HostAllocatorImpl();

    return kInstance;
  }

#ifdef INFERX_WITH_CUDA
  return CudaAllocatorFor(device.index);
#else
  return FailedPreconditionError(
      "cannot allocate on ", device.ToString(),
      ": built without CUDA support (INFERX_ENABLE_CUDA=OFF)");
#endif
}

}  // namespace inferx
