#include "inferx/core/device_buffer.h"

#include <cstdlib>

#include "absl/strings/str_cat.h"
#include "inferx/core/cuda_utils.h"

namespace inferx {
namespace {

size_t RoundUp(size_t bytes, size_t alignment) {
  return (bytes + alignment - 1) / alignment * alignment;
}

}  // namespace

StatusOr<DeviceBuffer> DeviceBuffer::Allocate(size_t bytes, DeviceId device) {
  if (bytes == 0) {
    return DeviceBuffer(nullptr, 0, device);
  }

  if (device.IsCpu()) {
    // Rounded only because std::aligned_alloc requires the size to be a
    // multiple of the alignment.
    const size_t padded = RoundUp(bytes, kTensorAlignment);
    void* p = std::aligned_alloc(kTensorAlignment, padded);
    if (p == nullptr) {
      return ResourceExhaustedError("failed to allocate ", padded,
                                    " bytes of host memory");
    }
    return DeviceBuffer(static_cast<std::byte*>(p), padded, device);
  }

#ifdef INFERX_WITH_CUDA
  int prev_device = 0;
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetDevice(&prev_device));
  const bool switch_device = prev_device != device.index;
  if (switch_device) {
    INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(device.index));
  }

  // Unpadded: cudaMalloc's own alignment and size granularity are far coarser
  // than anything we could usefully request.
  void* p = nullptr;
  const cudaError_t err = cudaMalloc(&p, bytes);

  if (switch_device) {
    // Restore before surfacing any error, so a failed allocation does not also
    // leave the calling thread pointed at the wrong device.
    INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(prev_device));
  }
  if (err != cudaSuccess) {
    return CudaErrorToStatus(err, "cudaMalloc", __FILE__, __LINE__);
  }
  return DeviceBuffer(static_cast<std::byte*>(p), bytes, device);
#else
  return FailedPreconditionError(
      "cannot allocate on ", device.ToString(),
      ": built without CUDA support (INFERX_ENABLE_CUDA=OFF)");
#endif
}

void DeviceBuffer::Reset() {
  if (data_ == nullptr) return;

  if (device_.IsCpu()) {
    std::free(data_);
  } else {
#ifdef INFERX_WITH_CUDA
    // Destructors cannot report failure. cudaFree only fails here if the
    // context is already being torn down, in which case the memory is gone
    // anyway; clear the sticky error so it does not surface at an unrelated
    // call site later.
    if (cudaFree(data_) != cudaSuccess) {
      cudaGetLastError();
    }
#endif
  }
  data_ = nullptr;
  size_ = 0;
}

}  // namespace inferx
