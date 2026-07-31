// DeviceBuffer -- an owning, move-only allocation on a specific device.
//
// This is the only type in the engine that calls cudaMalloc/cudaFree. Every
// other allocation is a sub-allocation carved out of a DeviceBuffer by BumpArena
// or CachingAllocator. cudaFree implicitly synchronizes the device, so a
// DeviceBuffer being destroyed on a hot path would stall the whole pipeline;
// the intent is that all of them are created at startup and outlive the engine.
//
// A DeviceBuffer with DeviceKind::kCpu is backed by aligned host memory using
// the same alignment as device memory. This is what allows the arena and
// allocator layers -- and their tests -- to run on a machine with no GPU.

#ifndef INFERX_CORE_DEVICE_BUFFER_H_
#define INFERX_CORE_DEVICE_BUFFER_H_

#include <cstddef>
#include <utility>

#include "inferx/core/device.h"
#include "inferx/core/status.h"

namespace inferx {

class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  // Allocates `bytes`. Host allocations round up to kTensorAlignment; device
  // allocations are passed to cudaMalloc unpadded, since its own alignment and
  // size granularity are already far coarser. A zero-byte request yields a
  // valid, empty buffer with a null pointer.
  static StatusOr<DeviceBuffer> Allocate(size_t bytes, DeviceId device);

  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        device_(other.device_) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      device_ = other.device_;
    }
    return *this;
  }

  std::byte* data() const { return data_; }
  size_t size() const { return size_; }
  DeviceId device() const { return device_; }
  bool valid() const { return data_ != nullptr; }

  // Frees eagerly. Named rather than implicit because on a CUDA device this
  // synchronizes, and call sites should be visible in review.
  void Reset();

 private:
  DeviceBuffer(std::byte* data, size_t size, DeviceId device)
      : data_(data), size_(size), device_(device) {}

  std::byte* data_ = nullptr;
  size_t size_ = 0;
  DeviceId device_;
};

}  // namespace inferx

#endif  // INFERX_CORE_DEVICE_BUFFER_H_
