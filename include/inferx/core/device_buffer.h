#ifndef INFERX_CORE_DEVICE_BUFFER_H_
#define INFERX_CORE_DEVICE_BUFFER_H_

#include <cstddef>
#include <utility>

#include "inferx/core/device.h"
#include "inferx/core/status.h"

namespace inferx {

/// \brief A uniquely-owned contiguous device allocation.
class DeviceBuffer {
 public:
  /// \brief Constructs an empty, invalid buffer.
  DeviceBuffer() = default;

  /// \brief Allocates `bytes` on `device`.
  ///
  /// Host allocations round up to kTensorAlignment; device allocations are
  /// passed to cudaMalloc unpadded, since its own alignment and size
  /// granularity are already far coarser. A zero-byte request yields a valid,
  /// empty buffer with a null pointer.
  ///
  /// \param bytes  Number of bytes to allocate.
  /// \param device Device to allocate on.
  /// \return       The buffer, or an error status.
  static StatusOr<DeviceBuffer> Allocate(size_t bytes, DeviceId device);

  /// \brief Frees the buffer, if it holds one.
  ~DeviceBuffer() { Reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  /// \brief Moves the allocation out of `other`, leaving it empty.
  ///
  /// \param other Buffer to move from.
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        device_(other.device_) {}

  /// \brief Move-assigns, freeing whatever this buffer held first.
  ///
  /// \param other Buffer to move from.
  /// \return      Reference to this buffer.
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      Reset();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      device_ = other.device_;
    }
    return *this;
  }

  /// \brief Returns the buffer's bytes, or nullptr when invalid.
  std::byte* data() const { return data_; }
  /// \brief Returns the buffer's size in bytes.
  size_t size() const { return size_; }
  /// \brief Returns the device the buffer lives on.
  DeviceId device() const { return device_; }
  /// \brief True when the buffer holds an allocation.
  bool valid() const { return data_ != nullptr; }

  /// \brief Frees eagerly.
  ///
  /// Named rather than implicit because on a CUDA device this synchronizes,
  /// and call sites should be visible in review.
  void Reset();

 private:
  /// \brief Adopts an already-allocated block.
  ///
  /// \param data   The allocation's address.
  /// \param size   The allocation's size in bytes.
  /// \param device The device the allocation lives on.
  DeviceBuffer(std::byte* data, size_t size, DeviceId device)
      : data_(data), size_(size), device_(device) {}

  std::byte* data_ = nullptr;
  size_t size_ = 0;
  DeviceId device_;
};

}  // namespace inferx

#endif  // INFERX_CORE_DEVICE_BUFFER_H_
