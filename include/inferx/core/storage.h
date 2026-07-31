#pragma once

#include <cstddef>
#include <string>

#include "inferx/core/allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/intrusive_ref_cnt_ptr.h"
#include "inferx/core/status.h"

namespace inferx {

class Storage;
using StoragePtr = IntrusiveRefCntPtr<Storage>;

/// \brief A reference-counted, possibly-borrowed range of device memory.
///
/// A Storage holds a contiguous block of bytes on some device and is shared
/// between tensors and views through intrusive reference counting. It exists
/// in exactly two modes, reachable only through the static factories:
///   * owned    -- The bytes came from an Allocator and are returned to it when
///                 the last reference is dropped;
///   * borrowed -- The bytes live elsewhere and are never freed by Storage.
class Storage final : public ThreadSafeRefCountedBase<Storage> {
 public:
  /// \brief Allocates `bytes` from `allocator`, which must outlive the Storage.
  ///
  /// \param bytes     The number of bytes to allocate.
  /// \param allocator The allocator to use for the allocation. Must not be
  ///                  nullptr and must outlive the Storage.
  /// \return          A StatusOr containing a StoragePtr on success, or an
  ///                  error status on failure.
  static StatusOr<StoragePtr> Allocate(size_t bytes, Allocator* allocator);

  /// \brief Allocates `bytes` from the default allocator for `device`.
  ///
  /// \param bytes  The number of bytes to allocate.
  /// \param device The device to allocate on.
  /// \return       A StatusOr containing a StoragePtr on success, or an error
  ///               status on failure.
  static StatusOr<StoragePtr> Allocate(size_t bytes, DeviceId device);

  /// \brief Wraps memory owned elsewhere, freeing nothing. The caller
  ///        guarantees the bytes outlive every handle derived from this one.
  ///
  /// \param data   The pointer to the memory block to wrap. Must not be
  /// nullptr.
  /// \param bytes  The size of the memory block to wrap, in bytes.
  /// \param device The device on which the memory block resides.
  /// \return       A StoragePtr that wraps the given memory block.
  static StoragePtr Borrow(void* data, size_t bytes, DeviceId device);

  ~Storage();

  /// \brief Returns a pointer to the underlying memory block.
  ///
  /// \return A pointer to the underlying memory block.
  std::byte* Data() const { return static_cast<std::byte*>(data_); }

  /// \brief Returns the size of the underlying memory block, in bytes.
  ///
  /// \return The size of the underlying memory block, in bytes.
  size_t Size() const { return size_; }

  /// \brief Returns the device on which the underlying memory block resides.
  ///
  /// \return The device on which the underlying memory block resides.
  DeviceId Device() const { return device_; }

  /// \brief Check whether the storage is borrowed.
  ///
  /// \return True if the storage is borrowed; otherwise false.
  bool IsBorrowed() const { return allocator_ == nullptr; }

  std::string ToString() const;

 private:
  /// \brief Constructs a Storage object.
  ///
  /// \param data      The pointer to the memory block.
  /// \param bytes     The size of the memory block, in bytes.
  /// \param device    The device on which the memory block resides.
  /// \param allocator The allocator responsible for the memory block. nullptr
  ///                  if borrowed.
  Storage(void* data, size_t bytes, DeviceId device, Allocator* allocator)
      : data_(data), size_(bytes), device_(device), allocator_(allocator) {}

  void* data_ = nullptr;
  size_t size_ = 0;
  DeviceId device_;
  Allocator* allocator_ = nullptr;  // null => borrowed
};

}  // namespace inferx
