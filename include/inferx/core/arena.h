#pragma once

#include <cstddef>

#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx {

/// \brief A simple bump allocator over a pre-allocated region.
///
/// The arena does not own the memory it manages; the caller must ensure that
/// the backing region outlives the arena. The arena is not thread-safe; callers
/// must synchronize access if multiple threads may allocate from the same arena
/// concurrently.
class BumpArena {
 public:
  /// \brief Constructs an empty arena with no backing memory.
  BumpArena() = default;

  /// \brief Constructs an arena over a pre-allocated region.
  ///
  /// \param base     The base address of the pre-allocated region.
  /// \param capacity The capacity of the region in bytes.
  /// \param device   The device ID associated with the arena.
  BumpArena(std::byte* base, size_t capacity, DeviceId device)
      : base_(base), capacity_(capacity), device_(device) {}

  /// \brief Allocates a block of memory from the arena.
  ///
  /// \note If the arena is not initialized, or if the requested alignment is
  ///       not a non-zero power of two, this function will return an error
  ///       status.
  ///
  /// \param bytes     The number of bytes to allocate.
  /// \param alignment The alignment requirement for the allocation.
  /// \return          A pointer to the allocated memory, or an error status if
  ///                  the allocation fails.
  StatusOr<void*> Allocate(size_t bytes, size_t alignment);

  /// \brief Allocates a block of memory from the arena, aligned for tensor
  ///        data.
  ///
  /// \param bytes The number of bytes to allocate.
  /// \return      A pointer to the allocated memory, or an error status if the
  ///              allocation fails.
  StatusOr<void*> Allocate(size_t bytes) {
    return Allocate(bytes, kTensorAlignment);
  }

  /// \brief Allocates a TensorView from the arena with the specified data type
  ///        and shape.
  ///
  /// \param dtype The data type of the TensorView to allocate.
  /// \param shape The shape of the TensorView to allocate.
  /// \return      A StatusOr containing the allocated TensorView, or an error
  ///              status if the allocation fails.
  StatusOr<TensorView> AllocateView(DataType dtype, const Shape& shape);

  // Releases everything at once. Any Tensor still pointing into the arena
  // dangles afterwards; this is only for teardown and for reuse between tests.
  //
  // Deliberately does not rewind the high-water mark: the point of that number
  // is to size the arena across a whole run, and a per-step Reset() would erase
  // exactly the peak it is meant to record.
  void Reset() { offset_ = 0; }

  /// \brief Returns the total capacity of the arena in bytes.
  ///
  /// \return The total capacity of the arena in bytes.
  size_t Capacity() const { return capacity_; }

  /// \brief Returns the number of bytes currently used in the arena.
  ///
  /// \return The number of bytes currently used in the arena.
  size_t Used() const { return offset_; }

  /// \brief Returns the high-water mark of bytes used, across Reset()s.
  ///
  /// This is what an arena is sized from: the largest a single step ever grew,
  /// not what it happens to hold now.
  ///
  /// \return The greatest number of bytes ever used at one time.
  size_t PeakUsed() const { return peak_; }

  /// \brief Returns the number of bytes remaining in the arena.
  ///
  /// \return The number of bytes remaining in the arena.
  size_t Remaining() const { return capacity_ - offset_; }

  /// \brief Returns the device ID associated with the arena.
  ///
  /// \return The device ID associated with the arena.
  DeviceId Device() const { return device_; }

  /// \brief Returns the base address of the arena.
  ///
  /// \return The base address of the arena.
  std::byte* Base() const { return base_; }

 private:
  std::byte* base_ = nullptr;
  size_t capacity_ = 0;
  size_t offset_ = 0;
  size_t peak_ = 0;
  DeviceId device_;
};

}  // namespace inferx
