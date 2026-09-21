#pragma once

#include <cstddef>
#include <string_view>

#include "inferx/core/device.h"
#include "inferx/core/status.h"

namespace inferx {

/// \brief Abstract interface for memory allocators.
///
/// Allocators allocate and deallocate memory on a specific device. Storage uses
/// them to manage the lifetime of memory buffers. Implementations must be
/// thread-safe and must tolerate nullptr in Deallocate(), which is a common
/// pattern in destructors.
class Allocator {
 public:
  virtual ~Allocator() = default;

  /// \brief Allocates `bytes` aligned to at least `alignment`.
  ///
  /// Implementations may return memory more strongly aligned than requested;
  /// cudaMalloc in particular returns far coarser alignment than anything we
  /// would ask for.
  ///
  /// \param bytes     The size of the memory block to allocate, in bytes.
  /// \param alignment Minimum alignment of the returned pointer. Must be a
  ///                  non-zero power of two.
  /// \return          A pointer to the allocated block, or an error status.
  ///                  A zero-byte request yields a null pointer and OK.
  virtual StatusOr<void*> Allocate(size_t bytes, size_t alignment) = 0;

  /// \brief Allocates `bytes` aligned for tensor data.
  ///
  /// Non-virtual on purpose. Expressing the default as `alignment =
  /// kTensorAlignment` on the virtual above would bind it from the *static*
  /// type of the caller's pointer, so an override that declared a different
  /// default would silently change what a caller got depending on which type it
  /// held the allocator by. A separate overload has one meaning everywhere.
  ///
  /// \param bytes The size of the memory block to allocate, in bytes.
  /// \return      A pointer to the allocated block, or an error status.
  StatusOr<void*> Allocate(size_t bytes) {
    return Allocate(bytes, kTensorAlignment);
  }

  /// \brief Deallocates a block previously returned by this allocator.
  ///
  /// Returns InvalidArgument for a pointer this allocator did not hand out or
  /// has already reclaimed -- both are real bugs, so the reason is reported
  /// rather than discarded here. Callers that cannot act on the failure
  /// (destructors) must discard it explicitly at the call site, which keeps
  /// every such discard greppable.
  ///
  /// Not noexcept: constructing an error Status may allocate. The OK path does
  /// not allocate and does not throw.
  ///
  /// \param ptr The block to deallocate. nullptr is accepted and does nothing.
  /// \return    OK, or InvalidArgument if `ptr` is not a live allocation.
  virtual Status Deallocate(void* ptr) = 0;

  /// \brief Returns the device this allocator allocates on.
  ///
  /// \return The device ID of the allocator.
  virtual DeviceId Device() const = 0;

  /// \brief Returns a short human-readable name, for diagnostics.
  ///
  /// \return A string view of the allocator's name.
  virtual std::string_view Name() const = 0;
};

/// \brief Returns the default allocator for `device`, defaulting to the host.
///
/// The sole entry point for obtaining an allocator: `DeviceId` is the only way
/// to name a device, and omitting the argument selects the CPU.
///
/// \note The returned allocator is a singleton for the device, and is
///       intentionally leaked: it must outlive every Storage that frees through
///       it, including those destroyed during static teardown. The caller must
///       not delete it.
///
/// \param device The device to allocate on; defaults to the CPU.
/// \return       A pointer to the allocator, or FailedPrecondition when the
///               binary was built without CUDA support, or InvalidArgument for
///               an out-of-range CUDA ordinal.
StatusOr<Allocator*> AllocatorFor(DeviceId device = DeviceId::Cpu());

}  // namespace inferx
