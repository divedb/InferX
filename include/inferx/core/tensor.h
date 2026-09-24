#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "inferx/core/allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/intrusive_ref_cnt_ptr.h"
#include "inferx/core/status.h"
#include "inferx/core/storage.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

/// \brief The immutable, reference-counted body behind a `Tensor` handle.
///
/// A TensorImpl binds a Storage (the bytes) to a view of those bytes: a byte
/// offset, a DataType, and a Shape. It is what a Tensor points at, and the
/// reason sizeof(Tensor) == sizeof(void*): copying a Tensor is a single atomic
/// retain on the impl, never a metadata copy.
///
/// It is immutable after construction -- there are no setters. The operations
/// that read as mutational (Slice, Reshape, Bitcast) instead return a new
/// Tensor whose TensorImpl shares the same Storage, possibly at a different
/// offset. That shared-storage fan-out is why Storage is refcounted on its own:
/// several TensorImpls can view one Storage, and the two refcounts count
/// different relationships. See docs/ARCHITECTURE.md section 7.1.
///
/// Construct through Tensor's factories (Empty, FromBlob, FromStorage), which
/// validate the dtype/shape pair and bound-check the offset against the
/// storage. Data() is GetStorage()->Data() + StorageOffset(), in bytes.
class TensorImpl final : public ThreadSafeRefCountedBase<TensorImpl> {
 public:
  /// \brief Constructs a TensorImpl with the given Storage, offset, dtype, and
  ///        shape.
  ///
  /// \param storage The StoragePtr that owns the underlying bytes.
  /// \param offset  The byte offset into the Storage where this TensorImpl's
  ///                data begins.
  /// \param dtype   The DataType of the elements in this TensorImpl.
  /// \param shape   The Shape of the tensor, describing the dimensions and
  ///                extents.
  TensorImpl(StoragePtr storage, int64_t offset, DataType dtype, Shape shape)
      : storage_(std::move(storage)), offset_(offset), spec_(dtype, std::move(shape)) {}

  ~TensorImpl() = default;

  /// \brief Returns the Storage backing this tensor.
  ///
  /// Named GetStorage/GetDataType/GetShape rather than
  /// Storage/DataType/Shape: a member function may not share a name with a
  /// type used in the same class.
  const StoragePtr& GetStorage() const { return storage_; }
  /// \brief Returns the byte offset into the Storage where this tensor's data
  ///        begins.
  int64_t StorageOffset() const { return offset_; }
  /// \brief Returns the DataType of the elements in this tensor.
  DataType GetDataType() const { return spec_.GetDataType(); }
  /// \brief Returns the Shape of this tensor.
  const Shape& GetShape() const { return spec_.GetShape(); }
  /// \brief Returns the device the bytes live on.
  DeviceId Device() const { return storage_->Device(); }

  /// \brief Returns a pointer to the tensor's data.
  void* Data() const { return storage_->Data() + StorageOffset(); }

  /// \brief Returns the number of bytes occupied by the tensor's data.
  ///
  /// \return The number of bytes occupied by the tensor's data.
  int64_t NBytes() const noexcept { return spec_.NBytes(); }

 private:
  StoragePtr storage_;
  int64_t offset_ = 0;
  TensorSpec spec_;
};

/// \brief A Tensor is a handle to a TensorImpl: a dtype, a shape, and the
///        bytes backing them.
///
/// Like `torch::Tensor`, a Tensor is a cheap, copyable value that shares
/// storage. Copying (copy-construct, copy-assign, passing by value, storing in
/// containers) is shallow: every copy is another handle to the same
/// TensorImpl, writes through one are visible through the others, and the
/// bytes stay alive until the last handle drops. There is no separate view
/// type; Slice/Reshape/Bitcast return new Tensors that share the same
/// Storage, and FromBlob wraps memory the caller owns.
///
/// Tensors are contiguous and row-major. A Tensor always points at a
/// TensorImpl -- there is no undefined state. Construction goes through the
/// factories (Empty, FromBlob, FromStorage), which validate the dtype/shape
/// pair; failure travels in the returned StatusOr, and a maybe-absent tensor
/// is the caller's std::optional.
class Tensor {
 public:
  /// \brief Allocates uninitialized memory from the default allocator for
  ///        `device`.
  ///
  /// \param dtype  The DataType of the tensor elements.
  /// \param shape  The Shape of the tensor, describing the dimensions and
  ///               extents.
  /// \param device The DeviceId of the device on which to allocate the tensor.
  /// \return       A StatusOr containing a Tensor on success, or an error
  ///               status on failure.
  static StatusOr<Tensor> Empty(DataType dtype, const Shape& shape, DeviceId device);

  /// \brief Allocates uninitialized memory from a specific allocator -- a
  ///        per-rank workspace, say.
  ///
  /// \param dtype     The DataType of the tensor elements.
  /// \param shape     The Shape of the tensor, describing the dimensions and
  ///                  extents.
  /// \param allocator The allocator to allocate from; must not be nullptr.
  /// \return          A StatusOr containing a Tensor on success, or an error
  ///                  status on failure.
  static StatusOr<Tensor> Empty(DataType dtype, const Shape& shape, Allocator* allocator);

  /// \brief Wraps memory owned elsewhere, freeing nothing.
  ///
  /// The caller guarantees the bytes outlive every handle derived from this
  /// one. This is how mmap'd weights and arena carve-outs enter the tensor
  /// world.
  ///
  /// \param data   Pointer to the memory to wrap.
  /// \param dtype  The DataType of the tensor elements.
  /// \param shape  The Shape of the tensor.
  /// \param device The device the memory resides on.
  /// \return       A StatusOr containing a Tensor on success, or an error
  ///               status on failure.
  static StatusOr<Tensor> FromBlob(void* data, DataType dtype, const Shape& shape,
                                   DeviceId device);

  /// \brief Shares an existing Storage at a byte offset.
  ///
  /// Bounds are checked against the storage size.
  ///
  /// \param storage The StoragePtr to share.
  /// \param offset  The byte offset into the Storage where the tensor begins.
  /// \param dtype   The DataType of the tensor elements.
  /// \param shape   The Shape of the tensor.
  /// \return        A StatusOr containing a Tensor on success, or an error
  ///                status on failure.
  static StatusOr<Tensor> FromStorage(StoragePtr storage, int64_t offset, DataType dtype,
                                      const Shape& shape);

  /// \brief True when the tensor holds zero elements.
  bool IsEmpty() const { return Numel() == 0; }

  /// \brief Returns a pointer to the tensor's data.
  void* Data() const { return impl_->Data(); }

  /// \brief Returns the tensor's element type.
  DataType GetDataType() const { return impl_->GetDataType(); }

  /// \brief Returns the device the tensor's bytes live on.
  DeviceId Device() const { return impl_->Device(); }

  /// \brief Returns the tensor's shape.
  Shape GetShape() const { return impl_->GetShape(); }

  /// \brief Zero-copy access to the extents.
  ///
  /// Unlike GetShape(), which returns a copy, the span points into storage
  /// this tensor's impl owns, so it stays valid while the tensor does. Prefer
  /// this on hot paths.
  absl::Span<const int64_t> Dims() const { return impl_->GetShape().Dims(); }
  /// \brief Returns the tensor's rank.
  int Rank() const { return impl_->GetShape().Rank(); }

  /// \brief Returns the extent along dimension `i`.
  int64_t Dim(int i) const { return impl_->GetShape().Dim(i); }

  /// \brief Returns the number of elements.
  int64_t Numel() const { return impl_->GetShape().Numel(); }

  /// \brief Returns the number of bytes occupied.
  int64_t NBytes() const { return impl_->NBytes(); }

  /// \brief Returns the byte offset into the Storage.
  int64_t StorageOffset() const { return impl_->StorageOffset(); }

  /// \brief True when the tensor's bytes live in host memory.
  bool IsCpu() const { return Device().IsCpu(); }

  /// \brief True when the tensor's bytes live on a CUDA device.
  bool IsCuda() const { return Device().IsCuda(); }

  /// \brief Typed access, checked against the tensor's dtype.
  ///
  /// \return A pointer to the data as `T*`, or nullptr if `T` is not the
  ///         tensor's element type.
  template <typename T>
  T* DataAs() const {
    return GetDataType() == kDataTypeOf<T> ? static_cast<T*>(Data()) : nullptr;
  }

  /// \brief Returns the data pointer as bytes.
  std::byte* Bytes() const { return static_cast<std::byte*>(Data()); }

  /// \brief Extracts rows `[begin, end)` of dimension 0.
  ///
  /// \param begin First row index (inclusive).
  /// \param end   Last row index (exclusive).
  /// \return      The slice as a new Tensor, or an error status.
  StatusOr<Tensor> Slice(int64_t begin, int64_t end) const;

  /// \brief Returns the same storage under a new shape.
  ///
  /// \param shape The new shape; its element count must match exactly.
  /// \return      The reshaped Tensor, or an error status.
  StatusOr<Tensor> Reshape(const Shape& shape) const;

  /// \brief Reinterprets the buffer as another dtype.
  ///
  /// \param dtype The new element type; the total byte count must match.
  /// \return      The bitcast Tensor, or an error status.
  StatusOr<Tensor> Bitcast(DataType dtype) const;

  /// \brief Returns this tensor's bytes on `device`.
  ///
  /// Returns *this, sharing storage, when the tensor already lives on
  /// `device`. Otherwise allocates a fresh tensor there with the same dtype
  /// and shape and copies the bytes synchronously. Coprocessor-to-coprocessor
  /// copies across different devices are rejected; route through the host.
  ///
  /// \param device The destination device.
  /// \return       The tensor on `device`, or an error status.
  StatusOr<Tensor> To(DeviceId device) const;

  /// \brief Returns the number of handles to this tensor's impl.
  ///
  /// Approximate under concurrency. Delegates to the vendored header's
  /// UseCount(); LLVM's naming stops at intrusive_ref_cnt_ptr.h.
  uint32_t UseCount() const { return impl_.UseCount(); }

  /// \brief Returns a human-readable description for diagnostics.
  std::string ToString() const;

 private:
  explicit Tensor(IntrusiveRefCntPtr<TensorImpl> impl) : impl_(std::move(impl)) {}

  IntrusiveRefCntPtr<TensorImpl> impl_;
};

static_assert(sizeof(Tensor) == sizeof(void*), "Tensor must stay a single-pointer handle");

}  // namespace inferx
