#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "inferx/core/allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/intrusive_ref_cnt_ptr.h"
#include "inferx/core/status.h"
#include "inferx/core/storage.h"
#include "inferx/core/tensor_spec.h"
#include "inferx/core/tensor_view.h"

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
/// storage. data() is storage()->data() + offset(), in bytes.
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
      : storage_(std::move(storage)),
        offset_(offset),
        spec_(dtype, std::move(shape)) {}

  ~TensorImpl() = default;

  // GetStorage/GetDataType/GetShape rather than Storage/DataType/Shape: a
  // member function may not share a name with a type used in the same class.
  const StoragePtr& GetStorage() const { return storage_; }
  int64_t StorageOffset() const { return offset_; }  // bytes
  DataType GetDataType() const { return spec_.GetDataType(); }
  const Shape& GetShape() const { return spec_.GetShape(); }
  DeviceId Device() const { return storage_->Device(); }

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

/// \brief A Tensor is a handle to a TensorImpl.
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
  /// status on failure.
  static StatusOr<Tensor> Empty(DataType dtype, const Shape& shape,
                                DeviceId device);

  // Allocates from a specific allocator -- a per-rank workspace, say.
  static StatusOr<Tensor> Empty(DataType dtype, const Shape& shape,
                                Allocator* allocator);

  // Wraps memory owned elsewhere, freeing nothing. The caller guarantees the
  // bytes outlive every handle derived from this one. This is how mmap'd
  // weights and arena carve-outs enter the tensor world.
  static StatusOr<Tensor> FromBlob(void* data, DataType dtype,
                                   const Shape& shape, DeviceId device);

  // Shares an existing Storage at a byte offset. Bounds are checked against the
  // storage size.
  static StatusOr<Tensor> FromStorage(StoragePtr storage, int64_t offset,
                                      DataType dtype, const Shape& shape);

  /// \brief Constructs an empty Tensor.
  Tensor() = default;

  bool defined() const { return static_cast<bool>(impl_); }
  bool empty() const { return numel() == 0; }

  // Undefined-tensor access returns zero/empty rather than dereferencing null,
  // so a missing defined() check is a wrong answer, not a crash.
  void* data() const { return impl_ ? impl_->Data() : nullptr; }
  DataType dtype() const {
    return impl_ ? impl_->GetDataType() : DataType::kUndefined;
  }
  DeviceId device() const { return impl_ ? impl_->Device() : DeviceId::Cpu(); }
  Shape shape() const { return impl_ ? impl_->GetShape() : Shape{}; }
  int rank() const { return impl_ ? impl_->GetShape().Rank() : 0; }
  int64_t dim(int i) const { return impl_ ? impl_->GetShape().Dim(i) : 0; }
  int64_t numel() const { return impl_ ? impl_->GetShape().Numel() : 0; }
  int64_t nbytes() const { return impl_ ? impl_->NBytes() : 0; }
  int64_t storage_offset() const { return impl_ ? impl_->StorageOffset() : 0; }

  bool is_cpu() const { return device().IsCpu(); }
  bool is_cuda() const { return device().IsCuda(); }

  template <typename T>
  T* data_as() const {
    return dtype() == kDataTypeOf<T> ? static_cast<T*>(data()) : nullptr;
  }

  // The non-owning view for kernels and the step path. Valid only while this
  // Tensor -- or another handle to the same storage -- is alive.
  TensorView view() const {
    return impl_ ? TensorView(impl_->Data(), impl_->GetDataType(),
                              impl_->GetShape(), impl_->Device())
                 : TensorView();
  }

  // All three share storage with `*this`; none copies data.
  StatusOr<Tensor> Slice(int64_t begin, int64_t end) const;
  StatusOr<Tensor> Reshape(const Shape& shape) const;
  StatusOr<Tensor> Bitcast(DataType dtype) const;

  // True when both handles are backed by the same Storage -- that is, when
  // writing through one may be observed through the other.
  bool is_alias_of(const Tensor& other) const {
    return impl_ && other.impl_ &&
           impl_->GetStorage().Get() == other.impl_->GetStorage().Get();
  }

  // Number of handles to this tensor's impl. Approximate under concurrency.
  //
  // Kept in the engine's snake_case style rather than exposing the vendored
  // header's camelCase; LLVM's naming stops at intrusive_ref_cnt_ptr.h.
  uint32_t use_count() const { return impl_.UseCount(); }

  // Number of TensorImpls sharing the underlying bytes.
  uint32_t storage_use_count() const {
    return impl_ ? impl_->GetStorage().UseCount() : 0;
  }

  const StoragePtr& storage() const { return impl_->GetStorage(); }
  const IntrusiveRefCntPtr<TensorImpl>& impl() const { return impl_; }

  void reset() { impl_.Reset(); }

  std::string ToString() const;

 private:
  explicit Tensor(IntrusiveRefCntPtr<TensorImpl> impl)
      : impl_(std::move(impl)) {}

  IntrusiveRefCntPtr<TensorImpl> impl_;
};

static_assert(sizeof(Tensor) == sizeof(void*),
              "Tensor must stay a single-pointer handle");

}  // namespace inferx
