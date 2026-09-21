#include "inferx/core/tensor.h"

#include "absl/strings/str_cat.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

StatusOr<Tensor> Tensor::Empty(DataType dtype, const Shape& shape, Allocator* allocator) {
  if (allocator == nullptr) return InvalidArgumentError("null allocator");

  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());

  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  INFERX_ASSIGN_OR_RETURN(StoragePtr storage,
                          Storage::Allocate(static_cast<size_t>(bytes), allocator));

  return Tensor(makeIntrusiveRefCnt<TensorImpl>(std::move(storage), 0, dtype, shape));
}

StatusOr<Tensor> Tensor::Empty(DataType dtype, const Shape& shape, DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(Allocator * alloc, AllocatorFor(device));

  return Empty(dtype, shape, alloc);
}

StatusOr<Tensor> Tensor::FromBlob(void* data, DataType dtype, const Shape& shape,
                                  DeviceId device) {
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());
  if (data == nullptr && shape.Numel() != 0) {
    return InvalidArgumentError("null data pointer for non-empty tensor ", shape.ToString());
  }
  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  StoragePtr storage = Storage::Borrow(data, static_cast<size_t>(bytes), device);
  return Tensor(makeIntrusiveRefCnt<TensorImpl>(std::move(storage), 0, dtype, shape));
}

StatusOr<Tensor> Tensor::FromStorage(StoragePtr storage, int64_t offset, DataType dtype,
                                     const Shape& shape) {
  if (!storage) {
    return InvalidArgumentError("null storage");
  }
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());

  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  const int64_t capacity = static_cast<int64_t>(storage->Size());

  // Compared against remaining capacity rather than by summing first, so a
  // large offset or size cannot wrap into an apparently valid range.
  if (offset < 0 || offset > capacity || bytes > capacity - offset) {
    return OutOfRangeError("tensor of ", bytes, " bytes at offset ", offset,
                           " does not fit in ", storage->ToString());
  }
  return Tensor(makeIntrusiveRefCnt<TensorImpl>(std::move(storage), offset, dtype, shape));
}

StatusOr<Tensor> Tensor::Slice(int64_t begin, int64_t end) const {
  if (!IsDefined()) {
    return FailedPreconditionError("Slice on an undefined tensor");
  }
  const Shape& shape = impl_->GetShape();
  if (shape.Rank() == 0) {
    return InvalidArgumentError("cannot slice a rank-0 tensor");
  }

  const int64_t extent = shape.Dim(0);
  if (begin < 0 || end < begin || end > extent) {
    return OutOfRangeError("slice [", begin, ", ", end,
                           ") out of range for dimension 0 of extent ", extent);
  }

  int64_t inner = 1;
  for (int i = 1; i < shape.Rank(); ++i) inner *= shape.Dim(i);

  // Computed in bits rather than bytes so that a packed sub-byte row whose
  // offset lands mid-byte is caught here instead of silently truncating.
  const int64_t offset_bits =
      static_cast<int64_t>(DataTypeStorageBits(GetDataType())) * begin * inner;

  if (offset_bits % 8 != 0) {
    return InvalidArgumentError("slice offset for ", DataTypeName(GetDataType()),
                                " is not byte-aligned; begin=", begin, " inner=", inner);
  }

  Shape sliced = shape;
  sliced.SetDim(0, end - begin);
  return FromStorage(impl_->GetStorage(), StorageOffset() + offset_bits / 8, GetDataType(),
                     sliced);
}

StatusOr<Tensor> Tensor::Reshape(const Shape& new_shape) const {
  if (!IsDefined()) {
    return FailedPreconditionError("Reshape on an undefined tensor");
  }
  if (new_shape.Numel() != Numel()) {
    return InvalidArgumentError("reshape ", GetShape().ToString(), " -> ", new_shape.ToString(),
                                " changes element count (", Numel(), " vs ", new_shape.Numel(),
                                ")");
  }
  INFERX_RETURN_IF_ERROR(TensorSpec(GetDataType(), new_shape).Verify());

  return FromStorage(impl_->GetStorage(), StorageOffset(), GetDataType(), new_shape);
}

StatusOr<Tensor> Tensor::Bitcast(DataType new_dtype) const {
  if (!IsDefined()) {
    return FailedPreconditionError("Bitcast on an undefined tensor");
  }
  if (!DataTypeIsValid(new_dtype)) {
    return InvalidArgumentError("bitcast target dtype ", DataTypeName(new_dtype),
                                " has no layout");
  }

  const int64_t src_bits = static_cast<int64_t>(DataTypeStorageBits(GetDataType())) * Numel();
  const int64_t dst_bits = static_cast<int64_t>(DataTypeStorageBits(new_dtype));

  if (src_bits % dst_bits != 0) {
    return InvalidArgumentError("cannot bitcast ", DataTypeName(GetDataType()), " ",
                                GetShape().ToString(), " to ", DataTypeName(new_dtype), ": ",
                                src_bits, " bits is not a multiple of ", dst_bits);
  }

  // Collapses to rank 1. Preserving the original shape would be wrong whenever
  // the element widths differ, and guessing which dimension absorbs the change
  // is worse than making the caller reshape explicitly.
  const Shape out{src_bits / dst_bits};
  INFERX_RETURN_IF_ERROR(TensorSpec(new_dtype, out).Verify());

  return FromStorage(impl_->GetStorage(), StorageOffset(), new_dtype, out);
}

std::string Tensor::ToString() const {
  if (!IsDefined()) return "Tensor(<undefined>)";

  return absl::StrCat("Tensor(", DataTypeName(GetDataType()), ", ",
                      impl_->GetShape().ToString(), ", ", Device().ToString(), ", ", NBytes(),
                      "B, refs=", UseCount(), ", ", impl_->GetStorage()->ToString(), ")");
}

}  // namespace inferx
