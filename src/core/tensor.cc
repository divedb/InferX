#include "inferx/core/tensor.h"

#include "absl/strings/str_cat.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

StatusOr<Tensor> Tensor::Empty(DataType dtype, const Shape& shape,
                               Allocator* allocator) {
  if (allocator == nullptr) return InvalidArgumentError("null allocator");

  INFERX_RETURN_IF_ERROR(CheckRankFitsView(shape));
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());

  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  INFERX_ASSIGN_OR_RETURN(
      StoragePtr storage,
      Storage::Allocate(static_cast<size_t>(bytes), allocator));

  return Tensor(
      makeIntrusiveRefCnt<TensorImpl>(std::move(storage), 0, dtype, shape));
}

StatusOr<Tensor> Tensor::Empty(DataType dtype, const Shape& shape,
                               DeviceId device) {
  INFERX_ASSIGN_OR_RETURN(Allocator * alloc, AllocatorFor(device));

  return Empty(dtype, shape, alloc);
}

StatusOr<Tensor> Tensor::FromBlob(void* data, DataType dtype,
                                  const Shape& shape, DeviceId device) {
  INFERX_RETURN_IF_ERROR(CheckRankFitsView(shape));
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());
  if (data == nullptr && shape.Numel() != 0) {
    return InvalidArgumentError("null data pointer for non-empty tensor ",
                                shape.ToString());
  }
  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  StoragePtr storage =
      Storage::Borrow(data, static_cast<size_t>(bytes), device);
  return Tensor(
      makeIntrusiveRefCnt<TensorImpl>(std::move(storage), 0, dtype, shape));
}

StatusOr<Tensor> Tensor::FromStorage(StoragePtr storage, int64_t offset,
                                     DataType dtype, const Shape& shape) {
  if (!storage) {
    return InvalidArgumentError("null storage");
  }
  INFERX_RETURN_IF_ERROR(CheckRankFitsView(shape));
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());

  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  const int64_t capacity = static_cast<int64_t>(storage->Size());

  // Compared against remaining capacity rather than by summing first, so a
  // large offset or size cannot wrap into an apparently valid range.
  if (offset < 0 || offset > capacity || bytes > capacity - offset) {
    return OutOfRangeError("tensor of ", bytes, " bytes at offset ", offset,
                           " does not fit in ", storage->ToString());
  }
  return Tensor(makeIntrusiveRefCnt<TensorImpl>(std::move(storage), offset,
                                                dtype, shape));
}

namespace {

// Slice/Reshape/Bitcast defer their arithmetic and validation to TensorView so
// the two layers cannot drift apart, then translate the resulting pointer back
// into an offset within the shared Storage.
StatusOr<Tensor> RebaseOnto(const TensorImpl& impl, const TensorView& derived) {
  const std::byte* origin = static_cast<const std::byte*>(impl.Data());
  const int64_t delta = derived.Bytes() - origin;
  return Tensor::FromStorage(impl.GetStorage(), impl.StorageOffset() + delta,
                             derived.GetDataType(), derived.GetShape());
}

}  // namespace

StatusOr<Tensor> Tensor::Slice(int64_t begin, int64_t end) const {
  if (!defined()) {
    return FailedPreconditionError("Slice on an undefined tensor");
  }
  INFERX_ASSIGN_OR_RETURN(TensorView derived, view().Slice(begin, end));
  return RebaseOnto(*impl_, derived);
}

StatusOr<Tensor> Tensor::Reshape(const Shape& shape) const {
  if (!defined()) {
    return FailedPreconditionError("Reshape on an undefined tensor");
  }
  INFERX_ASSIGN_OR_RETURN(TensorView derived, view().Reshape(shape));
  return RebaseOnto(*impl_, derived);
}

StatusOr<Tensor> Tensor::Bitcast(DataType dtype) const {
  if (!defined()) {
    return FailedPreconditionError("Bitcast on an undefined tensor");
  }
  INFERX_ASSIGN_OR_RETURN(TensorView derived, view().Bitcast(dtype));
  return RebaseOnto(*impl_, derived);
}

std::string Tensor::ToString() const {
  if (!defined()) return "Tensor(<undefined>)";

  return absl::StrCat("Tensor(", DataTypeName(dtype()), ", ",
                      impl_->GetShape().ToString(), ", ", device().ToString(),
                      ", ", nbytes(), "B, refs=", use_count(), ", ",
                      impl_->GetStorage()->ToString(), ")");
}

}  // namespace inferx
