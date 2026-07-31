#include "inferx/core/tensor_view.h"

#include "absl/strings/str_cat.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

Status CheckRankFitsView(const Shape& shape) {
  if (shape.Rank() > TensorView::kMaxRank) {
    return InvalidArgumentError("tensor rank ", shape.Rank(),
                                " exceeds the maximum of ",
                                TensorView::kMaxRank, " in ", shape.ToString());
  }

  return OkStatus();
}

StatusOr<TensorView> TensorView::Create(void* data, DataType dtype,
                                        const Shape& shape, DeviceId device) {
  INFERX_RETURN_IF_ERROR(CheckRankFitsView(shape));
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());

  if (data == nullptr && shape.Numel() != 0) {
    return InvalidArgumentError("null data pointer for non-empty tensor ",
                                shape.ToString());
  }

  return TensorView(data, dtype, shape, device);
}

StatusOr<TensorView> TensorView::Slice(int64_t begin, int64_t end) const {
  if (rank_ == 0) {
    return InvalidArgumentError("cannot slice a rank-0 tensor");
  }

  const int64_t extent = dims_[0];

  if (begin < 0 || end < begin || end > extent) {
    return OutOfRangeError("slice [", begin, ", ", end,
                           ") out of range for dimension 0 of extent ", extent);
  }

  int64_t inner = 1;

  for (int i = 1; i < rank_; ++i) inner *= dims_[i];

  // Computed in bits rather than bytes so that a packed sub-byte row whose
  // offset lands mid-byte is caught here instead of silently truncating.
  const int64_t offset_bits =
      static_cast<int64_t>(DataTypeStorageBits(dtype_)) * begin * inner;

  if (offset_bits % 8 != 0) {
    return InvalidArgumentError("slice offset for ", DataTypeName(dtype_),
                                " is not byte-aligned; begin=", begin,
                                " inner=", inner);
  }

  TensorView out = *this;
  out.data_ = Bytes() + offset_bits / 8;
  out.dims_[0] = end - begin;

  return out;
}

StatusOr<TensorView> TensorView::Reshape(const Shape& shape) const {
  if (shape.Numel() != Numel()) {
    return InvalidArgumentError("reshape ", GetShape().ToString(), " -> ",
                                shape.ToString(), " changes element count (",
                                Numel(), " vs ", shape.Numel(), ")");
  }

  INFERX_RETURN_IF_ERROR(CheckRankFitsView(shape));
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype_, shape).Verify());

  return TensorView(data_, dtype_, shape, device_);
}

StatusOr<TensorView> TensorView::Bitcast(DataType dtype) const {
  if (!DataTypeIsValid(dtype)) {
    return InvalidArgumentError("bitcast target dtype ", DataTypeName(dtype),
                                " has no layout");
  }

  const int64_t src_bits =
      static_cast<int64_t>(DataTypeStorageBits(dtype_)) * Numel();
  const int64_t dst_bits = static_cast<int64_t>(DataTypeStorageBits(dtype));

  if (src_bits % dst_bits != 0) {
    return InvalidArgumentError("cannot bitcast ", DataTypeName(dtype_), " ",
                                GetShape().ToString(), " to ",
                                DataTypeName(dtype), ": ", src_bits,
                                " bits is not a multiple of ", dst_bits);
  }

  // Collapses to rank 1. Preserving the original shape would be wrong whenever
  // the element widths differ, and guessing which dimension absorbs the change
  // is worse than making the caller reshape explicitly.
  const Shape out{src_bits / dst_bits};
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, out).Verify());

  return TensorView(data_, dtype, out, device_);
}

std::string TensorView::ToString() const {
  if (!IsDefined()) return "TensorView(<undefined>)";

  return absl::StrCat("TensorView(", DataTypeName(dtype_), ", ",
                      GetShape().ToString(), ", ", device_.ToString(), ", ",
                      NBytes(), "B)");
}

}  // namespace inferx
