#include "inferx/core/arena.h"

#include <bit>

#include "inferx/common/align.h"
#include "inferx/common/math_utils.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

StatusOr<void*> BumpArena::Allocate(size_t bytes, size_t alignment) {
  if (base_ == nullptr) {
    return FailedPreconditionError("BumpArena is not initialized");
  }

  if (alignment == 0 || !std::has_single_bit(alignment)) {
    return InvalidArgumentError("alignment ", alignment,
                                " is not a power of two");
  }

  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base_);
  const uintptr_t aligned_addr = AlignUp(base_addr + offset_, alignment);
  const size_t aligned = static_cast<size_t>(aligned_addr - base_addr);

  const auto new_offset = CheckedAdd(aligned, bytes);

  if (!new_offset || *new_offset > capacity_) {
    return ResourceExhaustedError("BumpArena exhausted on ", device_.ToString(),
                                  ": requested ", bytes, " bytes (aligned to ",
                                  alignment, ") with ", capacity_ - offset_,
                                  " of ", capacity_, " remaining");
  }

  offset_ = *new_offset;

  if (offset_ > peak_) peak_ = offset_;

  return static_cast<void*>(base_ + aligned);
}

StatusOr<TensorView> BumpArena::AllocateView(DataType dtype,
                                             const Shape& shape) {
  INFERX_RETURN_IF_ERROR(CheckRankFitsView(shape));
  INFERX_RETURN_IF_ERROR(TensorSpec(dtype, shape).Verify());

  const int64_t bytes = DataTypeByteSize(dtype, shape.Numel());
  INFERX_ASSIGN_OR_RETURN(void* p, Allocate(static_cast<size_t>(bytes)));
  return TensorView::Create(p, dtype, shape, device_);
}

}  // namespace inferx
