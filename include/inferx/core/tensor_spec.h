#pragma once

#include <cstddef>
#include <utility>

#include "inferx/core/dtype.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"

namespace inferx {

/// \brief A validated (dtype, shape) pair, the layout half of a tensor.
class TensorSpec {
 public:
  /// \brief Constructs a spec with the zero-initialized dtype and an empty
  ///        shape.
  TensorSpec() = default;
  /// \brief Constructs a spec from a dtype and shape.
  ///
  /// \param dtype The element type.
  /// \param shape The shape.
  TensorSpec(DataType dtype, Shape shape) : dtype_(dtype), shape_(std::move(shape)) {}

  /// \brief Returns the spec's element type.
  DataType GetDataType() const { return dtype_; }
  /// \brief Returns the spec's shape.
  const Shape& GetShape() const { return shape_; }

  /// \brief Returns the number of elements in the spec.
  int64_t Numel() const { return shape_.Numel(); }

  /// \brief Returns the number of bytes this spec's elements occupy.
  ///
  /// Goes through DataTypeByteSize() rather than multiplying by an element
  /// size, so that packed sub-byte weights report their packed size.
  int64_t NBytes() const { return DataTypeByteSize(dtype_, shape_.Numel()); }

  /// \brief Checks the joint dtype/shape invariants of this spec.
  ///
  /// Verifies that no extent is negative, and -- for sub-byte dtypes -- the
  /// innermost extent is a multiple of the elements that pack into one byte.
  /// Rank itself is unchecked: a Shape holds any rank, and the tensor factories
  /// accept whatever TensorSpec::Verify passes.
  ///
  /// \return OK, or InvalidArgument naming the violated invariant.
  Status Verify() const {
    for (const int64_t extent : shape_.Dims()) {
      if (extent < 0) {
        return InvalidArgumentError("tensor shape ", shape_.ToString(),
                                    " has a negative extent");
      }
    }

    // Sub-byte types pack several elements per byte, so a row whose innermost
    // extent does not fill whole bytes would straddle a byte boundary and make
    // row addressing ill-defined. Reject that here rather than letting a loader
    // produce a tensor whose Slice() silently misaligns.
    if (DataTypeIsSubByte(dtype_) && shape_.Rank() > 0) {
      const int64_t inner = shape_.Dim(-1);
      const std::size_t per_byte = 8 / DataTypeBitWidth(dtype_);

      if (inner % static_cast<int64_t>(per_byte) != 0) {
        return InvalidArgumentError("sub-byte dtype ", DataTypeName(dtype_),
                                    " requires the innermost extent to be a multiple of ",
                                    per_byte, ", got ", inner, " in ", shape_.ToString());
      }
    }

    return OkStatus();
  }

  /// \brief Compares two specs for equal dtype and shape.
  friend bool operator==(const TensorSpec& a, const TensorSpec& b) {
    return a.dtype_ == b.dtype_ && a.shape_ == b.shape_;
  }

  /// \brief Compares two specs for inequality.
  friend bool operator!=(const TensorSpec& a, const TensorSpec& b) { return !(a == b); }

 private:
  DataType dtype_{};
  Shape shape_;
};

}  // namespace inferx
