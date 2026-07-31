#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"

namespace inferx {

/// \brief A shape of a tensor, i.e. the number of elements along each
///        dimension.
///
/// The shape is stored as a vector of int64_t, with a small inline capacity for
/// common ranks. A shape of rank 0 is a scalar, and a shape of rank 1 with
/// extent 0 is a vector of length 0. Shapes of rank 2 or higher are matrices or
/// higher-dimensional tensors. The shape does not own the data it describes; it
/// is a lightweight value type that can be copied cheaply.
class Shape {
 public:
  /// The default rank for shapes, which is 8. Shapes with rank <= 8 are
  /// stored inline, while shapes with rank > 8 are heap-allocated.
  static constexpr int kDefaultRank = 8;

  using DimsContainer = absl::InlinedVector<int64_t, kDefaultRank>;

  /// \brief Constructs an empty shape.
  Shape() = default;

  /// \brief Constructs a shape from an initializer list of dimensions.
  ///
  /// \param dims The dimensions of the shape.
  Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}

  /// \brief Constructs a shape from a span of dimensions.
  ///
  /// \param dims The dimensions of the shape.
  explicit Shape(absl::Span<const int64_t> dims)
      : dims_(dims.begin(), dims.end()) {}

  /// \brief Get the rank of the shape, i.e. the number of dimensions.
  ///
  /// \return The rank of the shape.
  int Rank() const { return static_cast<int>(dims_.size()); }

  /// \brief Check if the shape is a scalar, i.e. has rank 0.
  ///
  /// \return True if the shape is a scalar; otherwise false.
  bool IsScalar() const { return dims_.empty(); }

  /// \brief Get the extent of the shape along a given dimension.
  ///
  /// \param i The dimension index, which must be in the range
  ///          [-Rank(), Rank()).
  /// \return  The extent of the shape along dimension `i`.
  int64_t Dim(int i) const { return dims_[Normalize(i)]; }

  /// \brief Set the extent of the shape along a given dimension.
  ///
  /// \param i     The dimension index, which must be in the range
  ///              [-Rank(), Rank()).
  /// \param value The new extent of the shape along dimension `i`.
  void SetDim(int i, int64_t value) { dims_[Normalize(i)] = value; }

  /// \brief Append a new dimension to the shape with the given extent.
  ///
  /// \param d The extent of the new dimension, which must be positive.
  void PushBack(int64_t d) {
    assert(d > 0 && "Dimension extent must be positive");

    dims_.push_back(d);
  }

  /// \brief Clears all dimensions from the shape.
  void Clear() { dims_.clear(); }

  /// \brief Returns a span of the dimensions of the shape.
  ///
  /// \return A span of the dimensions of the shape.
  absl::Span<const int64_t> Dims() const { return absl::MakeConstSpan(dims_); }

  auto begin() const { return dims_.begin(); }
  auto end() const { return dims_.end(); }

  /// \brief Returns the number of elements in the shape.
  ///
  /// \return The number of elements in the shape.
  int64_t Numel() const {
    return std::accumulate(dims_.begin(), dims_.end(), int64_t{1},
                           std::multiplies<>());
  }

  /// \brief Returns the number of elements in the shape below a given
  /// dimension.
  ///
  /// \param i The dimension index, which must be in the range
  ///          [-Rank(), Rank()).
  /// \return  The number of elements in the shape below dimension `i`.
  int64_t InnerNumel(int i) const {
    assert(i >= 0 && i < Rank() &&
           "Dimension index must be in the range [0, Rank())");

    return std::accumulate(dims_.begin() + i, dims_.end(), int64_t{1},
                           std::multiplies<>());
  }

  friend bool operator==(const Shape& a, const Shape& b) {
    return a.dims_ == b.dims_;
  }

  friend bool operator!=(const Shape& a, const Shape& b) { return !(a == b); }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& s) {
    return H::combine(
        H::combine_contiguous(std::move(h), s.dims_.data(), s.dims_.size()),
        s.dims_.size());
  }

  /// \brief Converts the shape to a string representation.
  ///
  /// \return A string representation of the shape, like [dim1, dim2, ..., dimN]
  std::string ToString() const;

 private:
  /// \brief Normalizes a dimension index to be non-negative.
  ///
  /// \param i The dimension index to normalize. Must be in the range
  ///          [-Rank(), Rank()).
  /// \return  The normalized dimension index, which is non-negative.
  int Normalize(int i) const {
    assert(i >= -Rank() && i < Rank() &&
           "Dimension index must be in the range [-Rank(), Rank())");

    return i < 0 ? Rank() + i : i;
  }

  DimsContainer dims_;
};

}  // namespace inferx
