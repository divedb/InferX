#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

#include "absl/types/span.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

/// \brief Checks that `shape` fits in a TensorView's inline extent array.
///
/// This is TensorView's POD-storage limit rather than a property of the shape,
/// so `Shape` and `TensorSpec` both accept ranks this rejects. It is exposed
/// because every tensor factory has to agree on it: a shape that cannot round
/// trip through a TensorView would be silently truncated by `Tensor::View()`.
///
/// \param shape The shape to check.
/// \return      OK, or InvalidArgument naming the rank and the limit.
Status CheckRankFitsView(const Shape& shape);

/// \brief A non-owning view over a tensor's data, shape, and device.
///
/// TensorView is trivially copyable and can be passed by value. It does not
/// own the underlying data; the caller must ensure that the data outlives the
/// TensorView. TensorView is not thread-safe; callers must synchronize access
/// if multiple threads may access the same TensorView concurrently.
///
/// This is the type at the kernel boundary, and the type on the per-step decode
/// path. It carries no refcount because a `__global__` parameter must be
/// trivially copyable -- an owning Tensor holding an IntrusiveRefCntPtr can
/// never be passed to a kernel -- and because an atomic increment per tensor
/// copy would be pure overhead per decode step.
///
/// The rule: ownership is established once, at weight load or request
/// admission, as a Tensor. Everything downstream of that passes TensorView. A
/// TensorView is only valid while some Tensor (or arena, or DeviceBuffer) keeps
/// its bytes alive.
///
/// Views are contiguous and row-major, deliberately. A strided view would let
/// us express arbitrary slicing, but every kernel would then have to handle the
/// strided case or silently assume it away. The only slicing offered is along
/// dimension 0 -- which stays contiguous -- and anything else is an explicit
/// copy or a purpose-built kernel. See docs/ARCHITECTURE.md section 7.1.
class TensorView {
 public:
  /// The engine's real rank limit, and the only one there is.
  ///
  /// Unlike Shape -- which is generic and has no maximum -- TensorView stores
  /// its extents as a POD array, because a `__global__` parameter must be
  /// trivially copyable. That is a genuine constraint, so this is where the
  /// limit belongs.
  ///
  /// 8 is chosen, not derived: the deepest tensor the engine describes is the
  /// paged KV pool at rank 6 (docs/ARCHITECTURE.md section 6.2), so this is two
  /// dimensions of headroom, costing 64 of TensorView's 80 bytes at the launch
  /// boundary. Raising it is a deliberate trade against that parameter size,
  /// which is why it is a number here rather than an alias for Shape's inline
  /// capacity -- that constant is a host allocation knob and means something
  /// else. The assertion below keeps the two in the required relationship
  /// without making one silently follow the other.
  static constexpr int kMaxRank = 8;

  static_assert(kMaxRank <= Shape::kDefaultRank,
                "a shape a TensorView accepts must also be allocation-free on "
                "the host, so Shape's inline capacity must cover kMaxRank");

  /// \brief Constructs an undefined view over no data.
  TensorView() = default;

  /// \brief Constructs a TensorView without validating it.
  ///
  /// Prefer `Create()` at boundaries; this exists for hot paths that have
  /// already validated their inputs. A shape exceeding `kMaxRank` is truncated
  /// here rather than reported -- callers of this constructor are asserting
  /// they have already validated, and `Create()` rejects such shapes before
  /// they can reach a TensorView.
  ///
  /// \param data   Pointer to the tensor's data.
  /// \param dtype  Data type of the tensor.
  /// \param shape  Shape of the tensor.
  /// \param device Device on which the tensor resides.
  TensorView(void* data, DataType dtype, const Shape& shape, DeviceId device)
      : data_(data), dtype_(dtype), device_(device) {
    rank_ = shape.Rank() > kMaxRank ? kMaxRank : shape.Rank();

    for (int i = 0; i < rank_; ++i) dims_[i] = shape.Dim(i);
  }

  /// \brief Creates a TensorView with the specified data pointer, data type,
  ///        shape, and device.
  ///
  /// Validates the dtype, the rank, the extents, and sub-byte packing before
  /// constructing. This is the factory that should appear in loaders,
  /// config-driven code, and anything reachable from a network request.
  ///
  /// \param data   Pointer to the tensor's data.
  /// \param dtype  Data type of the tensor.
  /// \param shape  Shape of the tensor.
  /// \param device Device on which the tensor resides.
  /// \return       A StatusOr containing the created TensorView or an error
  ///               status.
  static StatusOr<TensorView> Create(void* data, DataType dtype,
                                     const Shape& shape, DeviceId device);

  /// \brief Creates a subview of the tensor by slicing along the first
  ///        dimension.
  ///
  /// The result stays contiguous, which is why dimension 0 is the only one that
  /// can be sliced.
  ///
  /// \param begin The starting index of the slice (inclusive).
  /// \param end   The ending index of the slice (exclusive).
  /// \return      A StatusOr containing the sliced TensorView or an error
  ///              status.
  StatusOr<TensorView> Slice(int64_t begin, int64_t end) const;

  // Same storage, new shape. Element count must match exactly; this never
  // allocates or copies.
  StatusOr<TensorView> Reshape(const Shape& shape) const;

  // Reinterpret the buffer as another dtype. Total byte count must match, which
  // rules out most accidental misuse while still allowing the legitimate cases
  // (viewing a packed i4 weight block as u8 for a memcpy, for instance).
  StatusOr<TensorView> Bitcast(DataType dtype) const;

  bool IsDefined() const { return data_ != nullptr && DataTypeIsValid(dtype_); }
  bool IsEmpty() const { return Numel() == 0; }

  void* Data() const { return data_; }
  std::byte* Bytes() const { return static_cast<std::byte*>(data_); }

  /// \brief Typed access, checked against the view's dtype.
  ///
  /// Returns nullptr on a mismatch rather than aborting, so callers in
  /// non-fatal paths can degrade. Hot paths that already know their dtype
  /// should use `Data()` and cast.
  ///
  /// \return A pointer to the data as `T*`, or nullptr if `T` is not the
  ///         view's element type.
  template <typename T>
  T* DataAs() const {
    return dtype_ == kDataTypeOf<T> ? static_cast<T*>(data_) : nullptr;
  }

  // Named GetDataType/GetShape rather than DataType/Shape because a member
  // function may not share a name with a type used in the same class -- the
  // accessors would change what `DataType` and `Shape` mean partway through the
  // declaration. TensorSpec spells them the same way.
  DataType GetDataType() const { return dtype_; }

  int Rank() const { return rank_; }

  /// \brief Zero-copy access to the extents.
  ///
  /// Prefer this on hot paths; `GetShape()` materializes a Shape and is for
  /// validation and diagnostics.
  absl::Span<const int64_t> Dims() const {
    return absl::MakeConstSpan(dims_.data(), rank_);
  }

  /// \brief Materializes a host-side Shape.
  ///
  /// Allocation-free for any rank a TensorView can hold, since Shape's inline
  /// capacity is at least kMaxRank.
  Shape GetShape() const { return Shape(Dims()); }

  /// \brief Returns the extent along dimension `i`.
  ///
  /// Negative indices count from the back. Out-of-range reads are clamped
  /// rather than undefined, and a rank-0 view answers 0.
  int64_t Dim(int i) const {
    if (rank_ == 0) return 0;

    int j = i < 0 ? rank_ + i : i;

    if (j < 0) j = 0;
    if (j >= rank_) j = rank_ - 1;

    return dims_[j];
  }

  int64_t Numel() const {
    int64_t n = 1;

    for (int i = 0; i < rank_; ++i) n *= dims_[i];

    return n;
  }

  int64_t NBytes() const { return DataTypeByteSize(dtype_, Numel()); }

  bool IsCpu() const { return device_.IsCpu(); }
  bool IsCuda() const { return device_.IsCuda(); }
  DeviceId Device() const { return device_; }

  std::string ToString() const;

 private:
  // Extents are stored inline as a POD rather than as a Shape. Shape is backed
  // by absl::InlinedVector, which has a user-provided copy constructor and
  // destructor and so is not trivially copyable -- embedding it here would both
  // break the kernel-launch contract below and put a copy that is theoretically
  // able to allocate on the decode path. TensorSpec is used for validation,
  // where materializing a Shape is free of consequence.
  void* data_ = nullptr;
  std::array<int64_t, kMaxRank> dims_{};
  int32_t rank_ = 0;
  DataType dtype_ = DataType::kUndefined;
  DeviceId device_ = DeviceId::Cpu();
};

static_assert(std::is_trivially_copyable_v<TensorView>,
              "TensorView is passed by value through kernel launch paths");

}  // namespace inferx
