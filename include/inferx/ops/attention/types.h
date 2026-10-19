#ifndef INFERX_ATTN_TYPES_H_
#define INFERX_ATTN_TYPES_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "inferx/core/dtype.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor.h"

namespace inferx::attn {

/// \brief Non-owning, trivially-copyable tensor descriptor for the kernel
///        boundary and CUDA-graph capture.
///
/// The owning type at model boundaries is `inferx::Tensor`; this POD exists
/// because captured graphs and device-side launches need flat descriptors,
/// and because backends must not hold refcounted handles alive implicitly.
/// Strides are in elements. Rank <= 4 covers every attention operand.
struct TensorView {
  void* data = nullptr;
  DataType dtype = DataType::kBFloat16;
  std::uint8_t rank = 0;
  std::array<std::int64_t, 4> sizes{};
  std::array<std::int64_t, 4> strides{};

  [[nodiscard]] constexpr bool Defined() const noexcept { return data != nullptr; }
  [[nodiscard]] constexpr std::int64_t Dim(int i) const noexcept { return sizes[i]; }
  [[nodiscard]] constexpr std::int64_t Stride(int i) const noexcept { return strides[i]; }
};
static_assert(std::is_trivially_copyable_v<TensorView>);

/// \brief Flattens a contiguous row-major `Tensor` into a `TensorView` over
///        the same storage. The tensor must outlive the view (step lifetime).
[[nodiscard]] inline TensorView ViewOf(const Tensor& tensor) noexcept {
  TensorView view;
  view.data = tensor.Data();
  view.dtype = tensor.GetDataType();
  view.rank = static_cast<std::uint8_t>(tensor.Rank());
  std::int64_t stride = 1;
  for (int i = tensor.Rank() - 1; i >= 0; --i) {
    view.sizes[i] = tensor.Dim(i);
    view.strides[i] = stride;
    stride *= tensor.Dim(i);
  }
  return view;
}

/// \brief Scratch memory handed out by the runtime; never allocated by
///        kernels on the step path.
struct WorkspaceRef {
  void* data = nullptr;
  std::size_t bytes = 0;
};

/// \brief How a backend interacts with CUDA graphs (docs/attention.md §6.4).
enum class GraphSupport : std::uint8_t {
  /// Never captured; eager only.
  kNever,
  /// May be captured for uniform decode-only batches (fixed query_len),
  /// padded to captured sizes.
  kUniformDecode,
  /// May be captured for any shape within captured limits (rare; requires
  /// fully device-driven kernels).
  kAny,
};

/// \brief Whether a kernel writes new K/V itself or expects a separate
///        layout-aware writer to have run (docs/attention.md §6.1).
enum class KvUpdateMode : std::uint8_t {
  /// Kernel writes in.k / in.v into the cache as part of Run().
  kFused,
  /// AttentionOp calls a layout-aware writer first (ops::WritePagedKv for
  /// paged GQA); the kernel only reads the cache.
  kSeparate,
  /// No per-token cache write (recurrent state lives inside the kernel;
  /// encoder-only layers).
  kNone,
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_TYPES_H_
