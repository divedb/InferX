#ifndef INFERX_ATTN_KERNEL_H_
#define INFERX_ATTN_KERNEL_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "inferx/core/status.h"
#include "inferx/ops/attention/cache_view.h"
#include "inferx/ops/attention/metadata.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/ops/attention/types.h"
#include "inferx/ops/op_context.h"

namespace inferx::attn {

/// Defined in backend.h; kernels only need it as an opaque limit bundle.
struct EngineLimits;

/// \brief Operands for one layer. Which slots are used is defined by the
///        mechanism (docs/attention.md §6.1).
struct Inputs {
  /// [T, Hq, D] packed varlen; strides arbitrary on H/D.
  TensorView q;
  /// [T, Hkv, D]; !Defined() => "already in cache" (decode after prefill).
  TensorView k, v;
  /// Mechanism-specific extras.
  std::array<TensorView, 2> ext{};
};

/// \brief Everything one layer's launch needs. All members are views or
///        PODs; built on the caller's stack.
struct RunArgs {
  /// This layer's cache-group metadata for this step.
  const GroupMeta& meta;
  /// Bound once at init.
  const CacheView& cache;
  /// Runtime + stream, as the ops layer receives them.
  ops::OpContext& ctx;
  const Inputs& in;
  /// Caller-allocated (graph-friendly).
  TensorView out;
  WorkspaceRef workspace;
  /// Host mirror of the batch (null under graph replay); planners use it.
  const BatchDesc* desc = nullptr;
};

class Kernel {
 public:
  virtual ~Kernel() = default;

  /// \brief Hot path. No allocation, no locking.
  [[nodiscard]] virtual Status Run(const RunArgs& args) const noexcept = 0;
  [[nodiscard]] virtual KvUpdateMode GetKvUpdateMode() const noexcept = 0;
  /// \brief Upper bound of scratch bytes for any step up to the configured
  ///        limits (queried at init).
  [[nodiscard]] virtual std::size_t WorkspaceBytes(const EngineLimits& limits) const noexcept = 0;
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_KERNEL_H_
