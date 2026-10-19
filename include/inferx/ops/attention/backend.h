#ifndef INFERX_ATTN_BACKEND_H_
#define INFERX_ATTN_BACKEND_H_

#include <concepts>
#include <memory>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/ops/attention/builder.h"
#include "inferx/ops/attention/capability.h"
#include "inferx/ops/attention/kernel.h"
#include "inferx/ops/attention/spec.h"

namespace inferx::attn {

/// \brief Static per-engine limits used for buffer sizing and, through
///        `device`, for backend specialization (the reference backend is
///        CPU-only; FlashInfer backends are CUDA-only).
struct EngineLimits {
  std::int32_t max_reqs = 0;
  std::int32_t max_tokens = 0;
  std::int32_t max_seq_len = 0;
  std::int32_t max_blocks_per_req = 0;
  /// Upper bound for split-KV style scheduling; fixed for graph capture.
  std::int32_t max_kv_splits = 0;
  /// Where kernels execute.
  Device device{};
};

/// \brief Implementation provider: a stateless capability value plus two
///        factories. No per-layer data (that lives in the Kernel), no
///        per-step data (that lives in GroupMeta) — trivially thread-safe.
class Backend {
 public:
  virtual ~Backend() = default;
  [[nodiscard]] virtual const Capability& GetCapability() const noexcept = 0;

  /// \brief Creates the per-layer executor bound to the layer's cache view.
  ///        `view` was flattened from the pool at init (BindTokenWiseView /
  ///        BindRecurrentView); kernels may capture its strides.
  [[nodiscard]] virtual StatusOr<std::unique_ptr<Kernel>> CreateKernel(
      const AttentionSpec& spec, const CacheView& view,
      const EngineLimits& limits) const = 0;

  /// \brief Creates the per-cache-group metadata builder.
  [[nodiscard]] virtual StatusOr<std::unique_ptr<MetadataBuilder>> CreateBuilder(
      const AttentionSpec& spec, const EngineLimits& limits) const = 0;
};

/// \brief Compile-time guard for backend authors: catches missing members
///        at the definition site.
template <class B>
concept BackendLike = std::derived_from<B, Backend> && std::default_initializable<B> &&
    requires(const B b) {
      { b.GetCapability() } -> std::convertible_to<const Capability&>;
    };

}  // namespace inferx::attn

#endif  // INFERX_ATTN_BACKEND_H_
