#ifndef INFERX_ATTN_BACKENDS_FLASH_DENSE_H_
#define INFERX_ATTN_BACKENDS_FLASH_DENSE_H_

#include "inferx/ops/attention/backend.h"

namespace inferx::attn {

/// \brief Dense-family backend over the existing FlashInfer wrappers
///        (ops::PagedAttention with its plan workspace, KV append via
///        ops::WritePagedKv). Causal, bidirectional, and sliding-window
///        masks; the internal dispatcher already falls back from the
///        FlashInfer fast path to the generic paged kernel for geometries
///        FlashInfer's template set does not cover.
///
///        CUDA devices only. Graph support is kNever for now: the wrappers
///        allocate metadata lazily on the host, which graph capture forbids;
///        the persistent-buffer path is future work (docs/attention.md §10).
class FlashDenseBackend final : public Backend {
 public:
  static constexpr std::array<std::uint32_t, 1> kMechanisms{
      static_cast<std::uint32_t>(Mechanism::kDense)};

  [[nodiscard]] const Capability& GetCapability() const noexcept override;
  [[nodiscard]] StatusOr<std::unique_ptr<Kernel>> CreateKernel(
      const AttentionSpec& spec, const CacheView& view,
      const EngineLimits& limits) const override;
  [[nodiscard]] StatusOr<std::unique_ptr<MetadataBuilder>> CreateBuilder(
      const AttentionSpec& spec, const EngineLimits& limits) const override;
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_BACKENDS_FLASH_DENSE_H_
