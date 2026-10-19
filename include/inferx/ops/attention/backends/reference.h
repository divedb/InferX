#ifndef INFERX_ATTN_BACKENDS_REFERENCE_H_
#define INFERX_ATTN_BACKENDS_REFERENCE_H_

#include "inferx/ops/attention/backend.h"

namespace inferx::attn {

/// \brief Correctness oracle and bring-up backend: naive single-threaded
///        attention on the host, CPU devices only. Selectable only when
///        forced (tests, bring-up) so a missing kernel is never a silent
///        50x slowdown (docs/attention.md §7.3).
///
///        Implements the full dense family (MHA/MQA/GQA) under causal,
///        bidirectional, sliding-window, and chunked masks, with the fused
///        KV write into the byte-denominated pool layout. Stateless
///        (encoder-self) layers run without a cache view.
class ReferenceBackend final : public Backend {
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

#endif  // INFERX_ATTN_BACKENDS_REFERENCE_H_
