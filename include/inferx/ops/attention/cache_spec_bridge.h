#ifndef INFERX_ATTN_CACHE_SPEC_BRIDGE_H_
#define INFERX_ATTN_CACHE_SPEC_BRIDGE_H_

#include <span>

#include "inferx/cache/cache_spec.h"
#include "inferx/core/status.h"
#include "inferx/ops/attention/spec.h"

namespace inferx::attn {

/// \brief Whether the layer needs persistent cache storage at all.
///
/// Encoder-self attention is stateless within one forward: it contributes
/// `NoCache` to the CacheRequirements, not a CacheSpec.
[[nodiscard]] bool NeedsCache(const AttentionSpec& spec) noexcept;

/// \brief The cache requirements of one attention layer, derived from its
///        spec. Pure; init time only.
///
/// Mapping: the mechanism picks the components (GqaKvComponent /
/// MlaLatentComponent / FixedSlot), the mask picks the retention (full,
/// sliding window with sinks, chunked), and Role::kCross picks encoder
/// retention. Callers should check NeedsCache() first: encoder-self specs
/// are rejected here.
[[nodiscard]] StatusOr<CacheSpec> BuildCacheSpec(const AttentionSpec& spec);

/// \brief Bytes of cached state one token of this spec occupies (per layer).
///
/// The token-wise mechanisms only; Linear and Custom specs are errors (their
/// state is per-sequence, not per-token). Used by BuildCacheSpec's callers
/// and by the CacheView binding to cross-check a pool against its spec.
[[nodiscard]] StatusOr<int64_t> CacheBytesPerToken(const AttentionSpec& spec);

/// \brief Assembles a whole model's CacheRequirements from its attention
///        specs: encoder-self layers contribute NoCache, everything else
///        its BuildCacheSpec result. One call, one source of truth.
[[nodiscard]] StatusOr<CacheRequirements> BuildCacheRequirements(
    std::span<const AttentionSpec> specs);

}  // namespace inferx::attn

#endif  // INFERX_ATTN_CACHE_SPEC_BRIDGE_H_
