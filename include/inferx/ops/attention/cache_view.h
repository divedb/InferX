#ifndef INFERX_ATTN_CACHE_VIEW_H_
#define INFERX_ATTN_CACHE_VIEW_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "inferx/core/status.h"
#include "inferx/ops/attention/spec.h"

namespace inferx {

class KvBlockPool;
class RecurrentStatePool;

}  // namespace inferx

namespace inferx::attn {

/// \brief Kernel-facing view of one layer's cached bytes, bound once at
///        init. A POD mirror of what the pools expose; the pools remain the
///        owners and the single source of truth.
///
/// For token-wise layers this flattens `KvBlockPool::LayerCache(layer)`
/// (a [num_blocks + 1, block_size, bytes_per_token] uint8 region whose row
/// 0 is the null block's reserved row) plus the family's declared
/// `KvPageLayout`. For recurrent layers it flattens
/// `RecurrentStatePool::ComponentView(layer, component)`
/// ([max_slots, bytes_per_slot]).
struct CacheView {
  /// This layer's region.
  std::byte* base = nullptr;
  /// Page i -> page i+1 for this layer (token-wise kinds).
  std::int64_t page_stride_bytes = 0;
  /// plan.block_size (token-wise kinds).
  std::int32_t block_tokens = 0;
  /// Bytes of this layer's page (token-wise kinds).
  std::int32_t page_bytes = 0;
  /// Usable pages; rows in the region are num_pages + 1 (null row).
  std::int32_t num_pages = 0;
  /// Slot i -> slot i+1 (recurrent kinds).
  std::int32_t slot_stride_bytes = 0;
  KvPageLayout layout = KvPageLayout::kKvPlanarNhd;
  DataType dtype = DataType::kBFloat16;
  /// fp8/int8 scales (device), null if unquantized.
  const float* k_scale = nullptr;
  const float* v_scale = nullptr;

  /// \brief Address of block `id` (token-wise). Block 0 is kNullBlockId:
  /// its row exists but is never written; kernels skip null-block rows.
  [[nodiscard]] std::byte* Page(std::int64_t id) const noexcept {
    return base + id * page_stride_bytes;
  }
  /// \brief Address of slot `slot`'s state (recurrent).
  [[nodiscard]] std::byte* Slot(std::int64_t slot) const noexcept {
    return base + slot * slot_stride_bytes;
  }
};
static_assert(std::is_trivially_copyable_v<CacheView>);

/// \brief Binds a token-wise layer's view over the pool's raw region.
///
/// Cross-checks the pool's per-layer bytes against the spec's expected
/// bytes per token, so a pool built from different specs fails loudly at
/// init instead of producing silently wrong strides.
///
/// \param pool        The block pool owning this layer's region.
/// \param spec        The layer's attention spec (mechanism + kv layout).
/// \param layer       Model-layer index into the pool.
/// \param block_size  The cache plan's block size; must match the pool's.
[[nodiscard]] StatusOr<CacheView> BindTokenWiseView(const KvBlockPool& pool,
                                                    const AttentionSpec& spec, int64_t layer,
                                                    int64_t block_size);

/// \brief Binds one component of a recurrent layer's state view.
///
/// \param pool      The state pool owning this layer's regions.
/// \param spec      The layer's attention spec (must be Linear).
/// \param layer     Model-layer index belonging to the pool's group.
/// \param component Index into the group spec's components
///                  (state first, conv ring second when present).
[[nodiscard]] StatusOr<CacheView> BindRecurrentView(const RecurrentStatePool& pool,
                                                    const AttentionSpec& spec, int64_t layer,
                                                    int64_t component);

}  // namespace inferx::attn

#endif  // INFERX_ATTN_CACHE_VIEW_H_
