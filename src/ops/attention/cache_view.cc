#include "inferx/ops/attention/cache_view.h"

#include <variant>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/ops/attention/cache_spec_bridge.h"

namespace inferx::attn {
namespace {

/// Token-wise mechanisms are the ones whose cache grows per token.
bool IsTokenWise(const AttentionSpec& spec) {
  return std::holds_alternative<DenseParams>(spec.params) ||
         std::holds_alternative<MlaParams>(spec.params) ||
         std::holds_alternative<SparseParams>(spec.params);
}

}  // namespace

StatusOr<CacheView> BindTokenWiseView(const KvBlockPool& pool, const AttentionSpec& spec,
                                      int64_t layer, int64_t block_size) {
  if (!IsTokenWise(spec)) {
    return InvalidArgumentError(
        "token-wise cache views need a token-wise mechanism (Dense/Mla/Sparse)");
  }
  if (spec.kv_layout == KvPageLayout::kRecurrentFlat) {
    return InvalidArgumentError("kRecurrentFlat is a recurrent-state layout; bind with "
                                "BindRecurrentView");
  }
  if (block_size != pool.BlockSize()) {
    return InvalidArgumentError("block size ", block_size, " differs from the pool's ",
                                pool.BlockSize());
  }

  INFERX_ASSIGN_OR_RETURN(const Tensor region, pool.LayerCache(layer));
  // [num_blocks + 1, block_size, bytes_per_token]: row 0 is the null
  // block's reserved row; the stride between block rows is one page.
  if (region.Rank() != 3 || region.Dim(1) != pool.BlockSize()) {
    return InternalError("unexpected layer-cache shape for layer ", layer);
  }
  const int64_t bytes_per_token = region.Dim(2);

  // Cross-check the pool against the spec so a pool built from different
  // geometry fails loudly here, not silently in kernel strides.
  INFERX_ASSIGN_OR_RETURN(const int64_t expected, CacheBytesPerToken(spec));
  if (bytes_per_token != expected) {
    return InvalidArgumentError("pool layer ", layer, " holds ", bytes_per_token,
                                " bytes/token but the spec needs ", expected,
                                "; was the pool built from these specs?");
  }

  CacheView view;
  view.base = static_cast<std::byte*>(region.Data());
  view.block_tokens = static_cast<std::int32_t>(pool.BlockSize());
  view.page_bytes = static_cast<std::int32_t>(pool.BlockSize() * bytes_per_token);
  view.page_stride_bytes = view.page_bytes;
  view.layout = spec.kv_layout;
  view.dtype = spec.kv_dtype;
  return view;
}

StatusOr<CacheView> BindRecurrentView(const RecurrentStatePool& pool,
                                      const AttentionSpec& spec, int64_t layer,
                                      int64_t component) {
  if (!std::holds_alternative<LinearParams>(spec.params)) {
    return InvalidArgumentError("recurrent cache views need a Linear mechanism");
  }
  INFERX_ASSIGN_OR_RETURN(const Tensor region, pool.ComponentView(layer, component));
  // [max_slots, bytes_per_slot]: one contiguous region per component.
  if (region.Rank() != 2) {
    return InternalError("unexpected component-view shape for layer ", layer);
  }

  CacheView view;
  view.base = static_cast<std::byte*>(region.Data());
  view.slot_stride_bytes = static_cast<std::int32_t>(region.Dim(1));
  view.layout = KvPageLayout::kRecurrentFlat;
  view.dtype = spec.kv_dtype;
  return view;
}

}  // namespace inferx::attn
