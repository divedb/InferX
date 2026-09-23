#pragma once
#include "inferx/ops/flash_attention.h"
namespace inferx::ops::cuda {
Status PrepareFlashDecode(ExecutionContext& ctx, const Tensor& kv_indptr,
                          const Tensor& last_page_len, int block_size,
                          FlashDecodeWorkspace& workspace);
// Graph-safe GPU planning, reused across layers; 64 grouped query rows/tile.
Status PrepareFlashAttention(ExecutionContext& ctx, const Tensor& qo_indptr, Tensor& plan,
                             int group, int tiles, int tile_rows);
Status FlashPagedAttention(ExecutionContext& ctx, const Tensor& q, const Tensor& qo,
                           const Tensor& kv, const Tensor& indices, const Tensor& last_page_len,
                           const Tensor& key, const Tensor& value, int64_t block_size,
                           const AttentionParams& params, const Tensor& plan, int tiles,
                           Tensor& out, const FlashDecodeWorkspace* decode, int tile_rows);
}  // namespace inferx::ops::cuda
