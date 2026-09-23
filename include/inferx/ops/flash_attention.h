#pragma once
#include "inferx/ops/attention.h"
namespace inferx::ops {
struct FlashDecodeWorkspace {
  // Eight partitions per request, up to 16 requests. Shared across layers.
  static constexpr int kPartitions = 8;
  static constexpr int kMaxBatch = 16;
  Tensor plan, values, scores;
};
Status PrepareFlashDecode(ExecutionContext& ctx, const Tensor& kv_indptr,
                          const Tensor& last_page_len, int block_size,
                          FlashDecodeWorkspace& workspace);
// Graph-safe GPU planning, reused across layers. Inputs must describe positive
// query lengths and valid, nonempty KV page tables; each query chunk is the
// suffix of its sequence's current KV. Host/device metadata must agree.
// `tiles` must equal sum(ceil(query_length * group / tile_rows)). The caller
// owns validating device metadata values and keeping buffers alive on stream.
// Supported BF16 full causal geometry: head_dim 64/128/256, GQA ratio 1..32.
// Both tile sizes 64 and 128 are supported. Other geometry returns a Status.
Status PrepareFlashAttention(ExecutionContext& ctx, const Tensor& qo_indptr, Tensor& plan,
                             int group, int tiles, int tile_rows = 64);
Status FlashPagedAttention(ExecutionContext& ctx, const Tensor& q, const Tensor& qo,
                           const Tensor& kv, const Tensor& indices, const Tensor& last_page_len,
                           const Tensor& key, const Tensor& value, int64_t block_size,
                           const AttentionParams& params, const Tensor& plan, int tiles,
                           Tensor& out, const FlashDecodeWorkspace* decode = nullptr,
                           int tile_rows = 64);
}  // namespace inferx::ops
