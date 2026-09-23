#pragma once

#include <cstdint>
#include <string_view>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Appends this step's keys and values into a paged KV cache.
///
/// Each token's cache slot comes from its position and sequence's block
/// table: block = kv_indices[kv_indptr[seq] + pos / block_size], slot =
/// pos % block_size. `k` and `v` are [tokens, kv_heads * head_dim]; the
/// caches are [num_blocks, block_size, kv_heads, head_dim] views for one
/// layer (KvBlockPool::KeyCache/ValueCache shapes).
Status WritePagedKv(ExecutionContext& ctx, const Tensor& k, const Tensor& v,
                    const Tensor& positions, const Tensor& batch_indices,
                    const Tensor& kv_indptr, const Tensor& kv_indices,
                    const Tensor& key_cache, const Tensor& value_cache,
                    int64_t block_size);

/// \brief Parameters of one causal paged-attention call.
struct AttentionParams {
  int64_t query_heads = 0;
  int64_t kv_heads = 0;   ///< Divides query_heads (grouped-query attention).
  int64_t head_dim = 0;   ///< FlashInfer: 64, 128 or 256.
  float scale = 1.0f;     ///< Query-key product scale, typically 1/sqrt(head_dim).
  int64_t sliding_window = 0;  ///< 0 disables windowing; >0 is unimplemented.
};

/// Backend selection is explicit; the removed scalar CUDA path is not a fallback.
enum class AttentionBackend { kFlashInfer };
StatusOr<AttentionBackend> ParseAttentionBackend(std::string_view name);
Status ValidateAttentionGeometry(AttentionBackend backend, const AttentionParams& params);

}  // namespace inferx::ops
