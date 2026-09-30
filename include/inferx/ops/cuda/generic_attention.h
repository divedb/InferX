#ifndef INFERX_OPS_CUDA_GENERIC_ATTENTION_H_
#define INFERX_OPS_CUDA_GENERIC_ATTENTION_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops::cuda {

/// \brief Correctness-first paged attention for what FlashInfer cannot run.
///
/// One thread block per (query row, head): walks the sequence's KV blocks
/// with an online-softmax accumulator in fp32. Covers per-head attention
/// sinks (denominator-only logits), any head dimension, any GQA ratio, and
/// per-layer sliding windows. Slower than the FlashInfer fast path; the
/// dispatcher only selects it when FlashAttentionSupports() is false.
Status GenericPagedAttention(OpContext& ctx, const Tensor& q, const Tensor& qo_indptr,
                             const Tensor& kv_indptr, const Tensor& kv_indices,
                             const Tensor& last_page_len, const Tensor& key_cache,
                             const Tensor& value_cache, int64_t block_size,
                             const AttentionParams& params, Tensor& out);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_GENERIC_ATTENTION_H_
