#include <cmath>
#include "inferx/ops/attention.h"

#include "ops/cuda/attention.h"

namespace inferx::ops {
StatusOr<AttentionBackend> ParseAttentionBackend(std::string_view name) {
  if (name == "flashinfer" || name == "default" || name == "flash") {
    return AttentionBackend::kFlashInfer;
  }
  return InvalidArgumentError("unsupported attention backend: ", name,
                              "; available: flashinfer (aliases: default, flash)");
}

Status ValidateAttentionGeometry(AttentionBackend backend, const AttentionParams& p) {
  if (backend != AttentionBackend::kFlashInfer) {
    return UnimplementedError("attention backend has no implementation");
  }
  if (p.query_heads <= 0 || p.kv_heads <= 0 || p.head_dim <= 0 ||
      p.query_heads % p.kv_heads != 0 || !std::isfinite(p.scale) || p.sliding_window < 0) {
    return InvalidArgumentError("invalid attention heads, head dimension, scale or window");
  }
  if (p.head_dim != 64 && p.head_dim != 128 && p.head_dim != 256) {
    return UnimplementedError("FlashInfer integration supports head dimensions 64, 128, 256; got ",
                              p.head_dim);
  }
  if (p.query_heads / p.kv_heads > 32 || p.sliding_window != 0) {
    return UnimplementedError("FlashInfer integration requires GQA ratio <= 32 and full causal attention");
  }
  return OkStatus();
}

namespace {

/// Validates the flat ragged batch geometry shared by both attention ops.
Status CheckRaggedBatch(const Tensor& positions, const Tensor& batch_indices,
                        const Tensor& kv_indptr, const Tensor& kv_indices, int64_t num_tokens) {
  if (positions.Rank() != 1 || positions.GetDataType() != DataType::kInt32 ||
      positions.Dim(0) != num_tokens) {
    return InvalidArgumentError("attention positions must be rank-1 int32 with one entry per token");
  }
  if (batch_indices.Rank() != 1 || batch_indices.GetDataType() != DataType::kInt32 ||
      batch_indices.Dim(0) != num_tokens) {
    return InvalidArgumentError("attention batch_indices must be rank-1 int32 with one entry per token");
  }
  if (kv_indptr.Rank() != 1 || kv_indptr.GetDataType() != DataType::kInt32 ||
      kv_indptr.Dim(0) < 1) {
    return InvalidArgumentError("attention kv_indptr must be rank-1 int32 with num_seqs + 1 entries");
  }
  if (kv_indices.Rank() != 1 || kv_indices.GetDataType() != DataType::kInt32) {
    return InvalidArgumentError("attention kv_indices must be rank-1 int32");
  }
  return OkStatus();
}

Status CheckCache(const Tensor& key_cache, const Tensor& value_cache, int64_t block_size,
                  int64_t kv_heads, int64_t head_dim, const DeviceId& device) {
  for (const Tensor* cache : {&key_cache, &value_cache}) {
    if (cache->Rank() != 4 || cache->GetDataType() != DataType::kBFloat16) {
      return InvalidArgumentError("KV caches must be rank-4 bfloat16");
    }
    if (cache->Dim(1) != block_size || cache->Dim(2) != kv_heads || cache->Dim(3) != head_dim) {
      return InvalidArgumentError("KV cache geometry [", cache->Dim(0), ", ", cache->Dim(1),
                                  ", ", cache->Dim(2), ", ", cache->Dim(3),
                                  "] disagrees with block_size/kv_heads/head_dim");
    }
    if (cache->Device() != device) {
      return InvalidArgumentError("KV caches must live on the context's device ", device.ToString());
    }
  }
  return OkStatus();
}

}  // namespace

Status WritePagedKv(ExecutionContext& ctx, const Tensor& k, const Tensor& v,
                    const Tensor& positions, const Tensor& batch_indices,
                    const Tensor& kv_indptr, const Tensor& kv_indices,
                    const Tensor& key_cache, const Tensor& value_cache, int64_t block_size) {
  if (!k.IsDefined() || !v.IsDefined()) {
    return InvalidArgumentError("WritePagedKv requires defined k and v");
  }
  if (k.Rank() != 2 || v.Rank() != 2 || k.GetDataType() != DataType::kBFloat16 ||
      v.GetDataType() != DataType::kBFloat16) {
    return InvalidArgumentError("WritePagedKv expects rank-2 bfloat16 k and v");
  }
  if (k.Dim(0) != v.Dim(0) || k.Dim(1) != v.Dim(1) || k.IsEmpty()) {
    return InvalidArgumentError("WritePagedKv k and v must agree in shape and be non-empty");
  }
  if (block_size <= 0) {
    return InvalidArgumentError("WritePagedKv block_size must be positive");
  }
  if (key_cache.Rank() != 4 || value_cache.Rank() != 4 ||
      key_cache.Dim(0) <= 0 || key_cache.Dim(2) <= 0 || key_cache.Dim(3) <= 0 ||
      key_cache.Dim(0) != value_cache.Dim(0)) {
    return InvalidArgumentError("WritePagedKv requires matching nonempty rank-4 caches");
  }
  const int64_t kv_heads = key_cache.Dim(2);
  const int64_t head_dim = key_cache.Dim(3);
  if (k.Dim(1) != kv_heads * head_dim) {
    return InvalidArgumentError("WritePagedKv k width ", k.Dim(1), " disagrees with cache geometry ",
                                kv_heads * head_dim);
  }
  const DeviceId device = ctx.device();
  INFERX_RETURN_IF_ERROR(
      CheckRaggedBatch(positions, batch_indices, kv_indptr, kv_indices, k.Dim(0)));
  INFERX_RETURN_IF_ERROR(CheckCache(key_cache, value_cache, block_size, kv_heads, head_dim, device));
  for (const Tensor* metadata : {&positions, &batch_indices, &kv_indptr, &kv_indices}) {
    if (metadata->Device() != device)
      return InvalidArgumentError("WritePagedKv metadata must live on the context device");
  }
  if (k.Device() != device || v.Device() != device) {
    return InvalidArgumentError("WritePagedKv tensors must live on the context's device ",
                                device.ToString());
  }
  switch (device.kind) {
    case DeviceKind::kCuda:
      return cuda::WritePagedKv(ctx, k, v, positions, batch_indices, kv_indptr, kv_indices,
                                key_cache, value_cache, block_size);
    default:
      return UnimplementedError("WritePagedKv has no implementation for device ",
                                device.ToString());
  }
}

}  // namespace inferx::ops
