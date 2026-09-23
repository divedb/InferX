#include "inferx/ops/flash_attention.h"

#include <limits>

#include "inferx/ops/profile.h"
#include "ops/cuda/flash_attention.h"

namespace inferx::ops {
namespace {
Status CheckTensor(const Tensor& t, DataType dtype, int rank, const DeviceId& device) {
  if (!t.IsDefined() || t.Rank() != rank || t.GetDataType() != dtype || t.Device() != device) {
    return InvalidArgumentError("FlashInfer tensor rank, dtype or device mismatch");
  }
  return OkStatus();
}
Status CheckDevice(ExecutionContext& ctx) {
  if (!ctx.device().IsCuda()) return UnimplementedError("FlashInfer requires CUDA");
  return ctx.runtime().Activate();
}
Status CheckPlan(ExecutionContext& ctx, const Tensor& plan, int tiles, int tile_rows) {
  INFERX_RETURN_IF_ERROR(CheckTensor(plan, DataType::kInt32, 1, ctx.device()));
  if (tiles <= 0 || plan.Numel() < 3LL * tiles + 1 || (tile_rows != 64 && tile_rows != 128)) {
    return InvalidArgumentError("invalid FlashInfer plan capacity or tile size");
  }
  return OkStatus();
}
}  // namespace

Status PrepareFlashDecode(ExecutionContext& ctx, const Tensor& kv, const Tensor& last,
                          int block_size, FlashDecodeWorkspace& workspace) {
  INFERX_RETURN_IF_ERROR(CheckDevice(ctx));
  for (const Tensor* t : std::initializer_list<const Tensor*>{&kv, &last, &workspace.plan}) {
    INFERX_RETURN_IF_ERROR(CheckTensor(*t, DataType::kInt32, 1, ctx.device()));
  }
  const int64_t batch = last.Numel();
  if (batch <= 0 || batch > FlashDecodeWorkspace::kMaxBatch || kv.Numel() != batch + 1 ||
      block_size <= 0 ||
      workspace.plan.Numel() < 3 * batch * FlashDecodeWorkspace::kPartitions + batch + 2) {
    return InvalidArgumentError("invalid split decode batch, page size or workspace");
  }
  return cuda::PrepareFlashDecode(ctx, kv, last, block_size, workspace);
}

Status PrepareFlashAttention(ExecutionContext& ctx, const Tensor& qo, Tensor& plan, int group,
                             int tiles, int tile_rows) {
  INFERX_RETURN_IF_ERROR(CheckDevice(ctx));
  INFERX_RETURN_IF_ERROR(CheckTensor(qo, DataType::kInt32, 1, ctx.device()));
  INFERX_RETURN_IF_ERROR(CheckPlan(ctx, plan, tiles, tile_rows));
  if (qo.Numel() < 2 || qo.Numel() > std::numeric_limits<int>::max() || group <= 0 ||
      group > 32) {
    return InvalidArgumentError("invalid FlashInfer query offsets or group size");
  }
  return cuda::PrepareFlashAttention(ctx, qo, plan, group, tiles, tile_rows);
}

Status FlashPagedAttention(ExecutionContext& ctx, const Tensor& q, const Tensor& qo,
                           const Tensor& kv, const Tensor& indices, const Tensor& last,
                           const Tensor& key, const Tensor& value, int64_t block_size,
                           const AttentionParams& p, const Tensor& plan, int tiles, Tensor& out,
                           const FlashDecodeWorkspace* decode, int tile_rows) {
  INFERX_RETURN_IF_ERROR(ValidateAttentionGeometry(AttentionBackend::kFlashInfer, p));
  INFERX_RETURN_IF_ERROR(CheckDevice(ctx));
  INFERX_RETURN_IF_ERROR(CheckPlan(ctx, plan, tiles, tile_rows));
  for (const Tensor* t : {&qo, &kv, &indices, &last}) {
    INFERX_RETURN_IF_ERROR(CheckTensor(*t, DataType::kInt32, 1, ctx.device()));
  }
  for (const Tensor* t : std::initializer_list<const Tensor*>{&q, &out}) {
    INFERX_RETURN_IF_ERROR(CheckTensor(*t, DataType::kBFloat16, 2, ctx.device()));
  }
  for (const Tensor* t : {&key, &value}) {
    INFERX_RETURN_IF_ERROR(CheckTensor(*t, DataType::kBFloat16, 4, ctx.device()));
    if (t->Dim(0) <= 0 || t->Dim(1) != block_size || t->Dim(2) != p.kv_heads ||
        t->Dim(3) != p.head_dim) {
      return InvalidArgumentError("FlashInfer KV cache geometry mismatch");
    }
  }
  const int64_t batch = last.Numel();
  if (block_size <= 0 || block_size > std::numeric_limits<int>::max() || batch <= 0 ||
      batch > std::numeric_limits<int>::max() || qo.Numel() != batch + 1 ||
      kv.Numel() != batch + 1 || indices.Numel() < batch || q.Dim(0) < batch ||
      q.Dim(0) > std::numeric_limits<int>::max() / 32 ||
      p.query_heads > std::numeric_limits<int>::max() / p.head_dim ||
      q.Dim(1) != p.query_heads * p.head_dim || q.Dim(0) != out.Dim(0) ||
      q.Dim(1) != out.Dim(1) || key.Dim(0) != value.Dim(0)) {
    return InvalidArgumentError("FlashInfer query, cache or ragged batch geometry mismatch");
  }
  if (decode != nullptr) {
    INFERX_RETURN_IF_ERROR(CheckTensor(decode->plan, DataType::kInt32, 1, ctx.device()));
    INFERX_RETURN_IF_ERROR(CheckTensor(decode->values, DataType::kBFloat16, 1, ctx.device()));
    INFERX_RETURN_IF_ERROR(CheckTensor(decode->scores, DataType::kFloat, 1, ctx.device()));
    const int64_t splits = batch * FlashDecodeWorkspace::kPartitions;
    if (batch > FlashDecodeWorkspace::kMaxBatch || q.Dim(0) != batch ||
        decode->plan.Numel() < 3 * splits + batch + 2 ||
        decode->values.Numel() < splits * q.Dim(1) ||
        decode->scores.Numel() < splits * p.query_heads) {
      return InvalidArgumentError("invalid FlashInfer split decode workspace");
    }
  }
  return ProfileCall(ctx, "attention", [&] {
    return cuda::FlashPagedAttention(ctx, q, qo, kv, indices, last, key, value, block_size, p,
                                     plan, tiles, out, decode, tile_rows);
  });
}
}  // namespace inferx::ops
