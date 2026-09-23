#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

#include "ops/cuda/flash_attention.h"
namespace inferx::ops::cuda {
namespace {
__global__ void DecodePlan(const int* kv, const int* last, int batch, int page, int* plan) {
  constexpr int parts = FlashDecodeWorkspace::kPartitions;
  const int tiles = batch * parts;
  int* requests = plan;
  int* chunks = plan + tiles;
  bool* valid = reinterpret_cast<bool*>(plan + 2 * tiles);
  int* offsets = plan + 3 * tiles;
  int max_pages = 0;
  for (int s = 0; s < batch; ++s) max_pages = max(max_pages, kv[s + 1] - kv[s]);
  const int chunk_size = max(1, (max_pages + parts - 1) / parts) * page;
  offsets[batch + 1] = chunk_size;
  int count = 0;
  for (int s = 0; s < batch; ++s) {
    offsets[s] = count;
    const int length = (kv[s + 1] - kv[s] - 1) * page + last[s];
    for (int c = 0; c < (length + chunk_size - 1) / chunk_size; ++c) {
      requests[count] = s;
      chunks[count] = c;
      valid[count++] = true;
    }
  }
  offsets[batch] = count;
  for (; count < tiles; ++count) {
    requests[count] = chunks[count] = 0;
    valid[count] = false;
  }
}
__global__ void Plan(const int* qo, int batch, int group, int tiles, int tile_rows, int* plan) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) plan[3 * tiles] = 0x3fffffff;
  if (i >= tiles) return;
  int offset = 0;
  for (int s = 0; s < batch; ++s) {
    const int n = ((qo[s + 1] - qo[s]) * group + tile_rows - 1) / tile_rows;
    if (i < offset + n) {
      plan[i] = s;
      plan[tiles + i] = i - offset;
      plan[2 * tiles + i] = 0;
      return;
    }
    offset += n;
  }
}
Status Error(cudaError_t e) {
  return e == cudaSuccess ? OkStatus() : InternalError("FlashInfer: ", cudaGetErrorString(e));
}
template <class T>
T* Ptr(const Tensor& t) {
  return const_cast<T*>(static_cast<const T*>(t.Data()));
}
}  // namespace
Status PrepareFlashDecode(ExecutionContext& ctx, const Tensor& kv, const Tensor& last,
                          int block_size, FlashDecodeWorkspace& workspace) {
  const int batch = last.Numel();
  const int tiles = batch * FlashDecodeWorkspace::kPartitions;
  if (batch <= 0 || batch > FlashDecodeWorkspace::kMaxBatch ||
      workspace.plan.Numel() < 3 * tiles + batch + 2)
    return InvalidArgumentError("invalid split decode plan capacity");
  DecodePlan<<<1, 1, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
      Ptr<int>(kv), Ptr<int>(last), batch, block_size, Ptr<int>(workspace.plan));
  return Error(cudaGetLastError());
}
Status PrepareFlashAttention(ExecutionContext& ctx, const Tensor& qo, Tensor& plan, int group,
                             int tiles, int tile_rows) {
  if (tiles <= 0 || plan.Numel() < 3 * tiles + 1 || group <= 0 || group > 32 ||
      (tile_rows != 64 && tile_rows != 128))
    return InvalidArgumentError("invalid FlashInfer plan capacity");
  Plan<<<(tiles + 127) / 128, 128, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
      Ptr<int>(qo), qo.Numel() - 1, group, tiles, tile_rows, Ptr<int>(plan));
  return Error(cudaGetLastError());
}
template <int HeadDim>
Status FlashPagedAttentionImpl(ExecutionContext& ctx, const Tensor& q, const Tensor& qo,
                           const Tensor& kv, const Tensor& indices, const Tensor& last,
                           const Tensor& key, const Tensor& value, int64_t block_size,
                           const AttentionParams& p, const Tensor& plan, int tiles,
                           Tensor& out, const FlashDecodeWorkspace* decode, int tile_rows) {
  using namespace flashinfer;
  using T = __nv_bfloat16;
  using Variant = DefaultAttention<false, false, false, false>;
  const int group = p.query_heads / p.kv_heads;
  const int batch = qo.Numel() - 1;
  paged_kv_t<T, int> cache(p.kv_heads, block_size, HeadDim, batch, QKVLayout::kNHD, Ptr<T>(key),
                           Ptr<T>(value), Ptr<int>(indices), Ptr<int>(kv), Ptr<int>(last));
  auto stream = static_cast<cudaStream_t>(ctx.stream());
  // Ratios unsupported by the specialized decode dispatcher use FlashInfer
  // prefill with one query per sequence, which implements the same attention.
  const bool decode_group = group == 1 || group == 2 || group == 3 || group == 4 || group == 6 || group == 8;
  if (q.Dim(0) == batch && decode_group) {
    BatchDecodeParams<T, T, T, int> params;
    params.q = Ptr<T>(q);
    params.o = Ptr<T>(out);
    params.paged_kv = cache;
    params.num_qo_heads = p.query_heads;
    params.padded_batch_size = batch;
    params.q_stride_n = p.query_heads * HeadDim;
    params.q_stride_h = HeadDim;
    params.window_left = -1;
    params.sm_scale = p.scale;
    params.request_indices = Ptr<int>(plan);
    params.kv_tile_indices = Ptr<int>(plan) + 2 * tiles;
    params.o_indptr = Ptr<int>(qo);
    params.kv_chunk_size_ptr = Ptr<int>(plan) + 3 * tiles;
    if (decode != nullptr) {
      const int splits = batch * FlashDecodeWorkspace::kPartitions;
      if (decode->values.Numel() < splits * p.query_heads * HeadDim ||
          decode->scores.Numel() < splits * p.query_heads)
        return InvalidArgumentError("invalid split decode workspace capacity");
      params.padded_batch_size = splits;
      params.request_indices = Ptr<int>(decode->plan);
      params.kv_tile_indices = Ptr<int>(decode->plan) + splits;
      params.block_valid_mask = reinterpret_cast<bool*>(Ptr<int>(decode->plan) + 2 * splits);
      params.o_indptr = Ptr<int>(decode->plan) + 3 * splits;
      params.kv_chunk_size_ptr = Ptr<int>(decode->plan) + 3 * splits + batch + 1;
      return Error(BatchDecodeWithPagedKVCacheDispatched<HeadDim, PosEncodingMode::kNone, Variant>(
          params, Ptr<T>(decode->values), Ptr<float>(decode->scores), false, stream));
    }
    return Error(BatchDecodeWithPagedKVCacheDispatched<HeadDim, PosEncodingMode::kNone, Variant>(
        params, nullptr, nullptr, false, stream));
  }
  BatchPrefillPagedParams<T, T, T, int> params;
  params.q = Ptr<T>(q);
  params.o = Ptr<T>(out);
  params.paged_kv = cache;
  params.q_indptr = Ptr<int>(qo);
  params.group_size = uint_fastdiv(group);
  params.num_qo_heads = p.query_heads;
  params.q_stride_n = p.query_heads * HeadDim;
  params.q_stride_h = HeadDim;
  params.window_left = -1;
  params.sm_scale = p.scale;
  params.request_indices = Ptr<int>(plan);
  params.qo_tile_indices = Ptr<int>(plan) + tiles;
  params.kv_tile_indices = Ptr<int>(plan) + 2 * tiles;
  params.o_indptr = Ptr<int>(qo);
  params.kv_chunk_size_ptr = Ptr<int>(plan) + 3 * tiles;
  params.padded_batch_size = tiles;
  if (tile_rows == 128) {
    return Error(
        BatchPrefillWithPagedKVCacheDispatched<true, 128, HeadDim, HeadDim, PosEncodingMode::kNone, false,
                                               MaskMode::kCausal, Variant>(
            params, nullptr, nullptr, false, stream));
  }
  return Error(
      BatchPrefillWithPagedKVCacheDispatched<true, 64, HeadDim, HeadDim, PosEncodingMode::kNone, false,
                                             MaskMode::kCausal, Variant>(
          params, nullptr, nullptr, false, stream));
}
Status FlashPagedAttention(ExecutionContext& ctx, const Tensor& q, const Tensor& qo,
                           const Tensor& kv, const Tensor& indices, const Tensor& last,
                           const Tensor& key, const Tensor& value, int64_t block_size,
                           const AttentionParams& p, const Tensor& plan, int tiles,
                           Tensor& out, const FlashDecodeWorkspace* decode, int tile_rows) {
  switch (p.head_dim) {
    case 64: return FlashPagedAttentionImpl<64>(ctx, q, qo, kv, indices, last, key, value,
                                                block_size, p, plan, tiles, out, decode, tile_rows);
    case 128: return FlashPagedAttentionImpl<128>(ctx, q, qo, kv, indices, last, key, value,
                                                  block_size, p, plan, tiles, out, decode, tile_rows);
    case 256: return FlashPagedAttentionImpl<256>(ctx, q, qo, kv, indices, last, key, value,
                                                  block_size, p, plan, tiles, out, decode, tile_rows);
    default: return UnimplementedError("unsupported FlashInfer head dimension");
  }
}
}  // namespace inferx::ops::cuda
