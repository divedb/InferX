#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <string_view>

#include "ops/cuda/attention.h"

namespace inferx::ops::cuda {
namespace {

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

// Eight BF16 values per thread. Packing multiple heads/tokens into each block
// avoids launching one mostly memory-copy block for every cache row at prefill.
__global__ void WritePagedKvVectorKernel(const uint4* k, const uint4* v,
                                        const int32_t* positions, const int32_t* batches,
                                        const int32_t* indptr, const int32_t* indices,
                                        uint4* key_cache, uint4* value_cache,
                                        int vectors_per_token, int block_size, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int token = i / vectors_per_token;
  const int column = i % vectors_per_token;
  const int pos = positions[token];
  const int block = indices[indptr[batches[token]] + pos / block_size];
  const int64_t dst = (int64_t(block) * block_size + pos % block_size) * vectors_per_token + column;
  key_cache[dst] = k[i];
  value_cache[dst] = v[i];
}

/// One block per (token, kv head) writes that token's key and value rows
/// into their cache slot, derived from the position and block table.
__global__ void WritePagedKvKernel(const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v,
                                   const int32_t* __restrict__ positions,
                                   const int32_t* __restrict__ batch_indices,
                                   const int32_t* __restrict__ kv_indptr,
                                   const int32_t* __restrict__ kv_indices,
                                   __nv_bfloat16* __restrict__ key_cache,
                                   __nv_bfloat16* __restrict__ value_cache, int kv_heads,
                                   int head_dim, int64_t block_size) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int seq = batch_indices[token];
  const int32_t pos = positions[token];
  const int block = kv_indices[kv_indptr[seq] + pos / block_size];
  const int slot = pos % block_size;
  const int64_t cache_row =
      ((static_cast<int64_t>(block) * block_size) + slot) * kv_heads + head;
  const __nv_bfloat16* src_k = k + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
  const __nv_bfloat16* src_v = v + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
  __nv_bfloat16* dst_k = key_cache + cache_row * head_dim;
  __nv_bfloat16* dst_v = value_cache + cache_row * head_dim;
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    dst_k[i] = src_k[i];
    dst_v[i] = src_v[i];
  }
}

}  // namespace

Status WritePagedKv(ExecutionContext& ctx, const Tensor& k, const Tensor& v,
                    const Tensor& positions, const Tensor& batch_indices,
                    const Tensor& kv_indptr, const Tensor& kv_indices,
                    const Tensor& key_cache, const Tensor& value_cache, int64_t block_size) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  static const bool scalar_write = [] {
    const char* value = std::getenv("INFERX_DIAGNOSTIC_SCALAR_KV");
    return value != nullptr && std::string_view(value) == "1";
  }();
  if (!scalar_write && k.Dim(0) >= 32 && k.Dim(1) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(k.Data()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(v.Data()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(key_cache.Data()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(value_cache.Data()) % 16 == 0) {
    const int64_t count = k.Numel() / 8;
    WritePagedKvVectorKernel<<<(count + 127) / 128, 128, 0, ctx.stream()>>>(
        static_cast<const uint4*>(k.Data()), static_cast<const uint4*>(v.Data()),
        positions.DataAs<int32_t>(), batch_indices.DataAs<int32_t>(),
        kv_indptr.DataAs<int32_t>(), kv_indices.DataAs<int32_t>(),
        const_cast<uint4*>(static_cast<const uint4*>(key_cache.Data())),
        const_cast<uint4*>(static_cast<const uint4*>(value_cache.Data())),
        k.Dim(1) / 8, block_size, count);
    return CudaError(cudaGetLastError(), "vector write paged kv launch");
  }
  const dim3 grid(static_cast<uint32_t>(k.Dim(0)), static_cast<uint32_t>(key_cache.Dim(2)));
  const uint32_t threads = 128;
  WritePagedKvKernel<<<grid, threads, 0, static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(k.Data()),
      static_cast<const __nv_bfloat16*>(v.Data()),
      static_cast<const int32_t*>(positions.Data()),
      static_cast<const int32_t*>(batch_indices.Data()),
      static_cast<const int32_t*>(kv_indptr.Data()),
      static_cast<const int32_t*>(kv_indices.Data()),
      const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(key_cache.Data())),
      const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(value_cache.Data())),
      static_cast<int>(key_cache.Dim(2)), static_cast<int>(key_cache.Dim(3)), block_size);
  return CudaError(cudaGetLastError(), "write paged kv launch");
}

}  // namespace inferx::ops::cuda
