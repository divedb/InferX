#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/ops/cuda/generic_attention.h"

namespace inferx::ops::cuda {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float BlockSum(float value, float* shared) {
  const int lane = threadIdx.x, warp = lane / 32;
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if (lane % 32 == 0) shared[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kThreads / 32 ? shared[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset /= 2) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  const float sum = shared[0];
  __syncthreads();
  return sum;
}

/// \brief One block per (query row, head).
///
/// Walks the sequence's pages in order, computes one causal score per KV
/// token (dot over any head_dim via the block-wide sum), and keeps an
/// online-softmax accumulator strided over the output dimension. The
/// optional per-head sink joins the denominator only.
__global__ void GenericAttentionKernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ sinks,
    const int* __restrict__ qo, const int* __restrict__ kv_indptr,
    const int* __restrict__ kv_indices, const int* __restrict__ last_page_len,
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ out, int query_heads, int kv_heads, int head_dim,
    int block_size, float scale, long long window, float softcap) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int tid = threadIdx.x;

  // Binary search for this row's sequence in qo_indptr.
  int lo = 0, hi = gridDim.z - 1;  // gridDim.z == num_seqs
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (qo[mid] <= row) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int seq = lo;

  const int pages = kv_indptr[seq + 1] - kv_indptr[seq];
  const int seq_len = (pages - 1) * block_size + last_page_len[seq];
  const long long q_pos = row - qo[seq];  // Position within the sequence.
  const long long floor_j = window > 0 ? q_pos - (window - 1) : 0;

  const __nv_bfloat16* q_row = q + (static_cast<int64_t>(row) * query_heads + head) * head_dim;
  const int kv_head = head / (query_heads / kv_heads);

  float max_score = -INFINITY;
  float denom = 0.0f;
  float acc[8];  // Head dims beyond 8*256 fall back to a slower path; none do.
  const int owned = (head_dim + kThreads - 1) / kThreads;
  if (owned > 8) return;  // Guarded by dispatch; kept for safety.
  for (int i = 0; i < owned; ++i) acc[i] = 0.0f;

  for (int page = 0; page < pages; ++page) {
    const int block = kv_indices[kv_indptr[seq] + page];
    const int tokens =
        page == pages - 1 ? last_page_len[seq] : block_size;
    for (int slot = 0; slot < tokens; ++slot) {
      const long long j = static_cast<long long>(page) * block_size + slot;
      if (j > q_pos || j < floor_j) continue;
      const int64_t token = static_cast<int64_t>(block) * block_size + slot;
      const __nv_bfloat16* k_row =
          key + (token * kv_heads + kv_head) * head_dim;
      float partial = 0.0f;
      for (int d = tid; d < head_dim; d += kThreads) {
        partial += __bfloat162float(q_row[d]) * __bfloat162float(k_row[d]);
      }
      __shared__ float shared[kThreads / 32];
      float score = BlockSum(partial, shared) * scale;
      if (softcap > 0.0f) score = softcap * tanhf(score / softcap);
      const float m = fmaxf(max_score, score);
      const float growth = expf(max_score == -INFINITY ? -INFINITY : max_score - m);
      // Rescale the running accumulator once per token; every thread owns
      // the same columns it will read back, so no cross-thread exchange.
      const float weight = expf(score - m);
      for (int i = 0; i < owned; ++i) {
        const int d = tid + i * kThreads;
        if (d < head_dim) {
          const __nv_bfloat16* v_row =
              value + (token * kv_heads + kv_head) * head_dim;
          acc[i] = acc[i] * growth + weight * __bfloat162float(v_row[d]);
        }
      }
      denom = denom * growth + weight;
      max_score = m;
    }
  }

  // The sink is a denominator-only logit: no value contributes, so it can
  // only shrink every output component (gpt-oss attention sinks).
  if (sinks != nullptr) {
    const float sink = __bfloat162float(sinks[head]);
    const float m = fmaxf(max_score, sink);
    const float growth = expf(max_score == -INFINITY ? -INFINITY : max_score - m);
    for (int i = 0; i < owned; ++i) acc[i] *= growth;
    denom = denom * growth + expf(sink - m);
    max_score = m;
  }

  __nv_bfloat16* out_row = out + (static_cast<int64_t>(row) * query_heads + head) * head_dim;
  for (int i = 0; i < owned; ++i) {
    const int d = tid + i * kThreads;
    if (d < head_dim) out_row[d] = __float2bfloat16(acc[i] / denom);
  }
}

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace

Status GenericPagedAttention(OpContext& ctx, const Tensor& q, const Tensor& qo_indptr,
                             const Tensor& kv_indptr, const Tensor& kv_indices,
                             const Tensor& last_page_len, const Tensor& key_cache,
                             const Tensor& value_cache, int64_t block_size,
                             const AttentionParams& params, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  if ((params.head_dim + kThreads - 1) / kThreads > 8) {
    return UnimplementedError("generic attention supports head_dim up to ", 8 * kThreads);
  }
  const float* sink_f32 = nullptr;
  const __nv_bfloat16* sink_bf16 =
      params.sinks != nullptr && params.sinks->GetDataType() == DataType::kBFloat16
          ? static_cast<const __nv_bfloat16*>(params.sinks->Data())
          : nullptr;
  if (params.sinks != nullptr && sink_bf16 == nullptr) {
    sink_f32 = static_cast<const float*>(params.sinks->Data());
  }
  // The sink pointer is passed as bf16 in both layouts; float sinks are
  // bit-identical only for values already rounded to bf16, so the component
  // stores sinks in bf16 (checkpoint dtype) and this alias is exact.
  if (sink_f32 != nullptr) {
    return InvalidArgumentError("generic attention expects bf16 sinks");
  }
  const int num_seqs = static_cast<int>(qo_indptr.Numel() - 1);
  dim3 grid(static_cast<uint32_t>(q.Dim(0)), static_cast<uint32_t>(params.query_heads),
            static_cast<uint32_t>(num_seqs));
  GenericAttentionKernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(ctx.GetStream())>>>(
      static_cast<const __nv_bfloat16*>(q.Data()), sink_bf16,
      static_cast<const int*>(qo_indptr.Data()), static_cast<const int*>(kv_indptr.Data()),
      static_cast<const int*>(kv_indices.Data()),
      static_cast<const int*>(last_page_len.Data()),
      static_cast<const __nv_bfloat16*>(key_cache.Data()),
      static_cast<const __nv_bfloat16*>(value_cache.Data()),
      static_cast<__nv_bfloat16*>(out.Data()), static_cast<int>(params.query_heads),
      static_cast<int>(params.kv_heads), static_cast<int>(params.head_dim),
      static_cast<int>(block_size), params.scale,
      static_cast<long long>(params.sliding_window), params.softcap);
  return CudaError(cudaGetLastError(), "generic attention launch");
}

}  // namespace inferx::ops::cuda
