#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "flashinfer/norm.cuh"

#include "ops/cuda/rotary.h"

namespace inferx::ops::cuda {
namespace {

__global__ void NormRopeKernel(__nv_bfloat16* q, __nv_bfloat16* k,
                               const __nv_bfloat16* qw, const __nv_bfloat16* kw,
                               const int32_t* positions, int qheads, int kheads,
                               int rotary, float theta, float eps) {
  __shared__ __nv_bfloat16 normalized[128];
  __shared__ float inv_rms;
  const int token = blockIdx.x, head = blockIdx.y;
  const bool query = head < qheads;
  auto* row = query ? q + (int64_t(token) * qheads + head) * 128
                    : k + (int64_t(token) * kheads + head - qheads) * 128;
  const auto* weight = query ? qw : kw;
  const int i = threadIdx.x;
  float value = i < 128 ? __bfloat162float(row[i]) : 0.0f;
  const float sum = tensorrt_llm::common::blockReduceSum(value * value);
  if (i == 0) inv_rms = rsqrtf(sum / 128.0f + eps);
  __syncthreads();
  if (i < 128) {
    const auto norm = __float2bfloat16(value * inv_rms);
    normalized[i] = __float2bfloat16(__bfloat162float(norm) * __bfloat162float(weight[i]));
  }
  __syncthreads();
  const int half = rotary / 2;
  if (i < half) {
    const float inv_freq = 1.0f / powf(theta, (2.0f / float(rotary)) * float(i));
    const float angle = float(positions[token]) * inv_freq;
    const auto c = __float2bfloat16(cosf(angle)), s = __float2bfloat16(sinf(angle));
    const auto x1 = normalized[i], x2 = normalized[i + half];
    row[i] = __hsub_rn(__hmul_rn(x1, c), __hmul_rn(x2, s));
    row[i + half] = __hadd_rn(__hmul_rn(x2, c), __hmul_rn(x1, s));
  } else if (i >= rotary && i < 128) {
    row[i] = normalized[i];
  }
}

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

/// Neox-style rotate-half RoPE. Thread `j` owns the complex pair (i, i + r/2)
/// of one head, where i = j % (r/2); every head of q and k is covered by the
/// grid-stride loop. Frequencies are derived on device: theta^(-2i/r).
__global__ void ApplyRopeKernel(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                                const int32_t* __restrict__ positions, int query_heads,
                                int kv_heads, int head_dim, int rotary_dim, float theta) {
  const int token = blockIdx.x;
  const float pos = static_cast<float>(positions[token]);
  const int half = rotary_dim / 2;
  const float exponent_scale = 2.0f / static_cast<float>(rotary_dim);
  const int pairs = max(query_heads, kv_heads) * half;
  for (int j = threadIdx.x; j < pairs; j += blockDim.x) {
    const int head = j / half;
    const int i = j - head * half;
    const float inv_freq = 1.0f / powf(theta, exponent_scale * static_cast<float>(i));
    const float angle = pos * inv_freq;
    // Both the reference cache and its elementwise products use BF16.
    // Preserve those rounding boundaries instead of fusing the rotation in FP32.
    const __nv_bfloat16 c = __float2bfloat16(cosf(angle));
    const __nv_bfloat16 s = __float2bfloat16(sinf(angle));
    if (head < query_heads) {
      __nv_bfloat16* qp = q + (static_cast<int64_t>(token) * query_heads + head) * head_dim;
      const __nv_bfloat16 x1 = qp[i], x2 = qp[i + half];
      qp[i] = __hsub_rn(__hmul_rn(x1, c), __hmul_rn(x2, s));
      qp[i + half] = __hadd_rn(__hmul_rn(x2, c), __hmul_rn(x1, s));
    }
    if (head < kv_heads) {
      __nv_bfloat16* kp = k + (static_cast<int64_t>(token) * kv_heads + head) * head_dim;
      const __nv_bfloat16 x1 = kp[i], x2 = kp[i + half];
      kp[i] = __hsub_rn(__hmul_rn(x1, c), __hmul_rn(x2, s));
      kp[i + half] = __hadd_rn(__hmul_rn(x2, c), __hmul_rn(x1, s));
    }
  }
}

}  // namespace

Status NormalizeAndApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                             const Tensor& q_weight, const Tensor& k_weight,
                             const Tensor& positions, float eps, const RotaryParams& params) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  NormRopeKernel<<<dim3(q.Dim(0), q.Dim(1) + k.Dim(1)), 128, 0, ctx.stream()>>>(
      const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q.Data())),
      const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k.Data())),
      static_cast<const __nv_bfloat16*>(q_weight.Data()),
      static_cast<const __nv_bfloat16*>(k_weight.Data()), positions.DataAs<int32_t>(),
      q.Dim(1), k.Dim(1), params.rotary_dim, params.theta, eps);
  return CudaError(cudaGetLastError(), "normalize and apply rope");
}

Status ApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  // The kernels take non-const pointers although the update is element-local.
  auto* q_ptr = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q.Data()));
  auto* k_ptr = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k.Data()));
  ApplyRopeKernel<<<static_cast<uint32_t>(q.Dim(0)), 256, 0,
                    static_cast<cudaStream_t>(ctx.stream())>>>(
      q_ptr, k_ptr, static_cast<const int32_t*>(positions.Data()),
      static_cast<int>(q.Dim(1)), static_cast<int>(k.Dim(1)), static_cast<int>(q.Dim(2)),
      static_cast<int>(params.rotary_dim), params.theta);
  return CudaError(cudaGetLastError(), "apply rope launch");
}

}  // namespace inferx::ops::cuda
