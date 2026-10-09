#include "inferx/ops/cuda/status.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "inferx/ops/cuda/norm.h"

#include "inferx/ops/cuda/rotary.h"

namespace inferx::ops::cuda {
namespace {

/// Neox-style rotate-half RoPE. Thread `j` owns the complex pair (i, i + r/2)
/// of one head, where i = j % (r/2); every head of q and k is covered by the
/// grid-stride loop. Frequencies come from the shared InverseFrequency
/// formula, which carries every scaling flavor, and the YaRN temperature
/// multiplies cos/sin in fp32 before the activation-dtype rounding.
__global__ void ApplyRopeKernel(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                                const int32_t* __restrict__ positions, int query_heads,
                                int kv_heads, int head_dim, int rotary_dim,
                                ops::RotaryParams params) {
  const int token = blockIdx.x;
  const float pos = static_cast<float>(positions[token]);
  const int half = rotary_dim / 2;
  const int pairs = max(query_heads, kv_heads) * half;
  for (int j = threadIdx.x; j < pairs; j += blockDim.x) {
    const int head = j / half;
    const int i = j - head * half;
    const float angle = pos * ops::InverseFrequency(i, params);
    // Both the reference cache and its elementwise products use BF16.
    // Preserve those rounding boundaries instead of fusing the rotation in FP32.
    const __nv_bfloat16 c = __float2bfloat16(cosf(angle) * params.scaling.attention_scale);
    const __nv_bfloat16 s = __float2bfloat16(sinf(angle) * params.scaling.attention_scale);
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

Status NormalizeAndApplyRope(OpContext& ctx, const Tensor& q, const Tensor& k,
                             const Tensor& q_weight, const Tensor& k_weight,
                             const Tensor& positions, float eps, const RotaryParams& params) {
  Tensor q_out = q;
  Tensor k_out = k;
  const NormConfig config{NormKind::kRms, eps};
  INFERX_RETURN_IF_ERROR(::inferx::ops::cuda::RmsNorm(ctx, q, q_weight, q_out, config));
  INFERX_RETURN_IF_ERROR(::inferx::ops::cuda::RmsNorm(ctx, k, k_weight, k_out, config));
  return ::inferx::ops::cuda::ApplyRope(ctx, q, k, positions, params);
}

Status ApplyRope(OpContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  // The kernels take non-const pointers although the update is element-local.
  auto* q_ptr = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(q.Data()));
  auto* k_ptr = const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(k.Data()));
  ApplyRopeKernel<<<static_cast<uint32_t>(q.Dim(0)), 256, 0,
                    static_cast<cudaStream_t>(ctx.GetStream())>>>(
      q_ptr, k_ptr, static_cast<const int32_t*>(positions.Data()),
      static_cast<int>(q.Dim(1)), static_cast<int>(k.Dim(1)), static_cast<int>(q.Dim(2)),
      static_cast<int>(params.rotary_dim), params);
  return CudaStatus(cudaGetLastError(), "apply rope launch");
}

}  // namespace inferx::ops::cuda
