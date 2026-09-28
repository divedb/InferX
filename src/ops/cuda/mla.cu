#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/ops/cuda/mla.h"

namespace inferx::ops::cuda {
namespace {

constexpr int kThreads = 256;

__global__ void AssembleMlaKernel(const __nv_bfloat16* __restrict__ k_rope,
                                  const __nv_bfloat16* __restrict__ up, int rows, int heads,
                                  int nope, int rope, int v_dim, int head_dim,
                                  __nv_bfloat16* __restrict__ k_out,
                                  __nv_bfloat16* __restrict__ v_out) {
  const int64_t per_head = int64_t(heads) * head_dim;
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(rows) * per_head) return;
  const int64_t row = i / per_head;
  const int head = (i % per_head) / head_dim;
  const int col = i % head_dim;
  const __nv_bfloat16* rope_row = k_rope + row * rope;
  const __nv_bfloat16* up_row = up + row * heads * (nope + v_dim) + head * (nope + v_dim);
  if (col < rope) {
    k_out[i] = rope_row[col];
  } else {
    k_out[i] = up_row[col - rope];
  }
  v_out[i] = col < v_dim ? up_row[nope + col] : __float2bfloat16(0.0f);
}

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace

Status AssembleMlaCaches(ExecutionContext& ctx, const Tensor& k_rope, const Tensor& up_projected,
                         int64_t heads, int64_t nope, int64_t rope, int64_t v_dim, Tensor& k_out,
                         Tensor& v_out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t head_dim = nope + rope;
  const int64_t total = k_rope.Dim(0) * heads * head_dim;
  AssembleMlaKernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                      static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(k_rope.Data()),
      static_cast<const __nv_bfloat16*>(up_projected.Data()),
      static_cast<int>(k_rope.Dim(0)), static_cast<int>(heads),
      static_cast<int>(nope), static_cast<int>(rope), static_cast<int>(v_dim),
      static_cast<int>(head_dim), static_cast<__nv_bfloat16*>(k_out.Data()),
      static_cast<__nv_bfloat16*>(v_out.Data()));
  return CudaError(cudaGetLastError(), "assemble mla caches");
}

}  // namespace inferx::ops::cuda
