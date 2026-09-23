#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "ops/cuda/elementwise.h"

namespace inferx::ops::cuda {
namespace {

__global__ void SplitQkvKernel(const __nv_bfloat16* packed, __nv_bfloat16* q,
                               __nv_bfloat16* k, __nv_bfloat16* v,
                               int qw, int kw, int vw, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int width = qw + kw + vw;
  const int row = i / width, col = i % width;
  if (col < qw) q[int64_t(row) * qw + col] = packed[i];
  else if (col < qw + kw) k[int64_t(row) * kw + col - qw] = packed[i];
  else v[int64_t(row) * vw + col - qw - kw] = packed[i];
}

__global__ void PackedSiluKernel(const __nv_bfloat16* packed, __nv_bfloat16* out,
                                 int width, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int64_t offset = (i / width) * (2 * width) + i % width;
  const float g = __bfloat162float(packed[offset]);
  const float u = __bfloat162float(packed[offset + width]);
  out[i] = __float2bfloat16((g / (1.0f + __expf(-g))) * u);
}

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

__global__ void AddKernel(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                          __nv_bfloat16* __restrict__ out, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

__global__ void AddKernelF32(const float* __restrict__ a, const float* __restrict__ b,
                             float* __restrict__ out, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  out[i] = a[i] + b[i];
}

__global__ void SiluAndMulKernel(const __nv_bfloat16* __restrict__ gate,
                                 const __nv_bfloat16* __restrict__ up,
                                 __nv_bfloat16* __restrict__ out, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float g = __bfloat162float(gate[i]);
  const float silu = g / (1.0f + __expf(-g));
  out[i] = __float2bfloat16(silu * __bfloat162float(up[i]));
}

__global__ void SiluAndMulKernelF32(const float* __restrict__ gate, const float* __restrict__ up,
                                    float* __restrict__ out, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const float g = gate[i];
  out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

constexpr int kThreads = 256;

uint32_t BlocksFor(int64_t n) { return static_cast<uint32_t>((n + kThreads - 1) / kThreads); }

}  // namespace

Status SplitQkv(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor& k, Tensor& v) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  SplitQkvKernel<<<BlocksFor(packed.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<const __nv_bfloat16*>(packed.Data()), static_cast<__nv_bfloat16*>(q.Data()),
      static_cast<__nv_bfloat16*>(k.Data()), static_cast<__nv_bfloat16*>(v.Data()),
      q.Dim(1), k.Dim(1), v.Dim(1), packed.Numel());
  return CudaError(cudaGetLastError(), "split qkv");
}

Status PackedSiluAndMul(ExecutionContext& ctx, const Tensor& packed, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  PackedSiluKernel<<<BlocksFor(out.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<const __nv_bfloat16*>(packed.Data()), static_cast<__nv_bfloat16*>(out.Data()),
      out.Dim(1), out.Numel());
  return CudaError(cudaGetLastError(), "packed silu");
}

Status Add(ExecutionContext& ctx, const Tensor& a, const Tensor& b, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t n = a.Numel();
  const cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream());
  if (a.GetDataType() == DataType::kBFloat16) {
    AddKernel<<<BlocksFor(n), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.Data()), static_cast<const __nv_bfloat16*>(b.Data()),
        static_cast<__nv_bfloat16*>(out.Data()), n);
  } else {
    AddKernelF32<<<BlocksFor(n), kThreads, 0, stream>>>(
        static_cast<const float*>(a.Data()), static_cast<const float*>(b.Data()),
        static_cast<float*>(out.Data()), n);
  }
  return CudaError(cudaGetLastError(), "add launch");
}

Status SiluAndMul(ExecutionContext& ctx, const Tensor& gate, const Tensor& up, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t n = gate.Numel();
  const cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream());
  if (gate.GetDataType() == DataType::kBFloat16) {
    SiluAndMulKernel<<<BlocksFor(n), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.Data()),
        static_cast<const __nv_bfloat16*>(up.Data()),
        static_cast<__nv_bfloat16*>(out.Data()), n);
  } else {
    SiluAndMulKernelF32<<<BlocksFor(n), kThreads, 0, stream>>>(
        static_cast<const float*>(gate.Data()), static_cast<const float*>(up.Data()),
        static_cast<float*>(out.Data()), n);
  }
  return CudaError(cudaGetLastError(), "silu_and_mul launch");
}

}  // namespace inferx::ops::cuda
