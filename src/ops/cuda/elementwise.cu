#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/ops/cuda/elementwise.h"

namespace inferx::ops::cuda {
namespace {



__device__ __forceinline__ float GateActivation(float g, Activation act, float alpha,
                                               float limit) {
  switch (act) {
    case Activation::kGeluTanh: {
      const float x = 0.7978845608028654f * (g + 0.044715f * g * g * g);
      return 0.5f * g * (1.0f + tanhf(x));
    }
    case Activation::kSiluOai: {
      const float c = fminf(g, limit);  // Asymmetric: no lower clamp, as trained.
      return c / (1.0f + __expf(-alpha * c));
    }
    default:
      return g / (1.0f + __expf(-g));
  }
}

__global__ void PackedGatedActivationKernel(const __nv_bfloat16* packed, __nv_bfloat16* out,
                                            int width, Activation act, float alpha,
                                            float limit, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int64_t offset = (i / width) * (2 * width) + i % width;
  const float g = __bfloat162float(packed[offset]);
  const float u = __bfloat162float(packed[offset + width]);
  float gate = GateActivation(g, act, alpha, limit);
  float value = gate * u;
  if (act == Activation::kSiluOai) {
    // gpt-oss clamps the up half symmetrically and shifts it by +1.
    value = gate * (fminf(fmaxf(u, -limit), limit) + 1.0f);
  }
  out[i] = __float2bfloat16(value);
}

__global__ void SplitProjectionKernel(const __nv_bfloat16* packed, __nv_bfloat16* q,
                                      __nv_bfloat16* gate, __nv_bfloat16* k,
                                      __nv_bfloat16* v, int qw, int gw, int kw, int vw,
                                      int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int width = qw + gw + kw + vw;
  const int row = i / width, col = i % width;
  const __nv_bfloat16 value = packed[i];
  if (col < qw) q[int64_t(row) * qw + col] = value;
  else if (col < qw + gw) gate[int64_t(row) * gw + col - qw] = value;
  else if (col < qw + gw + kw) k[int64_t(row) * kw + col - qw - gw] = value;
  else v[int64_t(row) * vw + col - qw - gw - kw] = value;
}

__global__ void AddBiasKernel(const __nv_bfloat16* x, const __nv_bfloat16* bias,
                              __nv_bfloat16* out, int width, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = __float2bfloat16(__bfloat162float(x[i]) +
                            __bfloat162float(bias[i % width]));
}

__global__ void MulSigmoidGateKernel(__nv_bfloat16* x, const __nv_bfloat16* gate,
                                     int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float s = 1.0f / (1.0f + __expf(-__bfloat162float(gate[i])));
  x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

__global__ void MulSigmoidRowGateKernel(__nv_bfloat16* x, const __nv_bfloat16* gate,
                                        int width, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float s = 1.0f / (1.0f + __expf(-__bfloat162float(gate[i / width])));
  x[i] = __float2bfloat16(__bfloat162float(x[i]) * s);
}

__global__ void MulScalarKernel(__nv_bfloat16* x, float scalar, int64_t count) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  x[i] = __float2bfloat16(__bfloat162float(x[i]) * scalar);
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
  return cuda::SplitProjection(ctx, packed, q, nullptr, k, v);
}

Status SplitProjection(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor* gate,
                       Tensor& k, Tensor& v) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int gw = gate == nullptr ? 0 : static_cast<int>(gate->Dim(1));
  SplitProjectionKernel<<<BlocksFor(packed.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<const __nv_bfloat16*>(packed.Data()),
      static_cast<__nv_bfloat16*>(q.Data()),
      gate == nullptr ? nullptr : static_cast<__nv_bfloat16*>(gate->Data()),
      static_cast<__nv_bfloat16*>(k.Data()), static_cast<__nv_bfloat16*>(v.Data()),
      static_cast<int>(q.Dim(1)), gw, static_cast<int>(k.Dim(1)),
      static_cast<int>(v.Dim(1)), packed.Numel());
  return CudaError(cudaGetLastError(), "split projection");
}

Status PackedSiluAndMul(ExecutionContext& ctx, const Tensor& packed, Tensor& out) {
  return cuda::PackedGatedActivation(ctx, packed, out, Activation::kSilu, 1.702f, 7.0f);
}

Status PackedGatedActivation(ExecutionContext& ctx, const Tensor& packed, Tensor& out,
                             Activation act, float oai_alpha, float oai_limit) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  PackedGatedActivationKernel<<<BlocksFor(out.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<const __nv_bfloat16*>(packed.Data()), static_cast<__nv_bfloat16*>(out.Data()),
      static_cast<int>(out.Dim(1)), act, oai_alpha, oai_limit, out.Numel());
  return CudaError(cudaGetLastError(), "packed gated activation");
}

Status AddBias(ExecutionContext& ctx, const Tensor& x, const Tensor& bias, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  AddBiasKernel<<<BlocksFor(x.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<const __nv_bfloat16*>(x.Data()),
      static_cast<const __nv_bfloat16*>(bias.Data()),
      static_cast<__nv_bfloat16*>(out.Data()), static_cast<int>(x.Dim(1)), x.Numel());
  return CudaError(cudaGetLastError(), "add bias");
}

Status MulSigmoidGate(ExecutionContext& ctx, Tensor& x, const Tensor& gate) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  MulSigmoidGateKernel<<<BlocksFor(x.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<__nv_bfloat16*>(x.Data()),
      static_cast<const __nv_bfloat16*>(gate.Data()), x.Numel());
  return CudaError(cudaGetLastError(), "mul sigmoid gate");
}

Status MulSigmoidRowGate(ExecutionContext& ctx, Tensor& x, const Tensor& gate) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  MulSigmoidRowGateKernel<<<BlocksFor(x.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<__nv_bfloat16*>(x.Data()),
      static_cast<const __nv_bfloat16*>(gate.Data()), static_cast<int>(x.Dim(1)), x.Numel());
  return CudaError(cudaGetLastError(), "mul sigmoid row gate");
}

Status MulScalar(ExecutionContext& ctx, Tensor& x, float scalar) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  MulScalarKernel<<<BlocksFor(x.Numel()), kThreads, 0, ctx.stream()>>>(
      static_cast<__nv_bfloat16*>(x.Data()), scalar, x.Numel());
  return CudaError(cudaGetLastError(), "mul scalar");
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
