#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <string_view>

#include "ops/cuda/cublas.h"
#include "ops/cuda/linear.h"

namespace inferx::ops::cuda {
namespace {

// Each warp owns an output channel. Vector loads make the small decode
// projections bandwidth-oriented without padding the token dimension to a
// tensor-core tile. FP32 accumulation, one final BF16 rounding.
template <int Batch>
__global__ void DecodeLinear(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                              __nv_bfloat16* out, int m, int k) {
  const int row = blockIdx.x * 4 + threadIdx.y;
  if (row >= m) return;
  float sum[Batch] = {};
  for (int i = threadIdx.x * 8; i < k; i += 256) {
    const uint4 packed = *reinterpret_cast<const uint4*>(weight + int64_t(row) * k + i);
    const auto* w = reinterpret_cast<const __nv_bfloat16*>(&packed);
    #pragma unroll
    for (int b = 0; b < Batch; ++b) {
      const uint4 input = *reinterpret_cast<const uint4*>(x + b * k + i);
      const auto* v = reinterpret_cast<const __nv_bfloat16*>(&input);
      #pragma unroll
      for (int j = 0; j < 8; ++j) sum[b] = fmaf(float(w[j]), float(v[j]), sum[b]);
    }
  }
  #pragma unroll
  for (int b = 0; b < Batch; ++b) {
    #pragma unroll
    for (int offset = 16; offset; offset /= 2)
      sum[b] += __shfl_down_sync(0xffffffff, sum[b], offset);
    if (threadIdx.x == 0) out[b * m + row] = __float2bfloat16(sum[b]);
  }
}

Status CublasError(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    return InternalError(what, " failed with cuBLAS status ", static_cast<int>(status));
  }
  return OkStatus();
}

}  // namespace

Status Linear(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out) {
  // Row-major out[t, o] = sum_i x[t, i] * weight[o, i] is the column-major
  // problem out^T[o, t] = weight^T(out x in) * x^T(in x tokens), so the
  // checkpoint's [out, in] weight matrix participates with OP_T.
  const int m = static_cast<int>(weight.Dim(0));
  const int n = static_cast<int>(x.Dim(0));
  const int k = static_cast<int>(x.Dim(1));
  static const bool decode_linear = [] {
    const char* flag = std::getenv("INFERX_EXPERIMENTAL_DECODE_LINEAR");
    return flag != nullptr && std::string_view(flag) == "1";
  }();
  if (decode_linear && x.GetDataType() == DataType::kBFloat16 && n <= 4 &&
      k % 256 == 0 && m <= 16384 &&
      reinterpret_cast<uintptr_t>(weight.Data()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(x.Data()) % 16 == 0) {
    INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
    const auto* input = static_cast<const __nv_bfloat16*>(x.Data());
    const auto* weights = static_cast<const __nv_bfloat16*>(weight.Data());
    auto* output = static_cast<__nv_bfloat16*>(out.Data());
    const auto stream = static_cast<cudaStream_t>(ctx.stream());
    switch (n) {
      case 1: DecodeLinear<1><<<(m + 3) / 4, dim3(32, 4), 0, stream>>>(input, weights, output, m, k); break;
      case 2: DecodeLinear<2><<<(m + 3) / 4, dim3(32, 4), 0, stream>>>(input, weights, output, m, k); break;
      case 3: DecodeLinear<3><<<(m + 3) / 4, dim3(32, 4), 0, stream>>>(input, weights, output, m, k); break;
      case 4: DecodeLinear<4><<<(m + 3) / 4, dim3(32, 4), 0, stream>>>(input, weights, output, m, k); break;
    }
    const auto error = cudaGetLastError();
    return error == cudaSuccess ? OkStatus() : InternalError("decode linear: ", cudaGetErrorString(error));
  }
  INFERX_ASSIGN_OR_RETURN(cublasHandle_t handle, AcquireCublas(ctx));
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cudaDataType_t dtype = x.GetDataType() == DataType::kBFloat16 ? CUDA_R_16BF : CUDA_R_32F;
  const cublasStatus_t status = cublasGemmEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, weight.Data(), dtype, k, x.Data(),
      dtype, k, &beta, out.Data(), dtype, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return CublasError(status, "cublasGemmEx");
}

}  // namespace inferx::ops::cuda
