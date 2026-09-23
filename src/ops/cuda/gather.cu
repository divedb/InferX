#include <cuda_runtime.h>

#include "inferx/core/dtype.h"
#include "ops/cuda/gather.h"

namespace inferx::ops::cuda {
namespace {

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

/// One block per output row; the block copies that row's bytes.
__global__ void GatherRowsKernel(const char* __restrict__ src, const int32_t* __restrict__ indices,
                                 char* __restrict__ out, int64_t row_bytes) {
  const int64_t row = blockIdx.x;
  const char* in = src + static_cast<int64_t>(indices[row]) * row_bytes;
  char* dst = out + row * row_bytes;
  for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) dst[i] = in[i];
}

}  // namespace

Status GatherRows(ExecutionContext& ctx, const Tensor& src, const Tensor& indices, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t row_bytes = out.Dim(1) * DataTypeByteSize(src.GetDataType(), 1);
  GatherRowsKernel<<<static_cast<uint32_t>(out.Dim(0)), 256, 0,
                     static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const char*>(src.Data()), static_cast<const int32_t*>(indices.Data()),
      static_cast<char*>(out.Data()), row_bytes);
  return CudaError(cudaGetLastError(), "gather rows launch");
}

}  // namespace inferx::ops::cuda
