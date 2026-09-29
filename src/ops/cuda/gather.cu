#include <cuda_runtime.h>

#include "inferx/core/dtype.h"
#include "inferx/ops/cuda/gather.h"

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

/// One block per output row; rows whose index falls outside the shard's
/// range write zeros, so the cross-rank sum completes the embedding.
__global__ void GatherRowsRangeKernel(const char* __restrict__ src,
                                      const int32_t* __restrict__ indices,
                                      char* __restrict__ out, int64_t row_bytes,
                                      int64_t row_begin, int64_t src_rows) {
  const int64_t row = blockIdx.x;
  const int64_t local = static_cast<int64_t>(indices[row]) - row_begin;
  char* dst = out + row * row_bytes;
  if (local < 0 || local >= src_rows) {
    for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) dst[i] = 0;
    return;
  }
  const char* in = src + local * row_bytes;
  for (int64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) dst[i] = in[i];
}

/// One block per row; copies the source row into its column block of a
/// wider destination row.
__global__ void CopyColumnBlockKernel(const char* __restrict__ src, char* __restrict__ dst,
                                      int64_t src_cols, int64_t dst_cols, int64_t col_begin,
                                      int64_t elem_bytes) {
  const int64_t row = blockIdx.x;
  const char* in = src + row * src_cols * elem_bytes;
  char* out = dst + (row * dst_cols + col_begin) * elem_bytes;
  for (int64_t i = threadIdx.x * elem_bytes; i < src_cols * elem_bytes; i += blockDim.x * elem_bytes) {
    for (int64_t b = 0; b < elem_bytes; ++b) out[i + b] = in[i + b];
  }
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

Status GatherRowsRange(ExecutionContext& ctx, const Tensor& src, const Tensor& indices,
                       Tensor& out, int64_t row_begin) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t row_bytes = out.Dim(1) * DataTypeByteSize(src.GetDataType(), 1);
  GatherRowsRangeKernel<<<static_cast<uint32_t>(out.Dim(0)), 256, 0,
                          static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const char*>(src.Data()), static_cast<const int32_t*>(indices.Data()),
      static_cast<char*>(out.Data()), row_bytes, row_begin, src.Dim(0));
  return CudaError(cudaGetLastError(), "gather rows range launch");
}

Status CopyColumnBlock(ExecutionContext& ctx, const Tensor& src, Tensor& dst,
                       int64_t col_begin) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t elem_bytes = DataTypeByteSize(src.GetDataType(), 1);
  CopyColumnBlockKernel<<<static_cast<uint32_t>(src.Dim(0)), 256, 0,
                          static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const char*>(src.Data()), static_cast<char*>(dst.Data()), src.Dim(1),
      dst.Dim(1), col_begin, elem_bytes);
  return CudaError(cudaGetLastError(), "copy column block launch");
}

}  // namespace inferx::ops::cuda
