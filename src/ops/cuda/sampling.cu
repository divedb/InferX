#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>

#include "inferx/ops/sampling.h"
namespace inferx::ops {
namespace {
struct Candidate {
  float value;
  int index;
};
__device__ Candidate Better(Candidate a, Candidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}
__device__ Candidate Reduce(Candidate a) {
  __shared__ float values[8];
  __shared__ int indices[8];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int delta = 16; delta; delta /= 2) {
    Candidate b{__shfl_down_sync(0xffffffff, a.value, delta),
                __shfl_down_sync(0xffffffff, a.index, delta)};
    a = Better(a, b);
  }
  if (lane == 0) {
    values[warp] = a.value;
    indices[warp] = a.index;
  }
  __syncthreads();
  a = lane < 8 ? Candidate{values[lane], indices[lane]} : Candidate{-INFINITY, INT_MAX};
  if (warp == 0)
    for (int delta = 16; delta; delta /= 2) {
      Candidate b{__shfl_down_sync(0xffffffff, a.value, delta),
                  __shfl_down_sync(0xffffffff, a.index, delta)};
      a = Better(a, b);
    }
  return a;
}
template <class T>
__global__ void Partial(const T* x, int vocab, int parts, float* values, int* indices) {
  int row = blockIdx.x, part = blockIdx.y;
  Candidate a{-INFINITY, INT_MAX};
  for (int j = part * 4096 + threadIdx.x; j < min(vocab, (part + 1) * 4096); j += blockDim.x) {
    float v = static_cast<float>(x[static_cast<int64_t>(row) * vocab + j]);
    if (!isnan(v)) a = Better(a, {v, j});
  }
  a = Reduce(a);
  if (threadIdx.x == 0) {
    values[row * parts + part] = a.value;
    indices[row * parts + part] = a.index;
  }
}
template <class T>
__global__ void Finish(const T* x, int vocab, int parts, const float* values,
                       const int* indices, int* out) {
  const int row = blockIdx.x;
  Candidate a{-INFINITY, INT_MAX};
  for (int j = threadIdx.x; j < parts; j += blockDim.x)
    a = Better(a, {values[row * parts + j], indices[row * parts + j]});
  a = Reduce(a);
  // Match the original sequential scan even for NaN in column zero.
  if (threadIdx.x == 0)
    out[row] = isnan(static_cast<float>(x[static_cast<int64_t>(row) * vocab])) ? 0 : a.index;
}
}  // namespace
Status GreedyArgmax(ExecutionContext& ctx, const Tensor& logits, Tensor& values,
                    Tensor& indices, Tensor& output) {
  if (ctx.device().kind != DeviceKind::kCuda || logits.Rank() != 2 || logits.IsEmpty() ||
      (logits.GetDataType() != DataType::kBFloat16 && logits.GetDataType() != DataType::kFloat))
    return InvalidArgumentError("GreedyArgmax requires CUDA float32/bfloat16 matrix");
  const int batch = logits.Dim(0), vocab = logits.Dim(1), parts = (vocab + 4095) / 4096;
  if (values.GetDataType() != DataType::kFloat || indices.GetDataType() != DataType::kInt32 ||
      output.GetDataType() != DataType::kInt32 || values.Numel() < batch * parts ||
      indices.Numel() < batch * parts || output.Numel() < batch)
    return InvalidArgumentError("GreedyArgmax workspace too small or wrong dtype");
  const Tensor* tensors[] = {&logits, &values, &indices, &output};
  for (const Tensor* t : tensors)
    if (t->Device() != ctx.device())
      return InvalidArgumentError("GreedyArgmax device mismatch");
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  auto stream = static_cast<cudaStream_t>(ctx.stream());
  auto launch = [&]<class T>() {
    Partial<<<dim3(batch, parts), 256, 0, stream>>>(static_cast<const T*>(logits.Data()), vocab,
                                                    parts, static_cast<float*>(values.Data()),
                                                    static_cast<int*>(indices.Data()));
    Finish<<<batch, 256, 0, stream>>>(static_cast<const T*>(logits.Data()), vocab, parts,
                                      static_cast<const float*>(values.Data()),
                                      static_cast<const int*>(indices.Data()),
                                      static_cast<int*>(output.Data()));
  };
  if (logits.GetDataType() == DataType::kBFloat16)
    launch.template operator()<__nv_bfloat16>();
  else
    launch.template operator()<float>();
  const auto error = cudaGetLastError();
  return error == cudaSuccess ? OkStatus()
                              : InternalError("GreedyArgmax: ", cudaGetErrorString(error));
}
}  // namespace inferx::ops
