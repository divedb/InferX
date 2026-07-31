#include "inferx/kernels/smoke.h"

#include <type_traits>

#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {
namespace {

// Included and asserted here rather than in a .cc, because the question is
// specifically whether it holds under *nvcc*: TensorView is the type §7.1 says
// may cross the kernel-launch boundary, and that claim is only worth anything
// if nvcc's frontend agrees. The header also drags in status.h and absl, which
// makes this the canary for our core headers being nvcc-parseable at all --
// exactly what fails below the CUDA floor (docs/ARCHITECTURE.md R1).
static_assert(std::is_trivially_copyable_v<TensorView>,
              "TensorView must survive the __global__ parameter ABI");

// NOTE for M1: trivially copyable is necessary but *not* sufficient to pass a
// TensorView to a kernel usefully. Its accessors -- Data(), Numel(), Dim() --
// carry no __host__ __device__ annotation, so nvcc treats them as host-only and
// a kernel cannot call them. They are inline but not constexpr, so
// --expt-relaxed-constexpr does not reach them either. A kernel can therefore
// receive a TensorView by value but cannot read anything out of it. Annotating
// the accessors is the fix; it is a core-header change and deliberately not
// made here, where the subject is the toolchain. Until then, kernels take
// unpacked pointers and extents, as this one does.

__global__ void ScaleKernel(float* data, int64_t n, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n) data[i] *= scale;
}

}  // namespace

Status LaunchScale(float* device_data, int64_t n, float scale) {
  if (n == 0) return OkStatus();

  if (device_data == nullptr) {
    return InvalidArgumentError("LaunchScale: null data pointer with n=", n);
  }

  constexpr int kBlock = 256;
  const int64_t grid = (n + kBlock - 1) / kBlock;

  ScaleKernel<<<static_cast<unsigned int>(grid), kBlock>>>(device_data, n,
                                                           scale);

  // Checked separately: the launch error is sticky and would otherwise be
  // reported by the sync below, attributing a bad launch configuration to a
  // synchronization failure.
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  return OkStatus();
}

}  // namespace inferx::kernels
