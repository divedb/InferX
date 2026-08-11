#include <cuda_runtime.h>

#include <type_traits>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/tensor_view.h"
#include "inferx/ops/smoke.h"

namespace inferx::ops {
namespace {

// Included and asserted here rather than in a .cc, because the question is
// specifically whether it holds under *nvcc*: TensorView is the type §7.1 says
// may cross the kernel-launch boundary, and that claim is only worth anything
// if nvcc's frontend agrees. The header also drags in status.h and absl, which
// makes this the canary for our core headers being nvcc-parseable at all --
// exactly what fails below the CUDA floor (docs/ARCHITECTURE.md R1).
static_assert(std::is_trivially_copyable_v<TensorView>,
              "TensorView must survive the __global__ parameter ABI");

__global__ void ScaleKernel(float* data, int64_t n, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < n) data[i] *= scale;
}

// The same work, but reading the view rather than unpacked arguments. This is
// the pairing that makes the TensorView accessor annotations meaningful: being
// trivially copyable gets the view *into* the kernel, and INFERX_HOST_DEVICE on
// Numel()/DataAs() is what lets the kernel read it once it arrives. Both halves
// have to hold, and only a launch proves the second one.
__global__ void ScaleViewKernel(TensorView view, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (i < view.Numel()) view.DataAs<float>()[i] *= scale;
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

Status LaunchScale(const TensorView& view, float scale) {
  // Validation is host work, deliberately: the checks below run on types the
  // kernel cannot see (Status, string formatting), and a kernel that has to
  // decide whether its own arguments make sense is a kernel that branches per
  // thread on something the host already knew.
  if (!view.IsDefined()) {
    return InvalidArgumentError("LaunchScale: undefined view");
  }

  if (!view.IsCuda()) {
    return InvalidArgumentError("LaunchScale: view is on ",
                                view.Device().ToString(),
                                ", not a CUDA device");
  }

  if (view.GetDataType() != DataType::kFloat) {
    return InvalidArgumentError("LaunchScale: expected f32, got ",
                                DataTypeName(view.GetDataType()));
  }

  if (view.IsEmpty()) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid = (view.Numel() + kBlock - 1) / kBlock;

  ScaleViewKernel<<<static_cast<unsigned int>(grid), kBlock>>>(view, scale);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());

  return OkStatus();
}

}  // namespace inferx::ops
