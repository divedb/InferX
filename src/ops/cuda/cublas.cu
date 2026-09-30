#include <map>
#include <mutex>
#include <utility>

#include "inferx/ops/cuda/cublas.h"

namespace inferx::ops::cuda {
namespace {

Status CublasError(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    return InternalError(what, " failed with cuBLAS status ", static_cast<int>(status));
  }
  return OkStatus();
}

}  // namespace

StatusOr<cublasHandle_t> AcquireCublas(OpContext& ctx) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  static std::mutex mutex;
  // One handle per (device, stream): a handle shared across streams would
  // let one thread's cublasSetStream reroute another's pending GEMM, which
  // concurrent tensor-parallel ranks on one device would hit immediately.
  static std::map<std::pair<int, void*>, cublasHandle_t> handles;
  const std::lock_guard<std::mutex> lock(mutex);
  const auto key = std::make_pair(ctx.Device().index, static_cast<void*>(ctx.GetStream()));
  auto it = handles.find(key);
  if (it == handles.end()) {
    cublasHandle_t handle = nullptr;
    INFERX_RETURN_IF_ERROR(CublasError(cublasCreate(&handle), "cublasCreate"));
    it = handles.emplace(key, handle).first;
  }
  INFERX_RETURN_IF_ERROR(
      CublasError(cublasSetStream(it->second, static_cast<cudaStream_t>(ctx.GetStream())),
                  "cublasSetStream"));
  return it->second;
}

}  // namespace inferx::ops::cuda
