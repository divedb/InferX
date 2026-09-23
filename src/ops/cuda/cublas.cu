#include "ops/cuda/cublas.h"

#include <mutex>
#include <unordered_map>

namespace inferx::ops::cuda {
namespace {

Status CublasError(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    return InternalError(what, " failed with cuBLAS status ", static_cast<int>(status));
  }
  return OkStatus();
}

}  // namespace

StatusOr<cublasHandle_t> AcquireCublas(ExecutionContext& ctx) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  static std::mutex mutex;
  static std::unordered_map<int, cublasHandle_t> handles;
  const std::lock_guard<std::mutex> lock(mutex);
  const int ordinal = ctx.device().index;
  auto it = handles.find(ordinal);
  if (it == handles.end()) {
    cublasHandle_t handle = nullptr;
    INFERX_RETURN_IF_ERROR(CublasError(cublasCreate(&handle), "cublasCreate"));
    it = handles.emplace(ordinal, handle).first;
  }
  INFERX_RETURN_IF_ERROR(
      CublasError(cublasSetStream(it->second, static_cast<cudaStream_t>(ctx.stream())),
                  "cublasSetStream"));
  return it->second;
}

}  // namespace inferx::ops::cuda
