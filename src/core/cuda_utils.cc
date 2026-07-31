#include "inferx/core/cuda_utils.h"

#include "absl/strings/str_cat.h"

namespace inferx {

#ifdef INFERX_WITH_CUDA

Status CudaErrorToStatus(cudaError_t err, const char* expr, const char* file,
                         int line) {
  if (err == cudaSuccess) return OkStatus();

  const std::string detail =
      absl::StrCat(cudaGetErrorName(err), ": ", cudaGetErrorString(err), " [",
                   expr, " at ", file, ":", line, "]");

  switch (err) {
    case cudaErrorMemoryAllocation:
      return ResourceExhaustedError(detail);
    case cudaErrorNoDevice:
    case cudaErrorInsufficientDriver:
    case cudaErrorInitializationError:
      return FailedPreconditionError(detail);
    case cudaErrorInvalidValue:
    case cudaErrorInvalidDevice:
      return InvalidArgumentError(detail);
    default:
      return InternalError(detail);
  }
}

int CudaDeviceCount() {
  int count = 0;

  // TODO(gc):
  // Note:
  // Note that this function may also return error codes from previous,
  // asynchronous launches.
  //
  // Note that this function may also return cudaErrorInitializationError,
  // cudaErrorInsufficientDriver or cudaErrorNoDevice if this call tries to
  // initialize internal CUDA RT state.
  //
  // Note that as specified by cudaStreamAddCallback no CUDA function may be
  // called from callback. cudaErrorNotPermitted may, but is not guaranteed to,
  // be returned as a diagnostic in such case.
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error
    return 0;
  }

  return count;
}

#else

int CudaDeviceCount() { return 0; }

#endif  // INFERX_WITH_CUDA

}  // namespace inferx
