#include "inferx/backends/cuda/cuda_utils.h"

#include "absl/strings/str_cat.h"

namespace inferx::cuda {

Status ErrorToStatus(cudaError_t error, const char* expression,
                     const char* file, int line) {
  if (error == cudaSuccess) return OkStatus();

  const std::string detail =
      absl::StrCat(cudaGetErrorName(error), ": ", cudaGetErrorString(error),
                   " [", expression, " at ", file, ":", line, "]");

  switch (error) {
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

int DeviceCount() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();  // Clear the sticky error.
    return 0;
  }
  return count;
}

}  // namespace inferx::cuda
