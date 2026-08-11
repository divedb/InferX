#pragma once

#include "inferx/core/status.h"

#ifdef INFERX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace inferx::cuda {

/// True when the binary was compiled with CUDA device support. This is
/// distinct from a CUDA device being available to the process at runtime.
inline constexpr bool kEnabled =
#ifdef INFERX_WITH_CUDA
    true;
#else
    false;
#endif

#ifdef INFERX_WITH_CUDA

Status ErrorToStatus(cudaError_t error, const char* expression,
                     const char* file, int line);

#define INFERX_CUDA_RETURN_IF_ERROR(expr)                             \
  do {                                                                \
    cudaError_t _inferx_cuda_error = (expr);                          \
    if (_inferx_cuda_error != cudaSuccess) [[unlikely]] {             \
      return ::inferx::cuda::ErrorToStatus(_inferx_cuda_error, #expr, \
                                           __FILE__, __LINE__);       \
    }                                                                 \
  } while (0)

int DeviceCount();

inline bool Available() { return DeviceCount() > 0; }

#else

inline constexpr bool Available() { return false; }

#endif

}  // namespace inferx::cuda
