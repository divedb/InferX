#pragma once

#include "inferx/core/status.h"

#ifdef INFERX_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

namespace inferx {

/// True when the binary was compiled with device support at all. Distinct from
/// "a GPU is present at runtime".
/// \see CudaDeviceCount().
constexpr bool kCudaEnabled =
#ifdef INFERX_WITH_CUDA
    true;
#else
    false;
#endif

#ifdef INFERX_WITH_CUDA

/// \brief Converts a CUDA error code to an InferX status.
///
/// \param err  The CUDA error code.
/// \param expr The expression that produced the error.
/// \param file The file in which the error occurred.
/// \param line The line number at which the error occurred.
/// \return     The corresponding InferX status.
Status CudaErrorToStatus(cudaError_t err, const char* expr, const char* file,
                         int line);

#define INFERX_CUDA_RETURN_IF_ERROR(expr)                                   \
  do {                                                                      \
    cudaError_t _inferx_cuda_err = (expr);                                  \
    if (_inferx_cuda_err != cudaSuccess) [[unlikely]] {                     \
      return ::inferx::CudaErrorToStatus(_inferx_cuda_err, #expr, __FILE__, \
                                         __LINE__);                         \
    }                                                                       \
  } while (0)

#endif  // INFERX_WITH_CUDA

/// \brief Returns the number of visible CUDA devices.
///
/// \return The number of visible CUDA devices. Returns 0 when the binary has no
///         device support, when no driver is present, or when no GPU is
///         visible.
int CudaDeviceCount();

// True when the process can actually run device work right now. This is the
// predicate GPU-gated tests should use.

/// \brief Returns true if the process can run CUDA device work.
///
/// \return True if CUDA is enabled and at least one CUDA device is visible.
inline bool CudaAvailable() { return kCudaEnabled && CudaDeviceCount() > 0; }

}  // namespace inferx
