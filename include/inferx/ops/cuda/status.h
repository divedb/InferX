#ifndef INFERX_OPS_CUDA_STATUS_H_
#define INFERX_OPS_CUDA_STATUS_H_

#include <cuda_runtime_api.h>

#include "inferx/core/status.h"

namespace inferx::ops::cuda {

/// \brief Converts a CUDA error code into an InferX Status. Returns OK if the code is
///        cudaSuccess, otherwise returns an InternalError with the provided operation name and
///        the CUDA error string.
///
/// \param error     The CUDA error code to convert.
/// \param operation The name of the operation that produced the error, used in the error
///                  message.
/// \return          An OK status if the error code is cudaSuccess, otherwise an InternalError
///                  with a message describing the failure.
inline Status CudaStatus(cudaError_t error, const char* operation) {
  return error == cudaSuccess
             ? OkStatus()
             : InternalError(operation, " failed: ", cudaGetErrorString(error));
}

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_STATUS_H_
