#ifndef INFERX_OPS_CUDA_CUBLAS_H_
#define INFERX_OPS_CUDA_CUBLAS_H_

#include <cublas_v2.h>

#include "inferx/core/status.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops::cuda {

/// \brief Returns the process-wide cuBLAS handle, bound to the context's
///        device and stream.
///
/// The handle is created lazily on first use and never destroyed; the stream
/// is reset on every acquisition so callers can share one handle across
/// streams.
StatusOr<cublasHandle_t> AcquireCublas(ExecutionContext& ctx);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_CUBLAS_H_
