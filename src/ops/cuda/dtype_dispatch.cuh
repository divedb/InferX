#ifndef INFERX_OPS_CUDA_DTYPE_DISPATCH_CUH_
#define INFERX_OPS_CUDA_DTYPE_DISPATCH_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <utility>

#include "inferx/core/dtype.h"
#include "inferx/core/status.h"

namespace inferx::ops::cuda {

/// \brief Dispatches a callable based on the given DataType. Supports FP32, FP16, and BF16.
///
/// \tparam F       The type of the callable to be dispatched.
/// \tparam ...Args The types of the additional arguments to be passed to the callable.
/// \param dtype    The DataType to dispatch on.
/// \param f        The callable to be dispatched. It should accept a pointer to the
///                 corresponding type (float*, __half*, or __nv_bfloat16*) as its first
///                 argument, followed by any additional arguments.
/// \param ...args  The additional arguments to be passed to the callable.
/// \return         The Status returned by the callable or an UnimplementedError if the DataType
///                 is not supported.
template <typename F, typename... Args>
Status DispatchDtype(DataType dtype, F&& f, Args&&... args) {
  switch (dtype) {
    case DataType::kFloat32:
      return std::forward<F>(f)(static_cast<float*>(nullptr), std::forward<Args>(args)...);
    case DataType::kFloat16:
      return std::forward<F>(f)(static_cast<__half*>(nullptr), std::forward<Args>(args)...);
    case DataType::kBFloat16:
      return std::forward<F>(f)(static_cast<__nv_bfloat16*>(nullptr),
                                std::forward<Args>(args)...);
    default:
      return UnimplementedError("CUDA dtype dispatch: expected fp32, fp16, or bf16, got ",
                                DataTypeName(dtype));
  }
}

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_DTYPE_DISPATCH_CUH_
