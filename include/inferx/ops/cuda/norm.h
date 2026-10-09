#ifndef INFERX_OPS_CUDA_NORM_H_
#define INFERX_OPS_CUDA_NORM_H_

#include "inferx/ops/norm.h"

namespace inferx::ops::cuda {

// Backend entry points require validated, nonempty tensors and matching
// normalization config. Callers own shape, dtype/device agreement, aliasing,
// epsilon, and 32-bit extent validation. Leading dimensions are flattened
// into rows; the last dimension is the hidden size.

Status RmsNorm(OpContext&, const Tensor& in, const Tensor& weight, Tensor& out,
               const NormConfig&);

Status AddRmsNorm(OpContext&, const Tensor& in, Tensor& residual, const Tensor& weight,
                  Tensor& out, const NormConfig&);

Status LayerNorm(OpContext&, const Tensor& in, const Tensor& weight,
                 const std::optional<Tensor>& bias, Tensor& out, const NormConfig&);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_NORM_H_
