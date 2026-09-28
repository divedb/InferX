#ifndef INFERX_OPS_CUDA_ELEMENTWISE_H_
#define INFERX_OPS_CUDA_ELEMENTWISE_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops::cuda {

Status Add(ExecutionContext& ctx, const Tensor& a, const Tensor& b, Tensor& out);
Status SiluAndMul(ExecutionContext& ctx, const Tensor& gate, const Tensor& up, Tensor& out);
Status SplitQkv(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor& k, Tensor& v);
Status PackedSiluAndMul(ExecutionContext& ctx, const Tensor& packed, Tensor& out);
Status PackedGatedActivation(ExecutionContext& ctx, const Tensor& packed, Tensor& out,
                             Activation act, float oai_alpha, float oai_limit);
Status SplitProjection(ExecutionContext& ctx, const Tensor& packed, Tensor& q, Tensor* gate,
                       Tensor& k, Tensor& v);
Status AddBias(ExecutionContext& ctx, const Tensor& x, const Tensor& bias, Tensor& out);
Status MulSigmoidGate(ExecutionContext& ctx, Tensor& x, const Tensor& gate);
Status MulSigmoidRowGate(ExecutionContext& ctx, Tensor& x, const Tensor& gate);
Status MulScalar(ExecutionContext& ctx, Tensor& x, float scalar);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_ELEMENTWISE_H_
