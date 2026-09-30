#ifndef INFERX_OPS_CUDA_ELEMENTWISE_H_
#define INFERX_OPS_CUDA_ELEMENTWISE_H_

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops::cuda {

Status Add(OpContext& ctx, const Tensor& a, const Tensor& b, Tensor& out);
Status SiluAndMul(OpContext& ctx, const Tensor& gate, const Tensor& up, Tensor& out);
Status SplitQkv(OpContext& ctx, const Tensor& packed, Tensor& q, Tensor& k, Tensor& v);
Status PackedSiluAndMul(OpContext& ctx, const Tensor& packed, Tensor& out);
Status PackedGatedActivation(OpContext& ctx, const Tensor& packed, Tensor& out,
                             Activation act, float oai_alpha, float oai_limit);
Status SplitProjection(OpContext& ctx, const Tensor& packed, Tensor& q, Tensor* gate,
                       Tensor& k, Tensor& v);
Status AddBias(OpContext& ctx, const Tensor& x, const Tensor& bias, Tensor& out);
Status MulSigmoidGate(OpContext& ctx, Tensor& x, const Tensor& gate);
Status MulSigmoidRowGate(OpContext& ctx, Tensor& x, const Tensor& gate);
Status MulScalar(OpContext& ctx, Tensor& x, float scalar);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_ELEMENTWISE_H_
