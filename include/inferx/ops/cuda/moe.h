#ifndef INFERX_OPS_CUDA_MOE_H_
#define INFERX_OPS_CUDA_MOE_H_

#include <cstdint>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/moe.h"

namespace inferx::ops::cuda {

Status RouteTokens(ExecutionContext& ctx, const Tensor& hidden, const Tensor& router_weight,
                   const Tensor* router_bias, const Tensor* correction_bias,
                   const RoutingConfig& config, Tensor& topk_indices, Tensor& topk_weights);
Status ExpertHistogram(ExecutionContext& ctx, const Tensor& topk_indices, int64_t num_experts,
                       Tensor& counts);
Status ScatterSlots(ExecutionContext& ctx, const Tensor& topk_indices, const Tensor& topk_weights,
                    const Tensor& offsets, Tensor& cursor, Tensor& token_rows,
                    Tensor& weights_by_slot);
Status GatherRoutedTokens(ExecutionContext& ctx, const Tensor& hidden, const Tensor& token_rows,
                          Tensor& out);
Status ScatterRoutedOutputs(ExecutionContext& ctx, const Tensor& expert_rows,
                            const Tensor& token_rows, const Tensor& weights_by_slot,
                            int64_t begin, int64_t count, Tensor& out);

}  // namespace inferx::ops::cuda

#endif  // INFERX_OPS_CUDA_MOE_H_
