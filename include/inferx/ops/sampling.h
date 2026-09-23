#pragma once
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
namespace inferx::ops {
// Two-pass greedy reduction. Ties choose the lowest vocabulary index.
// Workspace shape is [batch, ceil(vocab/4096)], output is int32 [batch].
Status GreedyArgmax(ExecutionContext& ctx, const Tensor& logits, Tensor& values,
                    Tensor& indices, Tensor& output);
}  // namespace inferx::ops
