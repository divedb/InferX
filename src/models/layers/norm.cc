#include "inferx/models/layers/norm.h"

#include "inferx/ops/elementwise.h"

namespace inferx::models {

Status Norm::Forward(ops::OpContext& ctx, const Tensor& input, Tensor& output) const {
  switch (config_.kind) {
    case ops::NormKind::kRms:
      return ops::RmsNorm(ctx, input, weights_.weight, output, config_);
    case ops::NormKind::kLayer:
      return ops::LayerNorm(ctx, input, weights_.weight, weights_.bias, output, config_);
  }
  return InvalidArgumentError("unknown norm kind ", static_cast<int>(config_.kind));
}

Status Norm::AddForward(ops::OpContext& ctx, const Tensor& input, Tensor& residual,
                        Tensor& output) const {
  switch (config_.kind) {
    case ops::NormKind::kRms:
      return ops::AddRmsNorm(ctx, input, residual, weights_.weight, output, config_);
    case ops::NormKind::kLayer:
      // No fused kernel: round the sum into the residual stream first, then
      // normalize it.
      INFERX_RETURN_IF_ERROR(ops::Add(ctx, input, residual, residual));
      return ops::LayerNorm(ctx, residual, weights_.weight, weights_.bias, output, config_);
  }
  return InvalidArgumentError("unknown norm kind ", static_cast<int>(config_.kind));
}

}  // namespace inferx::models
