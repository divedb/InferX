#include "inferx/models/layers/rms_norm.h"

#include <utility>

#include "inferx/ops/norm.h"

namespace inferx {
namespace models {

StatusOr<RMSNorm> RMSNorm::Create(int64_t hidden_size, float eps, DataType dtype,
                                  Device device, bool round_before_weight) {
  if (hidden_size <= 0) {
    return InvalidArgumentError("RMSNorm needs a positive hidden size, got ", hidden_size);
  }
  INFERX_ASSIGN_OR_RETURN(Tensor weight, Tensor::Empty(dtype, Shape({hidden_size}), device));
  ops::NormConfig config;
  config.kind = ops::NormKind::kRms;
  config.eps = eps;
  config.round_before_weight = round_before_weight;
  return RMSNorm(std::move(weight), hidden_size, config);
}

Status RMSNorm::Forward(ops::OpContext& ctx, const Tensor& x, Tensor& out) const {
  return ops::RmsNorm(ctx, x, weight_, out, config_);
}

Status RMSNorm::AddForward(ops::OpContext& ctx, const Tensor& x, Tensor& residual,
                           Tensor& out) const {
  return ops::AddRmsNorm(ctx, x, residual, weight_, out, config_);
}

}  // namespace models
}  // namespace inferx
