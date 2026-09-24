#include "inferx/models/components/moe.h"

namespace inferx::components {

Status RunMoe(const MoeConfig& config, const MoeWeights& weights, const Tensor& normed,
              ops::ExecutionContext& ctx, Tensor& mixed_out) {
  (void)config;
  (void)weights;
  (void)normed;
  (void)ctx;
  (void)mixed_out;
  return UnimplementedError("expert feed-forward execution is not implemented");
}

}  // namespace inferx::components
