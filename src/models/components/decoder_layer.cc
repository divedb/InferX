#include "inferx/models/components/decoder_layer.h"

namespace inferx::components {

Status RunFeedForward(const std::variant<SwiGluConfig, MoeConfig>& config,
                      const std::variant<SwiGluWeights, MoeWeights>& weights,
                      const Tensor& normed, MlpWorkspace& ws, Tensor* packed_buffer,
                      ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                      std::string_view prefix, Tensor& mixed_out) {
  if (const auto* dense = std::get_if<SwiGluConfig>(&config)) {
    return RunSwiGlu(*dense, std::get<SwiGluWeights>(weights), normed, ws, packed_buffer, ctx,
                     trace, prefix, mixed_out);
  }
  return RunMoe(std::get<MoeConfig>(config), std::get<MoeWeights>(weights), normed, ctx,
                mixed_out);
}

}  // namespace inferx::components
