#include "inferx/models/execution/step_launcher.h"

namespace inferx::models::execution {

StatusOr<std::optional<Tensor>> LaunchStep(CausalLanguageModel& model, GraphManager& graphs,
                                           const Stream& stream, const ModelInputs& inputs,
                                           const LogitSelection& selection, GraphPolicy policy,
                                           ForwardContext& fctx) {
  INFERX_ASSIGN_OR_RETURN(const StepExecutionInfo step_info,
                          model.PrepareStep(inputs, selection, fctx));

  if (selection.rows.Numel() == 0) {
    // No request needs an output this step: state advance only.
    INFERX_ASSIGN_OR_RETURN(auto hidden, model.Forward(inputs, fctx));
    (void)hidden;
    return std::optional<Tensor>{};
  }

  const bool eligible = policy.allowed && step_info.graph.eligible;
  auto launch = [&]() -> StatusOr<Tensor> {
    INFERX_ASSIGN_OR_RETURN(auto hidden, model.Forward(inputs, fctx));
    INFERX_ASSIGN_OR_RETURN(auto logits, model.ComputeLogits(hidden, selection, fctx));
    return std::move(logits.values);
  };

  INFERX_ASSIGN_OR_RETURN(auto values,
                          graphs.Execute(policy.key, eligible, step_info.workspace_generation,
                                         stream, launch));
  return std::optional<Tensor>(std::move(values));
}

}  // namespace inferx::models::execution
