#ifndef INFERX_MODELS_EXECUTION_STEP_LAUNCHER_H_
#define INFERX_MODELS_EXECUTION_STEP_LAUNCHER_H_

#include <optional>

#include "inferx/models/execution/graph_manager.h"
#include "inferx/models/model_runner.h"  // Tensor, Stream, models::* interfaces

namespace inferx::models::execution {

/// Whether this step may replay a captured graph, and under which key.
struct GraphPolicy {
  bool allowed = false;
  int key = 0;  ///< Exact decode batch size; uniform decode implies one row per sequence.
};

/// \brief Runs one step on the device.
///
/// Returns the flat logits of the selected rows (eager result or the stable
/// view a graph replay fills), or an empty optional when no row was selected
/// (state advance only).
StatusOr<std::optional<Tensor>> LaunchStep(CausalLanguageModel& model, GraphManager& graphs,
                                           const Stream& stream, const ModelInputs& inputs,
                                           const LogitSelection& selection, GraphPolicy policy,
                                           ForwardContext& fctx);

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_STEP_LAUNCHER_H_
