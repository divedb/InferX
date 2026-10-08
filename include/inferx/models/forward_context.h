#ifndef INFERX_MODELS_FORWARD_CONTEXT_H_
#define INFERX_MODELS_FORWARD_CONTEXT_H_

#include "inferx/dist/comm.h"
#include "inferx/models/requirements.h"
#include "inferx/ops/op_context.h"

namespace inferx::models {

/// \brief Explicit execution dependencies for one Forward/ComputeLogits
///        call, borrowed and valid only for that execution.
///
/// The runner owns every resource referenced here: the operation context
/// orders work on the execution lane, `state` binds the runner-allocated
/// persistent pools to this model's layers, and `comm` is this rank's
/// communicator matching the loaded parallel topology. A model must not
/// retain any of these between calls.
struct ForwardContext {
  ops::OpContext& ops;       ///< Device, runtime, and stream for this step.
  ModelState& state;         ///< Bound persistent state for this model.
  dist::CommBackend& comm;   ///< Collective channel for this rank.
};

}  // namespace inferx::models

#endif  // INFERX_MODELS_FORWARD_CONTEXT_H_
