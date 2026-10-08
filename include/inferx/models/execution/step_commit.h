#ifndef INFERX_MODELS_EXECUTION_STEP_COMMIT_H_
#define INFERX_MODELS_EXECUTION_STEP_COMMIT_H_

#include "absl/types/span.h"
#include "inferx/models/execution/step_plan.h"

namespace inferx::models::execution {

/// \brief Commits progress for every request in `plan` and appends its output.
///
/// Call only after execution and sampling succeeded. `sampled` holds one token
/// per sampled row (empty when `sample_output` is false). Requests whose chunk
/// ends mid-prompt report an empty sample. On a non-sampling group rank the
/// output owner draws the token and publishes it via ApplyAcceptedTokens.
void CommitStep(const StepPlan& plan, absl::Span<const int32_t> sampled, bool sample_output,
                ModelRunnerOutput& out);

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_STEP_COMMIT_H_
