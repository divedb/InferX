#include "inferx/models/execution/step_commit.h"

#include <optional>

namespace inferx::models::execution {

void CommitStep(const StepPlan& plan, absl::Span<const int32_t> sampled, bool sample_output,
                ModelRunnerOutput& out) {
  size_t row = 0;  // Index into `sampled`; advances only for sampled requests.
  for (const auto& step : plan.requests) {
    RequestRecord& rec = *step.record;
    rec.committed += step.chunk;

    if (!step.sample) {
      out.samples.push_back({rec.request_id, {}, std::nullopt});
      continue;
    }
    if (sample_output) {
      rec.generated += 1;
      rec.last_sampled = sampled[row];
      out.samples.push_back({rec.request_id, {sampled[row]}, std::nullopt});
    } else {
      // Non-sampling group rank: it selects rows so the vocabulary gather
      // runs on every participant, but the output owner draws the token
      // and publishes it via ApplyAcceptedTokens before the next step.
      out.samples.push_back({rec.request_id, {}, std::nullopt});
    }
    ++row;
  }
}

}  // namespace inferx::models::execution
