#ifndef INFERX_MODELS_EXECUTION_STEP_PLAN_H_
#define INFERX_MODELS_EXECUTION_STEP_PLAN_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/models/execution/batch_buffers.h"  // PackedStep
#include "inferx/models/execution/request_state.h"
#include "inferx/models/model_runner.h"  // Status, StatusOr, SchedulerOutput, RequestId

namespace inferx::models::execution {

/// One scheduled request's plan within a step, resolved against its record.
struct StepRequest {
  RequestRecord* record;
  int start;    ///< Committed watermark the chunk begins at.
  int chunk;    ///< Tokens allocated this step.
  int end;      ///< Planned end; committed only after success.
  bool sample;  ///< Whether this step selects an output row for it.
};

/// Immutable capacities the planner validates against. Built once at init so
/// planning does not depend on the config structs.
struct PlanLimits {
  int max_tokens = 0;
  int max_seqs = 0;
  int64_t num_kv_blocks = 0;
  int64_t block_size = 0;
  int64_t max_context = 0;
  int64_t vocab_size = 0;
  bool has_recurrent = false;
};

/// Watermarks the scheduler announced for cached requests (id -> planned end).
using ScheduledEnds = absl::flat_hash_map<RequestId, int>;

/// Everything the device stages need, produced without touching the device.
struct StepPlan {
  std::vector<StepRequest> requests;
  PackedStep packed;
  bool pure_decode = true;

  int batch() const { return static_cast<int>(requests.size()); }
  int num_rows() const { return static_cast<int>(packed.logit_rows.size()); }
};

/// \brief Validates `output` against the request table and limits and packs it.
///
/// Pure host code: no stream, no allocation on device. Does not mutate the
/// records, so a failed plan leaves committed progress untouched.
StatusOr<StepPlan> BuildStepPlan(const SchedulerOutput& output, RequestStateTable& requests,
                                 KvBlockPool& kv, const PlanLimits& limits,
                                 const ScheduledEnds& expected_ends);

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_STEP_PLAN_H_
