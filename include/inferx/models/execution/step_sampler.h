#ifndef INFERX_MODELS_EXECUTION_STEP_SAMPLER_H_
#define INFERX_MODELS_EXECUTION_STEP_SAMPLER_H_

#include <cstdint>
#include <vector>

#include "inferx/models/execution/batch_buffers.h"
#include "inferx/models/execution/step_plan.h"
#include "inferx/models/execution/stream_waiter.h"
#include "inferx/models/model_runner.h"  // Tensor, Stream, DeviceRuntime, ops::OpContext
#include "inferx/sampling/sampler.h"

namespace inferx::models::execution {

/// \brief Samples one token per selected row and returns them on the host.
///
/// Sampling stays eager: parameter uploads and greedy/stochastic selection are
/// per-step host decisions a graph replay must not freeze. Drains the stream
/// via `waiter` before returning, so the step's work is complete on success.
StatusOr<std::vector<int32_t>> SampleStep(sampling::Sampler& sampler, ops::OpContext& ctx,
                                          DeviceRuntime& runtime, const Stream& stream,
                                          StreamWaiter& waiter, BatchBuffers& buffers,
                                          const Tensor& logits, const StepPlan& plan,
                                          int64_t vocab_size);

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_STEP_SAMPLER_H_
