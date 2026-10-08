#include "inferx/models/execution/step_sampler.h"

#include "absl/types/span.h"

namespace inferx::models::execution {
namespace {

sampling::SamplingMetadata BuildMetadata(const StepPlan& plan, int64_t vocab_size) {
  std::vector<sampling::SamplingMetadata::PerRequest> per(plan.num_rows());
  size_t row = 0;
  for (const auto& step : plan.requests) {
    if (!step.sample) continue;
    per[row++] = {&step.record->params, step.record->generated};
  }
  return sampling::SamplingMetadata::Build(vocab_size, absl::MakeConstSpan(per));
}

}  // namespace

StatusOr<std::vector<int32_t>> SampleStep(sampling::Sampler& sampler, ops::OpContext& ctx,
                                          DeviceRuntime& runtime, const Stream& stream,
                                          StreamWaiter& waiter, BatchBuffers& buffers,
                                          const Tensor& logits, const StepPlan& plan,
                                          int64_t vocab_size) {
  const int num_rows = plan.num_rows();
  const sampling::SamplingMetadata metadata = BuildMetadata(plan, vocab_size);

  auto samples = sampler.Sample(ctx, logits, metadata);
  if (!samples.ok()) return samples.status();
  if (samples->sampled_token_ids.Dim(0) != num_rows) {
    return InternalError("sampler returned malformed results for the scheduled batch");
  }

  std::vector<int32_t> sampled;
  if (buffers.host_samples() != nullptr) {
    INFERX_RETURN_IF_ERROR(runtime.CopyAsync(buffers.host_samples(),
                                             samples->sampled_token_ids.Data(),
                                             num_rows * sizeof(int32_t),
                                             CopyKind::kDeviceToHost, stream));
    INFERX_RETURN_IF_ERROR(waiter.Wait());
    sampled.assign(buffers.host_samples(), buffers.host_samples() + num_rows);
  } else {
    // CPU device: tensors are host-accessible once the stream has drained.
    INFERX_RETURN_IF_ERROR(waiter.Wait());
    const int32_t* ids = samples->sampled_token_ids.DataAs<int32_t>();
    sampled.assign(ids, ids + num_rows);
  }
  return sampled;
}

}  // namespace inferx::models::execution
