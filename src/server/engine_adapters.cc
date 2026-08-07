#include "engine_adapters.h"

namespace inferx::server {
namespace {

scheduler_client::FinishReason ConvertFinishReason(
    scheduler::FinishReason reason) {
  switch (reason) {
    case scheduler::FinishReason::kStopToken:
      return scheduler_client::FinishReason::kStop;
    case scheduler::FinishReason::kMaxTokens:
    case scheduler::FinishReason::kContextLimit:
      return scheduler_client::FinishReason::kLength;
    case scheduler::FinishReason::kCancelled:
      return scheduler_client::FinishReason::kCancelled;
    case scheduler::FinishReason::kOutOfMemory:
      return scheduler_client::FinishReason::kFailed;
    case scheduler::FinishReason::kNotFinished:
      return scheduler_client::FinishReason::kNone;
  }
  return scheduler_client::FinishReason::kFailed;
}

class EngineGeneration final : public scheduler_client::LegacyGeneration {
 public:
  explicit EngineGeneration(std::shared_ptr<Generation> generation)
      : generation_(std::move(generation)) {}

  bool Next(scheduler_client::LegacyGenerationEvent* event) override {
    Generation::Event legacy;
    if (!generation_->Next(&legacy)) return false;
    event->text = std::move(legacy.text);
    event->terminal = legacy.done;
    event->finish_reason = ConvertFinishReason(legacy.reason);
    event->generated_tokens = static_cast<uint32_t>(legacy.generated);
    return true;
  }

  void Cancel() override { generation_->Cancel(); }
  uint32_t prompt_tokens() const override {
    return static_cast<uint32_t>(generation_->prompt_tokens());
  }

 private:
  std::shared_ptr<Generation> generation_;
};

}  // namespace

StatusOr<std::shared_ptr<scheduler_client::LegacyGeneration>>
EngineSchedulerBackend::Submit(
    const scheduler_client::ScheduledRequest& request) {
  if (engine_ == nullptr) return InternalError("engine is not configured");
  scheduler::SamplingParams sampling;
  sampling.temperature = request.sampling.temperature;
  sampling.top_p = request.sampling.top_p;
  sampling.seed = request.sampling.seed;
  auto submitted = engine_->Submit(request.prompt_tokens,
                                   request.sampling.max_tokens,
                                   request.sampling.stop, sampling);
  if (!submitted.ok()) return submitted.status();
  return std::static_pointer_cast<scheduler_client::LegacyGeneration>(
      std::make_shared<EngineGeneration>(*submitted));
}

observability::MetricsSnapshot EngineMetricsSource::Snapshot() const {
  if (engine_ == nullptr) return {};
  const Engine::Stats stats = engine_->stats();
  return {.running = static_cast<uint64_t>(stats.running),
          .waiting = static_cast<uint64_t>(stats.waiting),
          .blocks_in_use = static_cast<uint64_t>(stats.blocks_in_use),
          .blocks_total = static_cast<uint64_t>(stats.blocks_total),
          .steps = static_cast<uint64_t>(stats.steps),
          .tokens_generated = static_cast<uint64_t>(stats.tokens_generated),
          .last_step_ms = stats.last_step_ms,
          .preemptions = static_cast<uint64_t>(stats.preemptions),
          .cached_blocks = static_cast<uint64_t>(stats.cached_blocks),
          .prefix_hit_tokens = static_cast<uint64_t>(stats.prefix_hit_tokens),
          .prefix_miss_tokens = static_cast<uint64_t>(stats.prefix_miss_tokens),
          .evicted_blocks = static_cast<uint64_t>(stats.evicted_blocks)};
}

}  // namespace inferx::server
