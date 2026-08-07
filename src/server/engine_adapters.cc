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

class EngineGeneration final : public scheduler_client::InProcessGeneration {
 public:
  explicit EngineGeneration(std::shared_ptr<Generation> generation)
      : generation_(std::move(generation)) {}

  folly::coro::Task<
      StatusOr<std::optional<scheduler_client::InProcessGenerationEvent>>>
  Next(folly::CancellationToken cancellation) override {
    auto next = co_await generation_->Next(cancellation);
    if (!next.ok()) co_return next.status();
    if (!next->has_value()) {
      co_return std::optional<scheduler_client::InProcessGenerationEvent>{};
    }
    auto engine_event = std::move(**next);
    scheduler_client::InProcessGenerationEvent event;
    event.text = std::move(engine_event.text);
    event.terminal = engine_event.done;
    event.finish_reason = ConvertFinishReason(engine_event.reason);
    event.generated_tokens = static_cast<uint32_t>(engine_event.generated);
    co_return std::optional<scheduler_client::InProcessGenerationEvent>(
        std::move(event));
  }

  void Cancel() override { generation_->Cancel(); }
  uint32_t prompt_tokens() const override {
    return static_cast<uint32_t>(generation_->prompt_tokens());
  }

 private:
  std::shared_ptr<Generation> generation_;
};

}  // namespace

StatusOr<std::shared_ptr<scheduler_client::InProcessGeneration>>
EngineSchedulerBackend::Submit(
    const scheduler_client::ScheduledRequest& request) {
  if (engine_ == nullptr) return InternalError("engine is not configured");
  scheduler::SamplingParams sampling;
  sampling.temperature = request.sampling.temperature;
  sampling.top_p = request.sampling.top_p;
  sampling.seed = request.sampling.seed;
  auto submitted =
      engine_->Submit(request.prompt_tokens, request.sampling.max_tokens,
                      request.sampling.stop, sampling);
  if (!submitted.ok()) return submitted.status();
  return std::static_pointer_cast<scheduler_client::InProcessGeneration>(
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
