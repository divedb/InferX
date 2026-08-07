#include <utility>

#include "inferx/server/engine.h"

namespace inferx::server {
namespace {

constexpr size_t kGenerationEventCapacity = 256;

}  // namespace

Generation::Generation() : Generation(Pipe::create(kGenerationEventCapacity)) {}

Generation::Generation(std::pair<EventGenerator, Pipe> pair)
    : events_(std::move(pair.first)), producer_(std::move(pair.second)) {}

folly::coro::Task<StatusOr<std::optional<Generation::Event>>> Generation::Next(
    folly::CancellationToken cancellation) {
  auto next =
      co_await folly::coro::co_withCancellation(cancellation, events_.next());
  if (next) co_return std::optional<Event>(std::move(*next));

  std::lock_guard lock(mutex_);
  if (overflowed_) {
    co_return ResourceExhaustedError("generation event buffer overflowed");
  }
  co_return std::optional<Event>{};
}

void Generation::Cancel() {
  std::lock_guard lock(mutex_);
  cancelled_ = true;
  if (!closed_) {
    std::move(producer_).close();
    closed_ = true;
  }
}

bool Generation::cancelled() const {
  std::lock_guard lock(mutex_);
  return cancelled_;
}

void Generation::Emit(Event event) {
  std::lock_guard lock(mutex_);
  if (finished_ || closed_) return;

  if (!producer_.try_write(std::move(event))) {
    overflowed_ = true;
    std::move(producer_).close();
    closed_ = true;
  }
}

void Generation::Finish(scheduler::FinishReason reason, int32_t generated) {
  std::lock_guard lock(mutex_);
  if (finished_ || closed_) return;

  Event event;
  event.done = true;
  event.reason = reason;
  event.generated = generated;

  if (!producer_.try_write(std::move(event))) overflowed_ = true;
  finished_ = true;
  std::move(producer_).close();
  closed_ = true;
}

}  // namespace inferx::server
