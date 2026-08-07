#include "inferx/server/streaming/event_buffer.h"

#include <utility>

namespace inferx::server::streaming {

StatusOr<std::unique_ptr<EventBuffer>> EventBuffer::Create(size_t max_events) {
  if (max_events == 0) return InvalidArgumentError("max_events must be positive");
  auto pair = Pipe::create(max_events);
  return std::unique_ptr<EventBuffer>(new EventBuffer(
      std::move(pair.first), std::move(pair.second), max_events));
}

EventBuffer::EventBuffer(Generator generator, Pipe pipe, size_t max_events)
    : generator_(std::move(generator)),
      pipe_(std::move(pipe)),
      max_events_(max_events) {}

EventBuffer::~EventBuffer() { Close(); }

PushResult EventBuffer::TryPush(scheduler_client::GenerationEvent event) {
  if (pipe_.isClosed()) return PushResult::kClosed;
  if (!pipe_.try_write(std::move(event))) return PushResult::kFull;
  return PushResult::kQueued;
}

folly::coro::Task<std::optional<scheduler_client::GenerationEvent>>
EventBuffer::Next(folly::CancellationToken cancellation) {
  auto result = co_await folly::coro::co_withCancellation(
      cancellation, generator_.next());
  if (!result) co_return std::nullopt;
  co_return std::move(*result);
}

void EventBuffer::Close() {
  if (!pipe_.isClosed()) std::move(pipe_).close();
}

size_t EventBuffer::queued_events() {
  return max_events_ - pipe_.getAvailableSpace();
}

}  // namespace inferx::server::streaming
