#include "inferx/server/streaming/event_buffer.h"

#include <algorithm>
#include <utility>

namespace inferx::server::streaming {

StatusOr<std::unique_ptr<EventBuffer>> EventBuffer::Create(size_t max_events,
                                                            size_t max_bytes) {
  if (max_events == 0) return InvalidArgumentError("max_events must be positive");
  if (max_bytes == 0) max_bytes = max_events * 4096;
  if (max_bytes == 0) return InvalidArgumentError("max_bytes must be positive");
  auto pair = Pipe::create(max_events);
  return std::unique_ptr<EventBuffer>(new EventBuffer(
      std::move(pair.first), std::move(pair.second), max_events, max_bytes));
}

EventBuffer::EventBuffer(Generator generator, Pipe pipe, size_t max_events,
                         size_t max_bytes)
    : generator_(std::move(generator)),
      pipe_(std::move(pipe)),
      max_events_(max_events),
      max_bytes_(max_bytes) {}

size_t EventBuffer::EventBytes(
    const scheduler_client::GenerationEvent& event) {
  return sizeof(event.sequence_number) + sizeof(event.generated_tokens) +
         event.text_delta.size() + event.finish_reason.size() + 1;
}

EventBuffer::~EventBuffer() { Close(); }

PushResult EventBuffer::TryPush(scheduler_client::GenerationEvent event) {
  if (pipe_.isClosed()) return PushResult::kClosed;
  const size_t bytes = EventBytes(event);
  if (bytes > max_bytes_ || queued_bytes_ > max_bytes_ - bytes) {
    return PushResult::kFull;
  }
  if (!pipe_.try_write(std::move(event))) return PushResult::kFull;
  queued_bytes_ += bytes;
  return PushResult::kQueued;
}

folly::coro::Task<std::optional<scheduler_client::GenerationEvent>>
EventBuffer::Next(folly::CancellationToken cancellation) {
  auto result = co_await folly::coro::co_withCancellation(
      cancellation, generator_.next());
  if (!result) co_return std::nullopt;
  queued_bytes_ -= std::min(queued_bytes_, EventBytes(*result));
  co_return std::move(*result);
}

void EventBuffer::Close() {
  if (!pipe_.isClosed()) std::move(pipe_).close();
}

size_t EventBuffer::queued_events() {
  return max_events_ - pipe_.getAvailableSpace();
}

}  // namespace inferx::server::streaming
