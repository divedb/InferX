#pragma once

#include <folly/CancellationToken.h>
#include <folly/coro/AsyncPipe.h>
#include <folly/coro/Task.h>

#include <cstddef>
#include <memory>
#include <optional>

#include "inferx/core/status.h"
#include "inferx/server/scheduler_client/scheduler_client.h"

namespace inferx::server::streaming {

enum class PushResult { kQueued, kFull, kClosed };

class EventBuffer {
 public:
  static StatusOr<std::unique_ptr<EventBuffer>> Create(size_t max_events);
  ~EventBuffer();

  EventBuffer(const EventBuffer&) = delete;
  EventBuffer& operator=(const EventBuffer&) = delete;

  PushResult TryPush(scheduler_client::GenerationEvent event);
  folly::coro::Task<std::optional<scheduler_client::GenerationEvent>> Next(
      folly::CancellationToken cancellation = {});
  void Close();
  size_t queued_events();

 private:
  using Pipe = folly::coro::BoundedAsyncPipe<
      scheduler_client::GenerationEvent, false>;
  using Generator = folly::coro::AsyncGenerator<
      scheduler_client::GenerationEvent&&>;

  EventBuffer(Generator generator, Pipe pipe, size_t max_events);
  Generator generator_;
  Pipe pipe_;
  size_t max_events_;
};

}  // namespace inferx::server::streaming
