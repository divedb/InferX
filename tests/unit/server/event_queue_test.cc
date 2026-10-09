// Exercise the engine-to-coroutine handoff without a model or GPU.
#include "inferx/server/engine_gateway.h"

#include <chrono>
#include <future>
#include <thread>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace inferx::server {
namespace {

namespace net = boost::asio;
using namespace std::chrono_literals;

// Stop and join even when a timeout assertion fails.
class IoWorkers {
 public:
  explicit IoWorkers(net::io_context& io)
      : io_(io), work_(net::make_work_guard(io)) {
    for (int i = 0; i < 4; ++i) {
      workers_.emplace_back([&io] { io.run(); });
    }
  }
  ~IoWorkers() {
    io_.stop();
    for (auto& worker : workers_) worker.join();
  }

 private:
  net::io_context& io_;
  net::executor_work_guard<net::io_context::executor_type> work_;
  std::vector<std::thread> workers_;
};

CompletionEvent Delta(int index) {
  CompletionEvent event;
  event.delta = std::to_string(index);
  return event;
}

net::awaitable<std::vector<CompletionEvent>> Drain(
    std::shared_ptr<EventQueue> queue) {
  std::vector<CompletionEvent> received;
  for (;;) {
    auto batch = co_await queue->TakeAll();
    if (!batch) co_return received;
    for (auto& event : *batch) {
      const bool terminal = event.kind != CompletionEvent::Kind::kDelta;
      received.push_back(std::move(event));
      if (terminal) co_return received;
    }
    // Simulate a consumer yielding for socket writes between batches.
    co_await net::post(net::use_awaitable);
  }
}

class EventQueueTest : public testing::TestWithParam<CompletionEvent::Kind> {};

TEST_P(EventQueueTest, CoalescedNotificationsPreserveBacklogAndTerminalEvent) {
  net::io_context io;
  auto queue = std::make_shared<EventQueue>(io.get_executor());
  constexpr int kEvents = 8192;
  for (int i = 0; i < kEvents; ++i) {
    queue->Push(Delta(i));
  }
  CompletionEvent terminal;
  terminal.kind = GetParam();
  queue->Push(terminal);
  // Every push armed the wake-up; the one-slot notifier keeps a single
  // pending token before a slow consumer starts draining.
  io.run();
  EXPECT_EQ(queue->Size(), kEvents + 1u);
  io.restart();
  auto result = net::co_spawn(io, Drain(queue), net::use_future);
  IoWorkers workers(io);
  ASSERT_EQ(result.wait_for(10s), std::future_status::ready);
  const auto events = result.get();
  ASSERT_EQ(events.size(), kEvents + 1u);
  for (int i = 0; i < kEvents; ++i) {
    EXPECT_EQ(events[i].kind, CompletionEvent::Kind::kDelta);
    EXPECT_EQ(events[i].delta, std::to_string(i));
  }
  EXPECT_EQ(events.back().kind, GetParam());
  EXPECT_EQ(queue->Size(), 0u);
}

TEST_P(EventQueueTest, ConcurrentNotificationsAndReceivesOnFourIoThreads) {
  net::io_context io;
  constexpr int kQueues = 16;
  constexpr int kEvents = 2048;
  std::vector<std::shared_ptr<EventQueue>> queues;
  std::vector<std::future<std::vector<CompletionEvent>>> results;
  for (int i = 0; i < kQueues; ++i) {
    queues.push_back(std::make_shared<EventQueue>(io.get_executor()));
    results.push_back(net::co_spawn(io, Drain(queues.back()), net::use_future));
  }
  // Start with every consumer suspended in async_receive on an empty queue.
  io.poll();
  io.restart();
  IoWorkers workers(io);
  std::thread engine([&] {
    for (int i = 0; i < kEvents; ++i) {
      for (const auto& queue : queues) {
        queue->Push(Delta(i));
      }
      // Wake-ups from pushes overlap receives on the I/O threads; the
      // one-slot channel coalesces them safely because TakeAll is
      // level-triggered on the deque.
      if (i % 8 == 0) std::this_thread::yield();
    }
    for (const auto& queue : queues) {
      CompletionEvent terminal;
      terminal.kind = GetParam();
      queue->Push(std::move(terminal));
    }
  });
  engine.join();

  for (int q = 0; q < kQueues; ++q) {
    SCOPED_TRACE(q);
    ASSERT_EQ(results[q].wait_for(10s), std::future_status::ready);
    const auto events = results[q].get();
    ASSERT_EQ(events.size(), kEvents + 1u);
    for (int i = 0; i < kEvents; ++i) {
      EXPECT_EQ(events[i].kind, CompletionEvent::Kind::kDelta);
      EXPECT_EQ(events[i].delta, std::to_string(i));
    }
    EXPECT_EQ(events.back().kind, GetParam());
    EXPECT_EQ(queues[q]->Size(), 0u);
  }
}

TEST_P(EventQueueTest, PushedEventsReachParkedConsumerWithoutSeparateNotify) {
  net::io_context io;
  auto queue = std::make_shared<EventQueue>(io.get_executor());
  auto result = net::co_spawn(io, Drain(queue), net::use_future);
  // Start with the consumer suspended in async_receive on an empty queue.
  io.poll();
  io.restart();
  IoWorkers workers(io);
  // Regression (2026-09-22 tail stall): a step's terminal event was pushed
  // after the queue's only notification had already been consumed; the
  // consumer re-parked and the event stayed undelivered forever. Pushing
  // must arm the wake-up itself — delivery cannot depend on a separate
  // notification the producer can forget or coalesce away.
  for (int i = 0; i < 16; ++i) {
    queue->Push(Delta(i));
  }
  CompletionEvent terminal;
  terminal.kind = GetParam();
  queue->Push(std::move(terminal));
  ASSERT_EQ(result.wait_for(10s), std::future_status::ready);
  const auto events = result.get();
  ASSERT_EQ(events.size(), 17u);
  EXPECT_EQ(events.back().kind, GetParam());
  EXPECT_EQ(queue->Size(), 0u);
}

TEST_P(EventQueueTest, EventPushedAfterConsumerDrainsIsStillDelivered) {
  net::io_context io;
  auto queue = std::make_shared<EventQueue>(io.get_executor());
  auto result = net::co_spawn(io, Drain(queue), net::use_future);
  io.poll();
  io.restart();
  IoWorkers workers(io);
  queue->Push(Delta(0));
  // Let the consumer wake, drain, and re-park: the exact window in which
  // the 2026-09-22 stall stranded a request's terminal event.
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (queue->Size() != 0u && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_EQ(queue->Size(), 0u) << "first delta was never delivered";
  queue->Push(Delta(1));
  CompletionEvent terminal;
  terminal.kind = GetParam();
  queue->Push(std::move(terminal));
  ASSERT_EQ(result.wait_for(10s), std::future_status::ready);
  const auto events = result.get();
  ASSERT_EQ(events.size(), 3u);
  EXPECT_EQ(events[0].delta, "0");
  EXPECT_EQ(events[1].delta, "1");
  EXPECT_EQ(events.back().kind, GetParam());
  EXPECT_EQ(queue->Size(), 0u);
}

INSTANTIATE_TEST_SUITE_P(TerminalEvents, EventQueueTest,
                        testing::Values(CompletionEvent::Kind::kFinish,
                                        CompletionEvent::Kind::kError));

}  // namespace
}  // namespace inferx::server
