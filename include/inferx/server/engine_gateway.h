// The seam between the coroutine HTTP front end and the single-threaded
// InferX engine (see server.md §8). One engine thread steps the scheduler
// and model runner exactly like `bench workload` does; new tokens cross to
// the I/O coroutines as value-type events in per-request queues with asio
// notifications. The engine thread never blocks on I/O; I/O threads never
// touch engine state.
#ifndef INFERX_SERVER_ENGINE_GATEWAY_H_
#define INFERX_SERVER_ENGINE_GATEWAY_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <boost/asio.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>

#include "inferx/cache/cache_config.h"
#include "inferx/core/status.h"
#include "inferx/engine/execution_config.h"
#include "inferx/engine/request.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_config.h"
#include "inferx/models/model_runner.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx::server {

/// \brief One event for a submitted completion.
struct CompletionEvent {
  enum class Kind { kDelta, kFinish, kError };
  Kind kind = Kind::kDelta;
  std::string delta;      ///< kDelta: incremental detokenized text (may be empty).
  FinishReason reason = FinishReason::kStopped;  ///< kFinish.
  int prompt_tokens = 0;      ///< kFinish.
  int completion_tokens = 0;  ///< kFinish.
  bool admission = false;     ///< kError: rejected before the engine (503).
  std::string message;        ///< kError.
};

/// \brief Unbounded engine-to-consumer event queue.
///
/// Payload lives in a mutex-guarded deque (engine pushes, consumer drains);
/// a one-slot thread-safe notification channel wakes the consumer. Events
/// are never dropped: a terminal event always reaches a live consumer even
/// if it lags far behind (a full-notification is level-triggered by the
/// remaining deque contents). Backpressure is enforced by the gateway via
/// queue size, not by dropping.
class EventQueue : public std::enable_shared_from_this<EventQueue> {
 public:
  explicit EventQueue(boost::asio::io_context::executor_type executor)
      : notify_(executor, 1) {}

  /// Engine thread. Appends one event and arms the wake-up. The
  /// concurrent channel synchronizes access from the engine thread against
  /// consumer receives on the I/O threads; `try_send` completes a parked
  /// receiver directly and otherwise leaves one buffered token. Delivery
  /// must be armed by the push itself: the 2026-09-22 tail stall was a
  /// step that pushed a request's terminal event behind a notification the
  /// consumer had already consumed, with no further wake-up ever posted.
  /// A `try_send` that finds the one slot taken is safe: the pending token
  /// wakes the consumer, and TakeAll re-checks the deque when it runs.
  void Push(CompletionEvent event) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      events_.push_back(std::move(event));
    }
    notify_.try_send(std::exception_ptr{});
  }

  /// Engine thread. Undelivered event count (slow-consumer detection).
  std::size_t Size() {
    std::lock_guard<std::mutex> lock(mu_);
    return events_.size();
  }

  /// Consumer coroutine. Returns all queued events, waiting for the next
  /// one when empty; nullopt once the notifier is closed.
  boost::asio::awaitable<std::optional<std::vector<CompletionEvent>>> TakeAll() {
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (!events_.empty()) {
          std::vector<CompletionEvent> out(std::make_move_iterator(events_.begin()),
                                           std::make_move_iterator(events_.end()));
          events_.clear();
          co_return out;
        }
      }
      auto [eptr] = co_await notify_.async_receive(
          boost::asio::as_tuple(boost::asio::use_awaitable));
      if (eptr) co_return std::nullopt;
    }
  }

 private:
  std::mutex mu_;
  std::deque<CompletionEvent> events_;
  boost::asio::experimental::concurrent_channel<void(std::exception_ptr)> notify_;
};

/// \brief A completion handed to the engine.
struct CompletionSpec {
  std::vector<int> prompt;          ///< Encoded prompt token ids.
  sampling::SamplingParams params;  ///< Sampling configuration.
};

struct SubmitResult {
  std::uint64_t id = 0;  ///< Engine request id; also the API completion id.
  std::shared_ptr<EventQueue> events;
};

class EngineGateway {
 public:
  /// \brief Creates the model runner (loading weights) and starts the
  /// engine thread. `io` must outlive the gateway.
  EngineGateway(boost::asio::io_context& io, ModelConfig model_config,
                CacheConfig cache_config, SchedulerConfig scheduler_config,
                ExecutionConfig execution_config,
                std::shared_ptr<Tokenizer> tokenizer);
  ~EngineGateway();

  EngineGateway(const EngineGateway&) = delete;
  EngineGateway& operator=(const EngineGateway&) = delete;

  /// \brief Encodes prompt text. Serialized: the tokenizer is shared with
  /// the engine thread's detokenizer.
  StatusOr<std::vector<int>> Encode(const std::string& text) const;

  /// \brief Submits a completion; never blocks. Fails fast with
  /// ResourceExhausted while waiting+running requests are at capacity
  /// (the session maps that to 503).
  StatusOr<SubmitResult> Submit(CompletionSpec spec);

  /// \brief Aborts a request (disconnect, consumer error). Idempotent;
  /// applied on the next engine step.
  void Cancel(std::uint64_t id);

  /// \brief Stops the engine thread. Idempotent; joins.
  void Shutdown();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_ENGINE_GATEWAY_H_
