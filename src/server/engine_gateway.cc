// EngineGateway implementation: the engine thread loop and the channel
// fan-out. The loop mirrors `bench workload`'s step sequence; the gateway
// adds submission, cancellation, detokenization, and event delivery.
//
// Threading contract (server.md §7): the scheduler, runner, and every
// `Live` entry are touched only by the engine thread. I/O threads interact
// exclusively through the mutex-guarded pending/cancel lists, atomics, and
// per-request channels. Events leave the engine thread inside one posted
// lambda per step.
#include "inferx/server/engine_gateway.h"

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include "inferx/core/status_util.h"
#include "inferx/server/completion_signal.h"
#include "inferx/server/text_delta.h"

namespace inferx::server {
namespace {

// Undelivered events tolerated per request before the consumer is judged
// hopelessly slow and its request aborted (the terminal event still
// reaches it; only further generation stops).
constexpr std::size_t kSlowConsumerEvents = 4096;

}  // namespace

struct EngineGateway::Impl {
  Impl(boost::asio::io_context& io_arg, ModelConfig model_config,
       CacheConfig cache_config, SchedulerConfig scheduler_config,
       ExecutionConfig execution_config, std::shared_ptr<Tokenizer> tokenizer_arg)
      : io(io_arg), tokenizer(std::move(tokenizer_arg)) {
    runner = Take(ModelRunner::Create(model_config, cache_config, scheduler_config,
                                      execution_config));
    const int64_t eos = runner->checkpoint_config().eos_token_id;
    if (eos < 0) {
      throw std::runtime_error("model config has no eos_token_id");
    }
    scheduler_config.queue_capacity = std::max(scheduler_config.queue_capacity, 1);
    capacity = scheduler_config.queue_capacity + scheduler_config.max_num_seqs;
    scheduler = std::make_unique<Scheduler>(scheduler_config, runner->kv_pool(),
                                            static_cast<TokenId>(eos));
  }

  boost::asio::io_context& io;
  std::shared_ptr<Tokenizer> tokenizer;
  std::unique_ptr<ModelRunner> runner;
  std::unique_ptr<Scheduler> scheduler;
  int capacity = 0;

  std::thread engine;
  std::atomic<bool> stopping{false};
  std::atomic<bool> joined{false};
  CompletionSignal drained;  // Fired from the engine thread at loop exit.

  // Shared between I/O threads and the engine thread.
  std::mutex mu;
  std::condition_variable cv;
  struct Pending {
    std::uint64_t id;
    CompletionSpec spec;
    std::shared_ptr<EventQueue> events;
  };
  std::vector<Pending> pending;
  std::vector<std::uint64_t> cancels;
  std::atomic<std::uint64_t> next_id{1};
  std::atomic<int> in_flight{0};

  // Engine-thread-only.
  struct Live {
    std::shared_ptr<EventQueue> events;
    std::vector<int> generated;  // Tokens produced so far.
    TextDelta text;              // Suffix of `generated`'s decode already emitted.
    int prompt_len = 0;
    int computed = 0;  // Prompt tokens with KV computed (prefill progress).
  };
  absl::flat_hash_map<std::uint64_t, Live> live;

  /// One step's outbound events. Queues are unbounded and never drop, so a
  /// terminal event always reaches a live consumer; each Push arms the
  /// queue's wake-up itself.
  struct Outbound {
    std::uint64_t id;
    std::shared_ptr<EventQueue> events;
    CompletionEvent event;
  };

  void Emit(std::vector<Outbound> batch) {
    if (batch.empty()) return;
    for (Outbound& out : batch) {
      out.events->Push(std::move(out.event));
    }
  }

  /// Slow-consumer guard: abort requests whose consumer fell implausibly
  /// far behind (dead socket with the connection still nominally open).
  void AbortSlowConsumers() {
    std::vector<std::uint64_t> stalled;
    for (const auto& [id, entry] : live) {
      if (entry.events->Size() > kSlowConsumerEvents) stalled.push_back(id);
    }
    if (!stalled.empty()) scheduler->AbortRequests(stalled);
  }

  /// Thread-safe cancellation request (also the public Cancel path). The
  /// engine applies it on the next step; unknown/finished ids are ignored
  /// by the scheduler, so stale cancels are harmless.
  void RequestCancel(std::uint64_t id) {
    {
      std::lock_guard<std::mutex> lock(mu);
      cancels.push_back(id);
    }
    cv.notify_all();
  }

  void FailAll(std::string_view message) {
    std::vector<Outbound> batch;
    for (auto& [id, entry] : live) {
      CompletionEvent event;
      event.kind = CompletionEvent::Kind::kError;
      event.message = std::string(message);
      batch.push_back({id, entry.events, std::move(event)});
      in_flight.fetch_sub(1);
    }
    live.clear();
    Emit(std::move(batch));
    stopping.store(true);
  }

  void EngineLoop() {
    std::unique_lock<std::mutex> lock(mu);
    for (;;) {
      cv.wait(lock, [&] {
        return stopping.load() || !pending.empty() || scheduler->HasRequests();
      });
      if (stopping.load() && pending.empty() && !scheduler->HasRequests()) break;

      std::vector<Pending> submits = std::move(pending);
      pending.clear();
      std::vector<std::uint64_t> aborts = std::move(cancels);
      cancels.clear();
      lock.unlock();

      std::vector<Outbound> batch;
      for (Pending& submit : submits) {
        const int prompt_len = static_cast<int>(submit.spec.prompt.size());
        const Status admitted = scheduler->AddRequest(
            Request(submit.id, std::move(submit.spec.prompt), submit.spec.params));
        if (!admitted.ok()) {
          CompletionEvent event;
          event.kind = CompletionEvent::Kind::kError;
          event.admission = true;
          event.message = std::string(admitted.message());
          batch.push_back({submit.id, submit.events, std::move(event)});
          in_flight.fetch_sub(1);
          continue;
        }
        Live entry;
        entry.events = submit.events;
        entry.prompt_len = prompt_len;
        live.emplace(submit.id, std::move(entry));
      }
      if (!aborts.empty()) {
        for (const std::uint64_t id : aborts) {
          if (live.erase(id) > 0) in_flight.fetch_sub(1);
        }
        scheduler->AbortRequests(aborts);
      }

      if (!scheduler->HasRequests()) {
        Emit(std::move(batch));
        lock.lock();
        continue;
      }

      auto plan = scheduler->Schedule();
      if (!plan.ok()) {
        FailAll(plan.status().message());
        break;
      }
      auto result = runner->Run(*plan);
      if (!result.ok()) {
        FailAll(result.status().message());
        break;
      }
      if (const Status updated = scheduler->UpdateFromOutput(*plan, *result);
          !updated.ok()) {
        FailAll(updated.message());
        break;
      }

      // Fan out new tokens. The computed-token guard mirrors the workload
      // loop: samples are real output only once the whole prompt has KV.
      for (std::size_t i = 0; i < plan->scheduled.size(); ++i) {
        const ScheduledRequest& sr = plan->scheduled[i];
        const SampledTokens& samples = result->samples[i];
        auto it = live.find(sr.request_id);
        if (it == live.end()) continue;
        Live& entry = it->second;
        entry.computed += sr.num_new_tokens;
        if (entry.computed < entry.prompt_len || samples.token_ids.empty()) continue;
        entry.generated.insert(entry.generated.end(), samples.token_ids.begin(),
                               samples.token_ids.end());
        // Phase-1 detokenizer: full re-decode, emit the new suffix
        // (server.md §7); empty deltas are still events so token timing
        // stays honest.
        auto text = tokenizer->Decode(entry.generated);
        if (!text.ok()) {
          CompletionEvent event;
          event.kind = CompletionEvent::Kind::kError;
          event.message = std::string(text.status().message());
          batch.push_back({sr.request_id, entry.events, std::move(event)});
          scheduler->AbortRequests({sr.request_id});
          continue;
        }
        const std::string_view delta = entry.text.NextDelta(*text);
        if (!delta.empty()) {
          CompletionEvent event;
          event.kind = CompletionEvent::Kind::kDelta;
          event.delta = std::string(delta);
          batch.push_back({sr.request_id, entry.events, std::move(event)});
        }
      }

      // Slow-consumer aborts move requests to the finished list; running
      // them before the PopFinished loop means their terminal events are
      // delivered this step instead of waiting for a next step that never
      // comes once the scheduler drains.
      AbortSlowConsumers();

      while (auto finished = scheduler->PopFinished()) {
        const std::uint64_t id = finished->id();
        auto it = live.find(id);
        if (it == live.end()) continue;
        const FinishReason reason = finished->finish_reason();
        CompletionEvent event;
        if (reason == FinishReason::kAborted || reason == FinishReason::kError) {
          event.kind = CompletionEvent::Kind::kError;
          event.message =
              reason == FinishReason::kAborted ? "request aborted" : "request failed";
        } else {
          // Emit any text held back for UTF-8 settling before the finish,
          // so a finished request's deltas concatenate to its final decode.
          if (auto text = tokenizer->Decode(it->second.generated); text.ok()) {
            const std::string_view rest = it->second.text.Flush(*text);
            if (!rest.empty()) {
              CompletionEvent flush;
              flush.kind = CompletionEvent::Kind::kDelta;
              flush.delta = std::string(rest);
              batch.push_back({id, it->second.events, std::move(flush)});
            }
          }
          event.kind = CompletionEvent::Kind::kFinish;
          event.reason = reason;
          event.prompt_tokens = static_cast<int>(finished->prompt().size());
          event.completion_tokens = static_cast<int>(finished->output().size());
        }
        batch.push_back({id, it->second.events, std::move(event)});
        live.erase(it);
        in_flight.fetch_sub(1);
      }
      Emit(std::move(batch));
      lock.lock();
    }

    // Shutdown or fatal: whatever is still live gets a terminal event so no
    // consumer waits forever. try_send into channels of dead sessions fails
    // harmlessly.
    std::vector<Outbound> batch;
    for (auto& [id, entry] : live) {
      CompletionEvent event;
      event.kind = CompletionEvent::Kind::kError;
      event.message = "server is shutting down";
      batch.push_back({id, entry.events, std::move(event)});
    }
    Emit(std::move(batch));
    live.clear();
    drained.Fire();  // Releases WaitDrained waiters (posted via the channel).
  }
};

EngineGateway::EngineGateway(boost::asio::io_context& io, ModelConfig model_config,
                             CacheConfig cache_config,
                             SchedulerConfig scheduler_config,
                             ExecutionConfig execution_config,
                             std::shared_ptr<Tokenizer> tokenizer)
    : impl_(std::make_unique<Impl>(io, model_config, cache_config, scheduler_config,
                                   execution_config, std::move(tokenizer))) {
  impl_->engine = std::thread([this] { impl_->EngineLoop(); });
}

EngineGateway::~EngineGateway() { Shutdown(); }

StatusOr<SubmitResult> EngineGateway::Submit(CompletionSpec spec) {
  if (impl_->stopping.load()) {
    return ResourceExhaustedError("server is shutting down");
  }
  if (impl_->in_flight.fetch_add(1) + 1 > impl_->capacity) {
    impl_->in_flight.fetch_sub(1);
    return ResourceExhaustedError(
        "request queue is full; retry after in-flight requests complete");
  }
  SubmitResult out;
  out.id = impl_->next_id.fetch_add(1);
  out.events = std::make_shared<EventQueue>(impl_->io.get_executor());
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->pending.push_back({out.id, std::move(spec), out.events});
  }
  impl_->cv.notify_all();
  return out;
}

void EngineGateway::Cancel(std::uint64_t id) { impl_->RequestCancel(id); }

void EngineGateway::RequestStop() {
  if (!impl_.get()) return;
  impl_->stopping.store(true);
  impl_->cv.notify_all();
}

boost::asio::awaitable<void> EngineGateway::WaitDrained() { co_await impl_->drained.Wait(); }

void EngineGateway::Shutdown() {
  if (!impl_.get()) return;
  if (impl_->joined.exchange(true)) return;
  RequestStop();
  if (impl_->engine.joinable()) impl_->engine.join();
}

}  // namespace inferx::server
