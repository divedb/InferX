// TokenizerPool implementation (docs/tokenizer_process_pool.md, stage 1,
// thread-executor variant). N worker threads each own one FFI tokenizer
// instance; jobs cross from the io side to workers through a mutex-guarded
// bounded deque (workers never hold the mutex during encode), and results
// cross back through per-request mailboxes woken by asio's thread-safe
// concurrent_channel — the same delivery pattern EventQueue uses. One
// repeating timer on the io_context sweeps request deadlines for both
// queued and running jobs.
#include "inferx/server/tokenizer_pool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "inferx/server/completion_signal.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx::server {
namespace {

namespace net = boost::asio;
using clock = std::chrono::steady_clock;
constexpr auto kSweepInterval = std::chrono::milliseconds(100);

/// FNV-1a over the tokenizer asset, reported in every PreparedPrompt so
/// callers can observe which revision encoded their prompt.
std::uint64_t FingerprintFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  std::string blob((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  std::uint64_t hash = 1469598103934665603ull;
  for (const char c : blob) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

/// Result mailbox for one Prepare call: a worker thread completes it, the
/// awaiting coroutine (on an io thread) takes it. The buffered-token
/// concurrent_channel closes the check-then-wait race (same pattern as
/// EventQueue): a try_send with no parked receiver is buffered. Complete is
/// exactly-once, so late deliveries after cancel/timeout are no-ops.
struct Waiter {
  explicit Waiter(net::any_io_executor executor) : wake(executor, 1) {}

  void Complete(StatusOr<PreparedPrompt> result) {  // Any thread.
    {
      std::lock_guard<std::mutex> lock(mu);
      if (delivered) return;
      delivered = true;
      this->result = std::move(result);
    }
    wake.try_send(std::exception_ptr{});
  }

  net::awaitable<StatusOr<PreparedPrompt>> Take() {
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mu);
        if (delivered) co_return std::move(*result);
      }
      auto [eptr] = co_await wake.async_receive(net::as_tuple(net::use_awaitable));
      if (eptr) co_return InternalError("tokenizer pool waiter closed");
    }
  }

  std::mutex mu;
  bool delivered = false;
  std::optional<StatusOr<PreparedPrompt>> result;
  net::experimental::concurrent_channel<void(std::exception_ptr)> wake;
};

struct Job {
  std::uint64_t request_id = 0;
  std::string text;
  clock::time_point deadline;
  std::shared_ptr<Waiter> waiter;
};

/// What a worker is currently running, published under the pool mutex. The
/// deadline sweep and Cancel set `discard` so the eventual result is
/// dropped; the encode itself always runs to completion.
struct Active {
  std::uint64_t request_id = 0;
  clock::time_point deadline;
  bool discard = false;
  std::shared_ptr<Waiter> waiter;  // Null when the worker is idle.
};

}  // namespace

struct TokenizerPool::Impl : std::enable_shared_from_this<Impl> {
  Impl(net::io_context& io_arg, TokenizerPoolConfig config_arg)
      : io(io_arg), config(std::move(config_arg)) {
    config.workers = std::max(config.workers, 1);
    config.queue_max_requests = std::max<std::size_t>(config.queue_max_requests, 1);
    fingerprint = FingerprintFile(config.tokenizer_path);
  }

  net::io_context& io;
  TokenizerPoolConfig config;
  std::uint64_t fingerprint = 0;

  // Guards queue, active slots, accounting, and lifecycle flags. Workers
  // hold it only to hand jobs across — never during encode.
  mutable std::mutex mu;
  std::condition_variable work_cv;
  std::deque<Job> queue;
  std::vector<Active> active;  // One slot per worker, indexed by thread.
  std::size_t bytes_in_use = 0;
  bool draining = false;
  bool stopped = false;

  std::vector<std::thread> threads;

  // Startup gate and exit latch. ready/failed are written under ready_mu by
  // the workers and read racily by admission/health, hence atomics.
  std::mutex ready_mu;
  std::condition_variable ready_cv;
  std::atomic<int> ready_count{0};
  std::atomic<int> failed_count{0};
  std::atomic<int> exited_count{0};
  CompletionSignal drained_signal;  // Every worker thread has exited.

  net::steady_timer sweep_timer{io};

  // ---- worker threads --------------------------------------------------

  void StartWorkers() {
    const std::size_t count = static_cast<std::size_t>(config.workers);
    active.assign(count, Active{});
    for (std::size_t i = 0; i < count; ++i) {
      threads.emplace_back([this, i] { WorkerLoop(i); });
    }
    sweep_timer.async_wait([self = shared_from_this()](const boost::system::error_code& ec) {
      self->Sweep(ec);
    });
  }

  void WorkerLoop(std::size_t index) {
    std::shared_ptr<Tokenizer> tokenizer = LoadAndReportReady();
    if (tokenizer != nullptr) {
      for (;;) {
        Job job;
        {
          std::unique_lock<std::mutex> lock(mu);
          work_cv.wait(lock, [&] { return stopped || !queue.empty() || DrainDone(); });
          if (stopped || queue.empty()) break;  // DrainDone implies empty.
          job = std::move(queue.front());
          queue.pop_front();
          active[index] = Active{job.request_id, job.deadline, false, job.waiter};
        }
        RunJob(index, std::move(job), tokenizer.get());
      }
    }
    if (exited_count.fetch_add(1) + 1 >= config.workers) drained_signal.Fire();
  }

  std::shared_ptr<Tokenizer> LoadAndReportReady() {
    auto loaded = Tokenizer::FromFile(config.tokenizer_path);
    std::shared_ptr<Tokenizer> tokenizer;
    if (loaded.ok()) {
      tokenizer = std::move(*loaded);
      ready_count.fetch_add(1);
    } else {
      std::fprintf(stderr, "inferx serve: tokenizer worker failed to load %s: %s\n",
                   config.tokenizer_path.c_str(),
                   std::string(loaded.status().message()).c_str());
      failed_count.fetch_add(1);
    }
    std::lock_guard<std::mutex> lock(ready_mu);
    ready_cv.notify_all();
    return tokenizer;
  }

  void RunJob(std::size_t index, Job job, Tokenizer* tokenizer) {
    StatusOr<PreparedPrompt> result = InvalidArgumentError("unreachable");
    if (auto ids = tokenizer->Encode(job.text); ids.ok()) {
      PreparedPrompt prepared;
      prepared.request_id = job.request_id;
      prepared.token_ids = std::move(*ids);
      prepared.fingerprint = fingerprint;
      result = std::move(prepared);
    } else {
      // Deterministic input/asset failure: the same text fails again, so it
      // surfaces as a 400, never retried.
      result = InvalidArgumentError("failed to tokenize prompt: ",
                                    ids.status().message());
    }
    {
      std::lock_guard<std::mutex> lock(mu);
      active[index] = Active{};
      bytes_in_use -= job.text.size();
      // On discard the caller was already woken (cancel/timeout); Complete
      // is exactly-once, so simply delivering is also a no-op then.
    }
    job.waiter->Complete(std::move(result));
    work_cv.notify_all();  // A drain waiting on the last active job.
  }

  /// Draining with no admitted work left; `mu` held.
  bool DrainDone() const {
    for (const Active& slot : active) {
      if (slot.waiter != nullptr) return false;
    }
    return draining && queue.empty();
  }

  // ---- io side ------------------------------------------------------------

  void Enqueue(std::uint64_t request_id, std::string text,
               const std::shared_ptr<Waiter>& waiter) {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (draining || stopped) {
        waiter->Complete(FailedPreconditionError("tokenizer pool is shutting down"));
        return;
      }
      if (failed_count.load() >= config.workers) {
        waiter->Complete(
            absl::UnavailableError("tokenizer pool has no available workers"));
        return;
      }
      if (text.size() > config.queue_max_bytes) {
        // A single prompt larger than the whole byte budget can never be
        // admitted; that is an input problem, not congestion.
        waiter->Complete(InvalidArgumentError("prompt too large"));
        return;
      }
      if (queue.size() + BusyCountLocked() + 1 > config.queue_max_requests ||
          bytes_in_use + text.size() > config.queue_max_bytes) {
        waiter->Complete(ResourceExhaustedError(
            "tokenizer queue is full; retry after pending requests complete"));
        return;
      }
      Job job;
      job.request_id = request_id;
      job.text = std::move(text);
      job.deadline = clock::now() + config.request_timeout;
      job.waiter = waiter;
      bytes_in_use += job.text.size();
      queue.push_back(std::move(job));
    }
    work_cv.notify_one();
  }

  void CancelJob(std::uint64_t request_id) {
    std::shared_ptr<Waiter> waiter;
    {
      std::lock_guard<std::mutex> lock(mu);
      for (auto it = queue.begin(); it != queue.end(); ++it) {
        if (it->request_id == request_id) {
          waiter = it->waiter;
          bytes_in_use -= it->text.size();
          queue.erase(it);
          break;
        }
      }
      if (waiter == nullptr) {
        for (Active& slot : active) {
          if (slot.waiter != nullptr && slot.request_id == request_id) {
            // Running: wake the caller now and discard the eventual result.
            // The encode runs to completion (native calls are not
            // interruptible) and the worker stays usable.
            slot.discard = true;
            waiter = slot.waiter;
            break;
          }
        }
      }
      if (waiter == nullptr) return;
      if (DrainDone()) work_cv.notify_all();
    }
    waiter->Complete(absl::CancelledError("tokenizer request cancelled"));
  }

  /// One repeating timer expiring jobs whose budget elapsed, queued or
  /// running; 100 ms granularity against seconds-scale budgets.
  void Sweep(const boost::system::error_code& ec) {
    if (ec) return;
    std::vector<std::shared_ptr<Waiter>> expired;
    bool drain_done = false;
    {
      std::lock_guard<std::mutex> lock(mu);
      const auto now = clock::now();
      for (auto it = queue.begin(); it != queue.end();) {
        if (now >= it->deadline) {
          bytes_in_use -= it->text.size();
          expired.push_back(it->waiter);
          it = queue.erase(it);
        } else {
          ++it;
        }
      }
      for (Active& slot : active) {
        if (slot.waiter != nullptr && !slot.discard && now >= slot.deadline) {
          slot.discard = true;
          expired.push_back(slot.waiter);
        }
      }
      drain_done = DrainDone();
      if (!drain_done && !stopped) {
        sweep_timer.expires_after(kSweepInterval);
        sweep_timer.async_wait(
            [self = shared_from_this()](const boost::system::error_code& again) {
              self->Sweep(again);
            });
      }
    }
    for (const auto& waiter : expired) {
      waiter->Complete(absl::DeadlineExceededError("tokenizer request timed out"));
    }
    if (drain_done) work_cv.notify_all();
  }

  void BeginDrain() {
    {
      std::lock_guard<std::mutex> lock(mu);
      draining = true;
    }
    work_cv.notify_all();
  }

  void ForceStop() {
    std::vector<std::shared_ptr<Waiter>> abandoned;
    {
      std::lock_guard<std::mutex> lock(mu);
      stopped = true;
      for (const Job& job : queue) {
        bytes_in_use -= job.text.size();
        abandoned.push_back(job.waiter);
      }
      queue.clear();
      for (Active& slot : active) {
        if (slot.waiter != nullptr && !slot.discard) {
          slot.discard = true;
          abandoned.push_back(slot.waiter);
        }
      }
    }
    work_cv.notify_all();
    for (const auto& waiter : abandoned) {
      waiter->Complete(absl::CancelledError("tokenizer pool stopped"));
    }
  }

  Status WaitUntilReady(std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(ready_mu);
    const bool done = ready_cv.wait_for(
        lock, timeout, [&] { return ready_count + failed_count >= config.workers; });
    if (!done) {
      return FailedPreconditionError("tokenizer workers did not become ready within ",
                                     std::to_string(timeout.count()), "ms");
    }
    if (failed_count.load() > 0) {
      return FailedPreconditionError("tokenizer workers failed to load '",
                                     config.tokenizer_path, "'");
    }
    return OkStatus();
  }

  PoolHealth Health() const {
    PoolHealth health;
    health.workers_total = config.workers;
    health.workers_ready = std::min(ready_count.load(), config.workers);
    health.workers_down = failed_count.load();
    {
      std::lock_guard<std::mutex> lock(mu);
      health.draining = draining;
      health.unready = !draining && failed_count.load() >= config.workers;
      health.workers_busy = static_cast<int>(BusyCountLocked());
      health.queued_requests = queue.size();
      health.queued_bytes = bytes_in_use;
    }
    return health;
  }

 private:
  std::size_t BusyCountLocked() const {
    std::size_t busy = 0;
    for (const Active& slot : active) {
      if (slot.waiter != nullptr) ++busy;
    }
    return busy;
  }
};

TokenizerPool::TokenizerPool(net::io_context& io, TokenizerPoolConfig config)
    : impl_(std::make_shared<Impl>(io, std::move(config))) {
  impl_->StartWorkers();
}

TokenizerPool::~TokenizerPool() {
  impl_->ForceStop();
  for (std::thread& thread : impl_->threads) thread.join();
}

net::awaitable<StatusOr<PreparedPrompt>> TokenizerPool::Prepare(
    std::uint64_t request_id, std::string text) {
  auto waiter = std::make_shared<Waiter>(impl_->io.get_executor());
  // If the awaiting coroutine is destroyed (client disconnect) the guard
  // abandons the request; after normal completion Cancel is a no-op.
  struct Abandon {
    std::shared_ptr<Impl> impl;
    std::uint64_t request_id;
    ~Abandon() { impl->CancelJob(request_id); }
  } abandon{impl_, request_id};
  impl_->Enqueue(request_id, std::move(text), waiter);
  co_return co_await waiter->Take();
}

void TokenizerPool::Cancel(std::uint64_t request_id) { impl_->CancelJob(request_id); }

void TokenizerPool::BeginDrain() { impl_->BeginDrain(); }

net::awaitable<Status> TokenizerPool::JoinUntil(clock::time_point deadline) {
  net::steady_timer timer(co_await net::this_coro::executor);
  for (;;) {
    if (impl_->drained_signal.Fired()) co_return OkStatus();
    if (clock::now() >= deadline) {
      co_return absl::DeadlineExceededError("tokenizer pool drain deadline exceeded");
    }
    timer.expires_at(std::min(deadline, clock::now() + std::chrono::milliseconds(25)));
    co_await timer.async_wait(net::as_tuple(net::use_awaitable));
  }
}

void TokenizerPool::ForceStop() { impl_->ForceStop(); }

Status TokenizerPool::WaitUntilReady(std::chrono::milliseconds timeout) {
  return impl_->WaitUntilReady(timeout);
}

PoolHealth TokenizerPool::Health() const { return impl_->Health(); }

}  // namespace inferx::server
