// Bounded prompt-processor pool (docs/tokenizer_design.md §8.2). N worker
// threads each own one tokenizer instance wrapped in a TextPromptProcessor;
// jobs cross from the io side to workers through a mutex-guarded bounded
// deque (workers never hold the mutex during encode), and results cross
// back through per-request ResultMailboxes. One repeating timer on the
// io_context sweeps job deadlines for both queued and running jobs.
#include "inferx/server/prompt_processor_pool.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "inferx/server/chat_template.h"
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include "inferx/core/logging.h"
#include "inferx/server/completion_signal.h"
#include "inferx/server/result_mailbox.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {
namespace {

namespace net = boost::asio;
using PoolClock = std::chrono::steady_clock;
constexpr auto kSweepInterval = std::chrono::milliseconds(100);

using PrepareMailbox = ResultMailbox<PreparedPrompt>;

/// FNV-1a over one file's bytes; 0 when the file cannot be read.
std::uint64_t FingerprintFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return 0;
  std::string blob((std::istreambuf_iterator<char>(input)),
                   std::istreambuf_iterator<char>());
  std::uint64_t hash = 1469598103934665603ull;
  for (const char c : blob) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ull;
  }
  return hash;
}

/// Fingerprint of the tokenizer asset: the merge of tokenizer_path and the
/// optional extra files, reported in every PreparedPrompt so callers can
/// observe which revision encoded their prompt.
std::uint64_t FingerprintAsset(const PromptPoolConfig& config) {
  std::uint64_t hash = FingerprintFile(config.tokenizer_path);
  hash = hash * 1099511628211ull ^ FingerprintFile(
      (std::filesystem::path(config.tokenizer_path).parent_path() / "tokenizer_config.json").string());
  for (const std::string& path : config.fingerprint_paths) {
    hash = hash * 1099511628211ull ^ FingerprintFile(path);
  }
  return hash;
}

/// Admission-size accounting for a PromptInput, in bytes: text and media
/// sizes directly, token ids at sizeof(TokenId) per token.
std::size_t PromptInputBytes(const PromptInput& input) {
  if (const auto* text = std::get_if<TextPrompt>(&input)) {
    return text->text.size();
  }
  if (const auto* ids = std::get_if<TokenIdsPrompt>(&input)) {
    return ids->ids.size() * sizeof(TokenId);
  }
  if (const auto* multimodal = std::get_if<MultimodalInput>(&input)) {
    std::size_t bytes = 0;
    for (const ContentPart& part : multimodal->parts) {
      if (const auto* text = std::get_if<TextPart>(&part)) {
        bytes += text->text.size();
      } else if (const auto* media = std::get_if<MediaPart>(&part)) {
        bytes += media->media.data ? media->media.data->bytes.size() : 0;
      }
    }
    return bytes;
  }
  const auto& chat = std::get<ChatPrompt>(input);
  std::size_t bytes = 0;
  for (const ChatMessage& message : chat.messages) {
    for (const ContentPart& part : message.content) {
      if (const auto* text = std::get_if<TextPart>(&part)) {
        bytes += text->text.size();
      }
    }
  }
  return bytes;
}

/// \brief Text/token prompt processor: one per worker, over that worker's
/// exclusive tokenizer instance. Recognized Qwen3 text chat is rendered
/// here; unrecognized templates and multimodal input return explicit errors.
class TextPromptProcessor : public PromptProcessor {
 public:
  TextPromptProcessor(std::shared_ptr<Tokenizer> tokenizer, std::uint64_t fingerprint, bool chat_supported)
      : tokenizer_(std::move(tokenizer)), fingerprint_(fingerprint), chat_supported_(chat_supported) {}

  StatusOr<PreparedPrompt> Prepare(const PromptInput& input,
                                   const PrepareContext& context) override {
    PreparedPrompt prepared;
    prepared.tokenizer_fingerprint = fingerprint_;
    if (const auto* text = std::get_if<TextPrompt>(&input)) {
      // Byte-size admission is the pool's job (config.max_prompt_bytes,
      // checked before queueing); the processor owns token/content checks.
      auto ids = tokenizer_->Encode(text->text);
      if (!ids.ok()) {
        // Deterministic input/asset failure: the same text fails again, so
        // it surfaces as a 400, never retried.
        return InvalidArgumentError("failed to tokenize prompt: ", ids.status().message());
      }
      prepared.token_ids = std::move(*ids);
    } else if (const auto* prompt_ids = std::get_if<TokenIdsPrompt>(&input)) {
      if (prompt_ids->ids.empty()) {
        return InvalidArgumentError("token prompt is empty");
      }
      const int64_t vocab = tokenizer_->vocab_size();
      for (const TokenId id : prompt_ids->ids) {
        if (id < 0 || static_cast<int64_t>(id) >= vocab) {
          return InvalidArgumentError("token id ", id, " is outside the vocabulary [0, ",
                                      vocab, ")");
        }
      }
      prepared.token_ids = prompt_ids->ids;
    } else if (const auto* chat = std::get_if<ChatPrompt>(&input)) {
      if (!chat_supported_) return UnimplementedError("chat_template_not_supported: unrecognized or missing checkpoint template");
      auto rendered = RenderQwen3Chat(*chat);
      if (!rendered.ok()) return rendered.status();
      if (rendered->size() > context.limits.max_prompt_bytes)
        return InvalidArgumentError("rendered chat exceeds prompt byte limit");
      auto ids = tokenizer_->Encode(*rendered);
      if (!ids.ok()) return ids.status();
      prepared.token_ids = std::move(*ids);
    } else {
      return UnimplementedError("multimodal prompts are not supported yet");
    }

    prepared.prompt_tokens = static_cast<int>(prepared.token_ids.size());
    if (context.limits.max_prompt_tokens > 0 &&
        static_cast<std::size_t>(prepared.prompt_tokens) > context.limits.max_prompt_tokens) {
      return InvalidArgumentError("prompt has ", prepared.prompt_tokens,
                                  " tokens; the limit is ", context.limits.max_prompt_tokens);
    }
    return prepared;
  }

 private:
  std::shared_ptr<Tokenizer> tokenizer_;
  std::uint64_t fingerprint_;
  bool chat_supported_ = false;
};

struct PoolJob {
  RequestId request_id = 0;
  PromptInput input;
  PrepareContext context;
  std::size_t bytes = 0;
  PoolClock::time_point deadline{};
  PoolClock::time_point enqueued_at{};
  std::shared_ptr<PrepareMailbox> waiter;
};

/// What a worker is currently running, published under the pool mutex. The
/// deadline sweep and Cancel set `discard` so the eventual result is
/// dropped; the encode itself always runs to completion.
struct ActiveJob {
  RequestId request_id = 0;
  PoolClock::time_point deadline{};
  bool discard = false;
  std::shared_ptr<PrepareMailbox> waiter;  // Null when the worker is idle.
};

class WorkerPromptProcessorPool
    : public PromptProcessorPool,
      public std::enable_shared_from_this<WorkerPromptProcessorPool> {
 public:
  WorkerPromptProcessorPool(net::io_context& io, PromptPoolConfig config)
      : io_(io), config_(std::move(config)) {
    config_.workers = std::max(config_.workers, 1);
    config_.queue_max_requests = std::max<std::size_t>(config_.queue_max_requests, 1);
    config_.queue_max_bytes = std::max<std::size_t>(config_.queue_max_bytes, 1);
    fingerprint_ = FingerprintAsset(config_);
  }

  ~WorkerPromptProcessorPool() override {
    ForceStop();
    for (std::thread& thread : threads_) thread.join();
  }

  void StartWorkers() {
    const std::size_t count = static_cast<std::size_t>(config_.workers);
    active_.assign(count, ActiveJob{});
    for (std::size_t i = 0; i < count; ++i) {
      threads_.emplace_back([this, i] { WorkerLoop(i); });
    }
    sweep_timer_.async_wait(
        [self = shared_from_this()](const boost::system::error_code& ec) { self->Sweep(ec); });
  }

  net::awaitable<StatusOr<PreparedPrompt>> Prepare(RequestId request_id, PromptInput input,
                                                   const PrepareContext& context) override {
    auto waiter = std::make_shared<PrepareMailbox>(io_.get_executor());
    // If the awaiting coroutine is destroyed (client disconnect) the guard
    // abandons the request; after normal completion Cancel is a no-op.
    struct Abandon {
      std::shared_ptr<WorkerPromptProcessorPool> pool;
      RequestId request_id;
      ~Abandon() { pool->CancelJob(request_id); }
    };
    Abandon abandon{shared_from_this(), request_id};

    Enqueue(request_id, std::move(input), context, waiter);
    co_return co_await waiter->Take();
  }

  void Cancel(RequestId request_id) override { CancelJob(request_id); }

  void BeginDrain() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      draining_ = true;
    }
    work_cv_.notify_all();
  }

  net::awaitable<Status> JoinUntil(PoolClock::time_point deadline) override {
    net::steady_timer timer(co_await net::this_coro::executor);
    for (;;) {
      if (drained_signal_.Fired()) co_return OkStatus();
      if (PoolClock::now() >= deadline) {
        co_return absl::DeadlineExceededError("prompt pool drain deadline exceeded");
      }
      timer.expires_at(std::min(deadline, PoolClock::now() + std::chrono::milliseconds(25)));
      co_await timer.async_wait(net::as_tuple(net::use_awaitable));
    }
  }

  Status WaitUntilReady(std::chrono::milliseconds timeout) override {
    std::unique_lock<std::mutex> lock(ready_mu_);
    const bool done = ready_cv_.wait_for(
        lock, timeout, [&] { return ready_count_ + failed_count_ >= config_.workers; });
    if (!done) {
      return FailedPreconditionError("prompt pool workers did not become ready within ",
                                     std::to_string(timeout.count()), "ms");
    }
    if (failed_count_.load() > 0) {
      return FailedPreconditionError("prompt pool workers failed to load '",
                                     config_.tokenizer_path, "'");
    }
    return OkStatus();
  }

  PoolHealth Health() const override {
    PoolHealth health;
    health.workers_total = config_.workers;
    health.workers_ready = std::min(ready_count_.load(), config_.workers);
    health.workers_down = failed_count_.load();
    {
      std::lock_guard<std::mutex> lock(mu_);
      health.draining = draining_;
      health.unready = !draining_ && failed_count_.load() >= config_.workers;
      health.workers_busy = static_cast<int>(BusyCountLocked());
      health.queued_requests = queue_.size();
      health.queued_bytes = bytes_in_use_;
    }
    health.metrics.submitted = metrics_.submitted.load();
    health.metrics.completed = metrics_.completed.load();
    health.metrics.rejected_full = metrics_.rejected_full.load();
    health.metrics.rejected_oversize = metrics_.rejected_oversize.load();
    health.metrics.timed_out = metrics_.timed_out.load();
    health.metrics.cancelled = metrics_.cancelled.load();
    health.metrics.queue_wait_micros_total = metrics_.queue_wait_micros_total.load();
    health.metrics.run_micros_total = metrics_.run_micros_total.load();
    return health;
  }

 private:
  void Enqueue(RequestId request_id, PromptInput input, const PrepareContext& context,
               const std::shared_ptr<PrepareMailbox>& waiter) {
    const std::size_t bytes = PromptInputBytes(input);
    const PoolClock::time_point deadline =
        context.deadline.time_since_epoch().count() > 0
            ? context.deadline
            : PoolClock::now() + config_.request_timeout;
    PrepareContext effective = context;
    effective.request_id = request_id;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (draining_ || stopped_) {
        waiter->Complete(FailedPreconditionError("prompt pool is shutting down"));
        return;
      }
      if (failed_count_.load() >= config_.workers) {
        waiter->Complete(
            absl::UnavailableError("prompt pool has no available workers"));
        return;
      }
      if (bytes > config_.max_prompt_bytes || bytes > config_.queue_max_bytes) {
        // A single prompt larger than the whole byte budget can never be
        // admitted; that is an input problem, not congestion.
        metrics_.rejected_oversize.fetch_add(1);
        waiter->Complete(InvalidArgumentError("prompt too large: ", bytes, " bytes"));
        return;
      }
      if (queue_.size() + BusyCountLocked() + 1 > config_.queue_max_requests ||
          bytes_in_use_ + bytes > config_.queue_max_bytes) {
        metrics_.rejected_full.fetch_add(1);
        waiter->Complete(ResourceExhaustedError(
            "prompt queue is full; retry after pending requests complete"));
        return;
      }
      PoolJob job;
      job.request_id = request_id;
      job.input = std::move(input);
      job.context = effective;
      job.bytes = bytes;
      job.enqueued_at = PoolClock::now();
      job.deadline = deadline;
      job.waiter = waiter;
      bytes_in_use_ += job.bytes;
      metrics_.submitted.fetch_add(1);
      queue_.push_back(std::move(job));
    }
    work_cv_.notify_one();
  }

  void CancelJob(RequestId request_id) {
    std::shared_ptr<PrepareMailbox> waiter;
    {
      std::lock_guard<std::mutex> lock(mu_);
      for (auto it = queue_.begin(); it != queue_.end(); ++it) {
        if (it->request_id == request_id) {
          waiter = it->waiter;
          bytes_in_use_ -= it->bytes;
          queue_.erase(it);
          break;
        }
      }
      if (waiter == nullptr) {
        for (ActiveJob& slot : active_) {
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
      if (DrainDoneLocked()) work_cv_.notify_all();
    }
    if (waiter != nullptr) {
      metrics_.cancelled.fetch_add(1);
      waiter->Complete(absl::CancelledError("prompt request cancelled"));
    }
  }

  /// One repeating timer expiring jobs whose budget elapsed, queued or
  /// running; 100 ms granularity against seconds-scale budgets.
  void Sweep(const boost::system::error_code& ec) {
    if (ec) return;
    std::vector<std::shared_ptr<PrepareMailbox>> expired;
    bool drain_done = false;
    {
      std::lock_guard<std::mutex> lock(mu_);
      const auto now = PoolClock::now();
      for (auto it = queue_.begin(); it != queue_.end();) {
        if (now >= it->deadline) {
          bytes_in_use_ -= it->bytes;
          expired.push_back(it->waiter);
          it = queue_.erase(it);
        } else {
          ++it;
        }
      }
      for (ActiveJob& slot : active_) {
        if (slot.waiter != nullptr && !slot.discard && now >= slot.deadline) {
          slot.discard = true;
          expired.push_back(slot.waiter);
        }
      }
      drain_done = DrainDoneLocked();
      if (!drain_done && !stopped_) {
        sweep_timer_.expires_after(kSweepInterval);
        sweep_timer_.async_wait(
            [self = shared_from_this()](const boost::system::error_code& again) {
              self->Sweep(again);
            });
      }
    }
    for (const auto& waiter : expired) {
      metrics_.timed_out.fetch_add(1);
      waiter->Complete(absl::DeadlineExceededError("prompt prepare timed out"));
    }
    if (drain_done) work_cv_.notify_all();
  }

  void WorkerLoop(std::size_t index) {
    std::shared_ptr<Tokenizer> tokenizer = LoadAndReportReady();
    if (tokenizer != nullptr) {
      bool chat_supported = false;
      try {
        std::ifstream metadata(std::filesystem::path(config_.tokenizer_path).parent_path() / "tokenizer_config.json");
        if (metadata) {
          auto json = nlohmann::json::parse(metadata);
          if (json.contains("chat_template") && json["chat_template"].is_string())
            chat_supported = IsSupportedChatTemplate(json["chat_template"].get<std::string>());
        }
      } catch (const std::exception&) {
        // Text completions remain available; chat reports an unsupported template.
      }
      TextPromptProcessor processor(tokenizer, fingerprint_, chat_supported);
      for (;;) {
        PoolJob job;
        {
          std::unique_lock<std::mutex> lock(mu_);
          work_cv_.wait(lock,
                        [&] { return stopped_ || !queue_.empty() || DrainDoneLocked(); });
          if (stopped_ || queue_.empty()) break;  // DrainDone implies empty.
          job = std::move(queue_.front());
          queue_.pop_front();
          active_[index] = ActiveJob{job.request_id, job.deadline, false, job.waiter};
          metrics_.queue_wait_micros_total.fetch_add(
              static_cast<std::uint64_t>(
                  std::chrono::duration_cast<std::chrono::microseconds>(PoolClock::now() -
                                                                        job.enqueued_at)
                      .count()));
        }
        RunJob(index, job, &processor);
      }
    }
    if (exited_count_.fetch_add(1) + 1 >= config_.workers) drained_signal_.Fire();
  }

  std::shared_ptr<Tokenizer> LoadAndReportReady() {
    auto loaded = Tokenizer::FromFile(config_.tokenizer_path);
    std::shared_ptr<Tokenizer> tokenizer;
    if (loaded.ok()) {
      tokenizer = std::move(*loaded);
      ready_count_.fetch_add(1);
    } else {
      INFERX_LOG(ERROR) << "inferx serve: prompt worker failed to load "
                        << config_.tokenizer_path << ": "
                        << std::string(loaded.status().message());
      failed_count_.fetch_add(1);
    }
    std::lock_guard<std::mutex> lock(ready_mu_);
    ready_cv_.notify_all();
    return tokenizer;
  }

  void RunJob(std::size_t index, const PoolJob& job, PromptProcessor* processor) {
    const auto run_started = PoolClock::now();
    StatusOr<PreparedPrompt> result = processor->Prepare(job.input, job.context);
    std::shared_ptr<PrepareMailbox> waiter;
    {
      std::lock_guard<std::mutex> lock(mu_);
      waiter = active_[index].waiter;
      const bool discard = active_[index].discard;
      active_[index] = ActiveJob{};
      bytes_in_use_ -= job.bytes;
      metrics_.completed.fetch_add(1);
      metrics_.run_micros_total.fetch_add(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(PoolClock::now() -
                                                                run_started)
              .count()));
      if (discard) {
        // Caller was already woken; this result is dead. Still notify: a
        // drain may be waiting on this last active slot.
        work_cv_.notify_all();
        return;
      }
    }
    // On discard the caller was already woken (cancel/timeout); Complete is
    // exactly-once, so simply delivering would also be a no-op then.
    waiter->Complete(std::move(result));
    work_cv_.notify_all();  // A drain waiting on the last active job.
  }

  void ForceStop() {
    std::vector<std::shared_ptr<PrepareMailbox>> abandoned;
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopped_ = true;
      for (const PoolJob& job : queue_) {
        bytes_in_use_ -= job.bytes;
        abandoned.push_back(job.waiter);
      }
      queue_.clear();
      for (ActiveJob& slot : active_) {
        if (slot.waiter != nullptr && !slot.discard) {
          slot.discard = true;
          abandoned.push_back(slot.waiter);
        }
      }
    }
    work_cv_.notify_all();
    for (const auto& waiter : abandoned) {
      metrics_.cancelled.fetch_add(1);
      waiter->Complete(absl::CancelledError("prompt pool stopped"));
    }
  }

  /// Draining with no admitted work left; `mu_` held.
  bool DrainDoneLocked() const {
    for (const ActiveJob& slot : active_) {
      if (slot.waiter != nullptr) return false;
    }
    return draining_ && queue_.empty();
  }

  std::size_t BusyCountLocked() const {
    std::size_t busy = 0;
    for (const ActiveJob& slot : active_) {
      if (slot.waiter != nullptr) ++busy;
    }
    return busy;
  }

  net::io_context& io_;
  PromptPoolConfig config_;
  std::uint64_t fingerprint_ = 0;

  // Guards queue, active slots, accounting, and lifecycle flags. Workers
  // hold it only to hand jobs across — never during encode.
  mutable std::mutex mu_;
  std::condition_variable work_cv_;
  std::deque<PoolJob> queue_;
  std::vector<ActiveJob> active_;  // One slot per worker, indexed by thread.
  std::size_t bytes_in_use_ = 0;
  bool draining_ = false;
  bool stopped_ = false;

  std::vector<std::thread> threads_;

  // Startup gate and exit latch. ready/failed are written under ready_mu_ by
  // the workers and read racily by admission/health, hence atomics.
  std::mutex ready_mu_;
  std::condition_variable ready_cv_;
  std::atomic<int> ready_count_{0};
  std::atomic<int> failed_count_{0};
  std::atomic<int> exited_count_{0};
  CompletionSignal drained_signal_;  // Every worker thread has exited.

  struct AtomicMetrics {
    std::atomic<std::uint64_t> submitted{0};
    std::atomic<std::uint64_t> completed{0};
    std::atomic<std::uint64_t> rejected_full{0};
    std::atomic<std::uint64_t> rejected_oversize{0};
    std::atomic<std::uint64_t> timed_out{0};
    std::atomic<std::uint64_t> cancelled{0};
    std::atomic<std::uint64_t> queue_wait_micros_total{0};
    std::atomic<std::uint64_t> run_micros_total{0};
  } metrics_;

  net::steady_timer sweep_timer_{io_};
};

}  // namespace

Status PromptPoolConfig::Validate() const {
  if (tokenizer_path.empty()) {
    return InvalidArgumentError("PromptPoolConfig: tokenizer_path must not be empty");
  }
  if (workers < 1) {
    return InvalidArgumentError("PromptPoolConfig: workers must be >= 1, got ", workers);
  }
  if (queue_max_requests < 1) {
    return InvalidArgumentError("PromptPoolConfig: queue_max_requests must be >= 1, got ",
                                queue_max_requests);
  }
  if (queue_max_bytes < 1) {
    return InvalidArgumentError("PromptPoolConfig: queue_max_bytes must be >= 1, got ",
                                queue_max_bytes);
  }
  if (max_prompt_bytes < 1 || max_prompt_bytes > queue_max_bytes) {
    return InvalidArgumentError(
        "PromptPoolConfig: max_prompt_bytes must be in [1, queue_max_bytes]");
  }
  for (const std::string& path : fingerprint_paths) {
    if (path.empty()) {
      return InvalidArgumentError(
          "PromptPoolConfig: fingerprint_paths must not contain empty entries");
    }
  }
  if (startup_timeout.count() <= 0) {
    return InvalidArgumentError("PromptPoolConfig: startup_timeout must be > 0");
  }
  if (request_timeout.count() <= 0) {
    return InvalidArgumentError("PromptPoolConfig: request_timeout must be > 0");
  }
  if (shutdown_grace.count() < 0) {
    return InvalidArgumentError("PromptPoolConfig: shutdown_grace must be >= 0");
  }
  return OkStatus();
}

std::shared_ptr<PromptProcessorPool> CreatePromptProcessorPool(
    net::io_context& io, PromptPoolConfig config) {
  auto pool = std::make_shared<WorkerPromptProcessorPool>(io, std::move(config));
  // StartWorkers casts shared_from_this(); the make_shared result is the
  // only owner so far, so this is race-free by construction.
  pool->StartWorkers();
  return pool;
}

}  // namespace inferx
