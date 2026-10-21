// Incremental detokenizer (docs/tokenizer_design.md §8.2). One decode
// worker thread owns the tokenizer instance and all per-request decode
// state; Append/Finish jobs cross from the io side through a bounded deque
// and results cross back through per-request ResultMailboxes bound to the
// awaiting coroutine's executor.
//
// Incremental algorithm: each request keeps its full token list, the latest
// full decode, and a byte offset into it. Every Append re-decodes the list
// (prefix-stable for the backends in use) and returns only the suffix past
// the emitted offset, held back at UTF-8 boundaries; Finish flushes the
// tail. The SGLang-style surrounding window — decoding only the last ~64
// tokens per step — is deliberately deferred: cleanup-style decoders (the
// test fixture joins tokens with spaces) are not suffix-consistent across a
// window slide, so bounding the decode needs per-request token/text
// boundary tracking that a generic Tokenizer cannot provide. The interface
// and state layout keep offsets so a backend-native incremental decoder can
// slot in later without touching callers.
#include "inferx/server/detokenizer.h"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "inferx/server/completion_signal.h"
#include "inferx/server/result_mailbox.h"
#include "inferx/server/utf8.h"

namespace inferx {
namespace {

namespace net = boost::asio;
using DecodeClock = std::chrono::steady_clock;

using DecodeMailbox = ResultMailbox<DecodeDelta>;

/// Pending decode jobs before admission fails; callers serialize
/// Append/Finish per request, so this bounds whole-flight concurrency.
constexpr std::size_t kMaxPendingJobs = 1024;

/// Per-request incremental decode state. Guarded by the detokenizer mutex
/// like the queue; the decode runs under the same lock, so Cancel never
/// races a state update.
struct DecodeState {
  std::vector<TokenId> tokens;
  std::string decode;  ///< Latest full decode of `tokens`.
  std::size_t emitted = 0;  ///< Bytes of `decode` already returned.
};

struct DecodeJob {
  RequestId request_id = 0;
  std::vector<TokenId> tokens;  // Empty with finish=false is a no-op append.
  bool finish = false;
  DecodeOptions options;
  std::shared_ptr<DecodeMailbox> waiter;
};

class WorkerDetokenizer : public Detokenizer,
                          public std::enable_shared_from_this<WorkerDetokenizer> {
 public:
  /// A null tokenizer yields an unready detokenizer: Append/Finish return
  /// Unavailable rather than serving degraded output.
  explicit WorkerDetokenizer(std::shared_ptr<Tokenizer> tokenizer)
      : tokenizer_(std::move(tokenizer)) {}

  ~WorkerDetokenizer() override {
    ForceStop();
    if (worker_.joinable()) worker_.join();
  }

  void Start() { worker_ = std::thread([this] { WorkerLoop(); }); }

  net::awaitable<StatusOr<DecodeDelta>> Append(RequestId request_id,
                                               std::vector<TokenId> tokens,
                                               const DecodeOptions& options) override {
    // The mailbox binds to the awaiting coroutine's executor, so no
    // io_context is captured here and completion resumes the caller.
    auto waiter = std::make_shared<DecodeMailbox>(co_await net::this_coro::executor);
    const Status admitted = Enqueue(request_id, std::move(tokens), /*finish=*/false,
                                    options, waiter);
    if (!admitted.ok()) co_return admitted;
    co_return co_await waiter->Take();
  }

  net::awaitable<StatusOr<DecodeDelta>> Finish(RequestId request_id) override {
    auto waiter = std::make_shared<DecodeMailbox>(co_await net::this_coro::executor);
    const Status admitted =
        Enqueue(request_id, {}, /*finish=*/true, DecodeOptions{}, waiter);
    if (!admitted.ok()) co_return admitted;
    co_return co_await waiter->Take();
  }

  void Cancel(RequestId request_id) override {
    std::lock_guard<std::mutex> lock(mu_);
    states_.erase(request_id);
  }

  void BeginDrain() override {
    {
      std::lock_guard<std::mutex> lock(mu_);
      draining_ = true;
    }
    work_cv_.notify_all();
  }

  net::awaitable<Status> JoinUntil(DecodeClock::time_point deadline) override {
    net::steady_timer timer(co_await net::this_coro::executor);
    for (;;) {
      if (drained_signal_.Fired()) co_return OkStatus();
      if (DecodeClock::now() >= deadline) {
        co_return absl::DeadlineExceededError("detokenizer drain deadline exceeded");
      }
      timer.expires_at(std::min(deadline, DecodeClock::now() + std::chrono::milliseconds(25)));
      co_await timer.async_wait(net::as_tuple(net::use_awaitable));
    }
  }

 private:
  Status Enqueue(RequestId request_id, std::vector<TokenId> tokens, bool finish,
                 const DecodeOptions& options,
                 const std::shared_ptr<DecodeMailbox>& waiter) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopped_) return FailedPreconditionError("detokenizer is shutting down");
      if (draining_) return FailedPreconditionError("detokenizer is draining");
      if (tokenizer_ == nullptr) {
        return absl::UnavailableError("detokenizer has no usable tokenizer");
      }
      if (queue_.size() >= kMaxPendingJobs) {
        return ResourceExhaustedError("detokenizer queue is full");
      }
      DecodeJob job;
      job.request_id = request_id;
      job.tokens = std::move(tokens);
      job.finish = finish;
      job.options = options;
      job.waiter = waiter;
      queue_.push_back(std::move(job));
    }
    work_cv_.notify_one();
    return OkStatus();
  }

  void WorkerLoop() {
    for (;;) {
      DecodeJob job;
      {
        std::unique_lock<std::mutex> lock(mu_);
        work_cv_.wait(lock, [&] { return stopped_ || !queue_.empty() || draining_; });
        if (stopped_) break;
        if (queue_.empty()) {
          if (draining_) break;
          continue;
        }
        job = std::move(queue_.front());
        queue_.pop_front();
      }
      {
        // State updates and the bounded window decode run under mu_ so
        // Cancel (which erases state) can never race an update.
        std::lock_guard<std::mutex> lock(mu_);
        job.waiter->Complete(job.finish ? FinishState(job.request_id)
                                        : AppendState(job.request_id, job.tokens,
                                                      job.options));
      }
    }
    // Fail whatever admission let in while the drain decision raced it.
    std::vector<std::shared_ptr<DecodeMailbox>> abandoned;
    {
      std::lock_guard<std::mutex> lock(mu_);
      abandoned.reserve(queue_.size());
      for (const DecodeJob& job : queue_) abandoned.push_back(job.waiter);
      queue_.clear();
    }
    for (const auto& waiter : abandoned) {
      waiter->Complete(absl::CancelledError("detokenizer stopped"));
    }
    drained_signal_.Fire();
  }

  /// Appends tokens and returns the newly settled suffix of the decode.
  /// Utf8Boundary keeps a trailing partial character for a later step or
  /// Finish; everything else settles immediately.
  StatusOr<DecodeDelta> AppendState(RequestId request_id, const std::vector<TokenId>& tokens,
                                    const DecodeOptions& options) {
    DecodeState& state = states_[request_id];
    state.tokens.insert(state.tokens.end(), tokens.begin(), tokens.end());
    if (state.tokens.empty()) return DecodeDelta{};

    auto text = DecodeTokens(state.tokens, options);
    if (!text.ok()) return text.status();
    // Decode cannot shrink the settled prefix; refuse to substr() past the
    // end if some backend ever violates that assumption.
    if (text->size() < state.emitted) {
      return InternalError("detokenizer decode shrank below the emitted offset");
    }
    state.decode = std::move(*text);

    const std::size_t boundary = detail::Utf8Boundary(state.decode);
    if (boundary <= state.emitted) return DecodeDelta{};
    DecodeDelta delta;
    delta.text = state.decode.substr(state.emitted, boundary - state.emitted);
    state.emitted = boundary;
    return delta;
  }

  /// Flushes held-back bytes (a trailing partial UTF-8 character, if any)
  /// and releases the state. Finish of an unknown id is an empty delta.
  StatusOr<DecodeDelta> FinishState(RequestId request_id) {
    const auto it = states_.find(request_id);
    if (it == states_.end()) return DecodeDelta{};
    DecodeDelta delta;
    if (it->second.emitted < it->second.decode.size()) {
      delta.text = it->second.decode.substr(it->second.emitted);
    }
    states_.erase(it);
    return delta;
  }

  StatusOr<std::string> DecodeTokens(const std::vector<TokenId>& tokens,
                                     const DecodeOptions& options) {
    std::vector<int> ids(tokens.begin(), tokens.end());
    auto text = tokenizer_->Decode(ids, options);
    if (!text.ok()) return text.status();
    return std::move(*text);
  }

  void ForceStop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopped_ = true;
    }
    work_cv_.notify_all();
  }

  std::shared_ptr<Tokenizer> tokenizer_;

  // Guards the queue, decode states, and lifecycle flags. The bounded
  // window decode runs under this mutex; handoffs never block on it.
  mutable std::mutex mu_;
  std::condition_variable work_cv_;
  std::deque<DecodeJob> queue_;
  std::unordered_map<RequestId, DecodeState> states_;
  bool draining_ = false;
  bool stopped_ = false;

  std::thread worker_;
  CompletionSignal drained_signal_;
};

}  // namespace

std::shared_ptr<Detokenizer> CreateDetokenizer(const std::string& tokenizer_path) {
  auto tokenizer = Tokenizer::FromFile(tokenizer_path);
  // A failed load keeps the object constructible: every Append/Finish
  // reports Unavailable, matching the pool's unready semantics.
  return CreateDetokenizer(tokenizer.ok() ? std::move(*tokenizer) : nullptr);
}

std::shared_ptr<Detokenizer> CreateDetokenizer(
    std::shared_ptr<Tokenizer> exclusive_tokenizer) {
  auto detokenizer = std::make_shared<WorkerDetokenizer>(std::move(exclusive_tokenizer));
  detokenizer->Start();
  return detokenizer;
}

}  // namespace inferx
