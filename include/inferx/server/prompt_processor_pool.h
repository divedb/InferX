/// \brief Bounded prompt-processor pool interface
/// (docs/tokenizer_design.md §8.2).
///
/// N worker threads each own one tokenizer/prompt-processor instance; jobs
/// cross from the io side through a bounded queue (requests and bytes) and
/// results cross back through per-request mailboxes. The pool never touches
/// request registry state and never runs tokenizer code on an I/O thread;
/// there is no inline fallback — queue-full, unready, and draining pools
/// return errors the HTTP layer maps to 503/400.
#ifndef INFERX_SERVER_PROMPT_PROCESSOR_POOL_H_
#define INFERX_SERVER_PROMPT_PROCESSOR_POOL_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <boost/asio.hpp>

#include "inferx/core/status.h"
#include "inferx/server/prompt_processor.h"
#include "inferx/server/request_model.h"

namespace inferx {

/// \brief Tuning knobs of the prompt pool.
struct PromptPoolConfig {
  /// tokenizer.json path every worker loads its own instance from.
  std::string tokenizer_path;

  /// Extra files that change encoding output (tokenizer_config.json,
  /// special_tokens_map.json, ...). Hashed into the fingerprint with
  /// `tokenizer_path`; missing files are skipped.
  std::vector<std::string> fingerprint_paths;

  int workers = 2;  ///< Worker thread count. >=1; there is no inline fallback.
  std::size_t queue_max_requests = 128;  ///< Pending + running jobs.

  std::size_t queue_max_bytes = static_cast<std::size_t>(32)
                                << 20;  ///< Pending + running input bytes.

  /// Largest prompt accepted by Prepare, checked before admission so one
  /// oversized prompt cannot occupy a worker. Must be <= queue_max_bytes.
  std::size_t max_prompt_bytes = static_cast<std::size_t>(8) << 20;

  std::chrono::milliseconds startup_timeout{30000};  ///< Parallel asset load.

  /// Default per-request budget (queueing included) when the PrepareContext
  /// carries no deadline; also bounds a running encode, whose late result is
  /// then discarded, never delivered.
  std::chrono::milliseconds request_timeout{10000};

  std::chrono::milliseconds shutdown_grace{10000};

  /// Checks ranges and cross-field constraints.
  Status Validate() const;
};

/// \brief Monotonic counters since pool construction (metrics export).
struct PoolMetrics {
  std::uint64_t submitted = 0;                ///< Jobs admitted to the queue.
  std::uint64_t completed = 0;                ///< Jobs a worker finished running.
  std::uint64_t rejected_full = 0;            ///< Queue or byte budget exceeded.
  std::uint64_t rejected_oversize = 0;        ///< Input larger than the limits.
  std::uint64_t timed_out = 0;                ///< Deadline hit before delivery.
  std::uint64_t cancelled = 0;                ///< Abandoned by the caller or stop.
  std::uint64_t queue_wait_micros_total = 0;  ///< Sum of time spent queued.
  std::uint64_t run_micros_total = 0;         ///< Sum of worker run time.
};

/// \brief Point-in-time pool state (for /health and tests).
struct PoolHealth {
  int workers_total = 0;
  int workers_ready = 0;
  int workers_busy = 0;
  int workers_down = 0;  ///< Failed to load the asset; permanent.
  bool draining = false;
  bool unready = false;  ///< Every worker down while not draining.
  std::size_t queued_requests = 0;
  std::size_t queued_bytes = 0;
  PoolMetrics metrics;
};

class PromptProcessorPool {
 public:
  virtual ~PromptProcessorPool() = default;

  /// \brief Prepares a prompt on a worker thread. The returned Task is lazy
  /// and must be started with an executor. The budget is the context
  /// deadline (or config.request_timeout when unset) from the moment the
  /// Task starts, queueing included. Cancellation is via Cancel(id); the
  /// awaiting coroutine's destruction also abandons the job, and a late
  /// worker result is discarded.
  virtual boost::asio::awaitable<StatusOr<PreparedPrompt>> Prepare(
      RequestId request_id, PromptInput input, const PrepareContext& context) = 0;

  /// \brief Abandons a request: queued work completes with Cancelled
  /// immediately; a running encode is marked so its late result is
  /// discarded. Idempotent; unknown ids are no-ops.
  virtual void Cancel(RequestId request_id) = 0;

  /// \brief Stops admitting new work; queued jobs drain through the
  /// workers, then they exit. Call JoinUntil to observe completion.
  virtual void BeginDrain() = 0;

  /// \brief Completes once every worker thread has exited (drained or
  /// forced), or the caller deadline passes.
  virtual boost::asio::awaitable<Status> JoinUntil(
      std::chrono::steady_clock::time_point deadline) = 0;

  /// \brief Blocks the calling thread until every worker loaded its
  /// tokenizer or failed. Startup gate; not for executor threads.
  virtual Status WaitUntilReady(std::chrono::milliseconds timeout) = 0;

  virtual PoolHealth Health() const = 0;
};

/// \brief Creates the production pool: worker threads, one tokenizer
/// instance each, bounded admission, deadline sweep, cooperative drain.
/// The io_context must outlive the returned pool.
std::shared_ptr<PromptProcessorPool> CreatePromptProcessorPool(
    boost::asio::io_context& io, PromptPoolConfig config);

}  // namespace inferx

#endif  // INFERX_SERVER_PROMPT_PROCESSOR_POOL_H_
