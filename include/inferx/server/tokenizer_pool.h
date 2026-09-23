// Bounded asynchronous tokenizer pool — stage 1 of
// docs/tokenizer_process_pool.md, implemented with dedicated worker threads
// instead of worker processes: every requirement of the stage (no tokenizer
// code on I/O threads, one FFI tokenizer instance per worker, bounded
// admission, cooperative drain) is met without a second executable, an IPC
// protocol, or process supervision — none of which the vendored
// third_party set provides utilities for. The process design in the doc
// remains the upgrade path if tokenizer crash isolation is ever required.
//
// Concurrency: pool state lives under one mutex (workers hold it only to
// hand jobs across, never during encode); the deadline sweep runs on the
// io_context; HTTP coroutines interact through Prepare/Cancel and Health.
// Workers never touch scheduler state or write HTTP responses; token ids
// remain the engine boundary.
#ifndef INFERX_SERVER_TOKENIZER_POOL_H_
#define INFERX_SERVER_TOKENIZER_POOL_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <boost/asio.hpp>

#include "inferx/core/status.h"

namespace inferx::server {

/// \brief Tuning knobs; defaults follow the doc's "Configuration and
/// validation" table minus the process-only restart/batching policy.
struct TokenizerPoolConfig {
  /// \brief tokenizer.json path every worker loads its own instance from.
  std::string tokenizer_path;
  int workers = 2;  ///< Worker thread count. >=1; there is no inline fallback.
  std::size_t queue_max_requests = 128;  ///< Pending + running prepare jobs.
  std::size_t queue_max_bytes =
      static_cast<std::size_t>(32) << 20;  ///< Pending input bytes.
  std::chrono::milliseconds startup_timeout{30000};  ///< Parallel asset load.
  /// \brief Per-request budget including queueing; also bounds a running
  /// encode (whose late result is then discarded, never delivered).
  std::chrono::milliseconds request_timeout{10000};
  std::chrono::milliseconds shutdown_grace{10000};
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
};

/// \brief Successful Prepare result.
struct PreparedPrompt {
  std::uint64_t request_id = 0;
  std::vector<int> token_ids;
  std::uint64_t fingerprint = 0;  ///< FNV-1a of the tokenizer asset.
};

class TokenizerPool {
 public:
  /// \brief Starts the worker threads; each loads its own tokenizer and
  /// reports readiness asynchronously. Call WaitUntilReady before serving.
  explicit TokenizerPool(boost::asio::io_context& io, TokenizerPoolConfig config);
  ~TokenizerPool();

  TokenizerPool(const TokenizerPool&) = delete;
  TokenizerPool& operator=(const TokenizerPool&) = delete;

  /// \brief Encodes prompt text on a worker thread. The caller owns the
  /// request id (unique among live calls); the budget is
  /// config.request_timeout from entry, queueing included. Never falls back
  /// to encoding inline: queue-full, unready, and drained pools return
  /// errors the HTTP layer maps to 503/400.
  boost::asio::awaitable<StatusOr<PreparedPrompt>> Prepare(std::uint64_t request_id,
                                                           std::string text);

  /// \brief Abandons a request. Queued work completes with Cancelled
  /// immediately; a running encode is marked so its late result is
  /// discarded. Idempotent.
  void Cancel(std::uint64_t request_id);

  /// \brief Stops admitting new work; queued jobs drain through the
  /// workers, then they exit. Call JoinUntil to observe completion.
  void BeginDrain();

  /// \brief Completes once every worker thread has exited (drained or
  /// forced), or the caller deadline passes.
  boost::asio::awaitable<Status> JoinUntil(
      std::chrono::steady_clock::time_point deadline);

  /// \brief Fails every waiter and stops the workers after their current
  /// encode; the shutdown path of last resort. Idempotent.
  void ForceStop();

  /// \brief Blocks the calling thread until every worker loaded its
  /// tokenizer or failed. Startup gate for RunServe; not for io threads.
  Status WaitUntilReady(std::chrono::milliseconds timeout);

  PoolHealth Health() const;

 private:
  struct Impl;
  /// Shared so io-posted lambdas outlive a pool destroyed while the
  /// io_context is still draining its queues.
  std::shared_ptr<Impl> impl_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_TOKENIZER_POOL_H_
