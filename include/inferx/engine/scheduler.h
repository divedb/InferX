#ifndef INFERX_ENGINE_SCHEDULER_H_
#define INFERX_ENGINE_SCHEDULER_H_

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/engine/request.h"
#include "inferx/engine/request_queue.h"
#include "inferx/engine/scheduler_output.h"

namespace inferx {

/// \brief Tuning knobs of the scheduling policy.
struct SchedulerConfig {
  /// \brief Token budget per step, across all scheduled requests.
  int max_num_batched_tokens = 4096;
  /// \brief Upper bound on concurrently running requests.
  int max_num_seqs = 32;
  /// \brief Bound on the waiting queue; admissions beyond it are rejected.
  int queue_capacity = 64;
  /// \brief Whether a long prompt may be split across steps within the token
  ///        budget (the scheduler's native mode; CLI contract for vLLM's
  ///        --enable-chunked-prefill).
  bool chunked_prefill = true;
};

/// \brief Snapshot of scheduler occupancy and cumulative counters.
struct SchedulerStats {
  int num_running = 0;             ///< Requests currently running.
  int num_waiting = 0;             ///< Requests in the waiting queue.
  float kv_cache_usage = 0.0f;     ///< Fraction of KV blocks in use.
  int steps = 0;                   ///< Engine steps executed.
  int scheduled_tokens_total = 0;  ///< Tokens scheduled across all steps.
  int num_finished = 0;            ///< Requests finished, any reason.
  int num_rejected = 0;            ///< Admissions refused (queue full).
};

/// \brief Owns request scheduling.
///
/// Requests enter through AddRequest and leave through Schedule() and
/// UpdateFromOutput(): Schedule() produces the next step's SchedulerOutput
/// (admitting, chunking, and allocating KV blocks), and UpdateFromOutput()
/// applies the model runner's samples, finishing requests and freeing their
/// blocks. Single-threaded by design: the engine loop calls these two in
/// sequence, never concurrently.
///
/// The policy is FCFS with a unified token budget: running requests get one
/// token per step (or the next prefill chunk when their prompt is longer
/// than the budget admits), and waiting requests are admitted in arrival
/// order while budget, sequence slots, and KV blocks allow. KV blocks come
/// from the KvBlockPool shared with the model runner; the scheduler never
/// touches tensors.
class Scheduler {
 public:
  /// \brief Wires the scheduler to the KV block pool.
  ///
  /// \param config       Policy knobs.
  /// \param pool         The KV block pool to allocate from; must outlive the
  ///                     scheduler.
  /// \param eos_token_id Model end-of-sequence token used for finish
  ///                     detection (requests may override via ignore_eos).
  Scheduler(SchedulerConfig config, KvBlockPool* pool, TokenId eos_token_id);

  /// \brief Admits a request to the waiting queue.
  ///
  /// \param request The request to schedule eventually.
  /// \return        OK, or ResourceExhausted when the queue is full.
  Status AddRequest(Request request);

  /// \brief Computes the next step's plan against the token budget.
  ///
  /// Advances each scheduled request's computed-token watermark and emits
  /// the finish notifications gathered by the previous UpdateFromOutput.
  ///
  /// \return The plan, or an error status.
  StatusOr<SchedulerOutput> Schedule();

  /// \brief Applies the model runner's samples for one step.
  ///
  /// Appends sampled tokens, finishes requests that hit a stop condition
  /// (EOS, stop tokens, or the token cap), and frees finished requests'
  /// KV blocks.
  ///
  /// \param output The plan produced by Schedule() for this step.
  /// \param result The runner's answer to `output`.
  /// \return       OK, or an error status.
  Status UpdateFromOutput(const SchedulerOutput& output, const ModelRunnerOutput& result);

  /// \brief Removes a finished request, transferring its output tokens.
  ///
  /// \return The finished request, or nullopt when none is waiting.
  std::optional<Request> PopFinished();

  /// \brief Aborts requests by id.
  ///
  /// \param request_ids Ids to abort; unknown ids are ignored.
  /// \return            The ids that were actually aborted.
  std::vector<uint64_t> AbortRequests(absl::Span<const uint64_t> request_ids);

  /// \brief True while the scheduler holds waiting or running requests.
  bool HasRequests() const { return !waiting_.empty() || !running_.empty(); }

  /// \brief Returns the current counter snapshot.
  SchedulerStats Stats() const;

 private:
  /// \brief Grants blocks until the request can hold `target_tokens`.
  ///
  /// \param request The request, already holding zero or more blocks.
  /// \param target_tokens Tokens the block table must cover.
  /// \param granted   Output: newly granted block ids, in order.
  /// \return          OK, or ResourceExhausted when the pool cannot cover the
  ///                  request this step (the caller leaves it scheduled-out).
  Status EnsureBlockCapacity(Request* request, int64_t target_tokens,
                             std::vector<int32_t>* granted);

  SchedulerConfig config_;
  KvBlockPool* pool_ = nullptr;
  TokenId eos_token_id_ = 0;

  RequestQueue waiting_;
  /// Running requests in admission order.
  std::vector<Request> running_;
  absl::flat_hash_map<RequestId, size_t> running_index_;

  /// Finished requests awaiting pickup, oldest first.
  std::deque<Request> finished_;
  /// Finishes gathered by UpdateFromOutput, delivered by the next Schedule().
  std::vector<RequestId> pending_finish_notifications_;

  SchedulerStats stats_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_SCHEDULER_H_
