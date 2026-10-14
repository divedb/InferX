#ifndef INFERX_ENGINE_SCHEDULER_H_
#define INFERX_ENGINE_SCHEDULER_H_

#include <cstdint>
#include <deque>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/cache/cache_manager.h"
#include "inferx/config/cache_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/engine/request.h"
#include "inferx/engine/request_queue.h"
#include "inferx/engine/scheduler_output.h"

namespace inferx {

/// \brief Snapshot of scheduler occupancy and cumulative counters.
struct SchedulerStats {
  int num_running = 0;             ///< Requests currently running.
  int num_waiting = 0;             ///< Requests in the waiting queue.
  float kv_cache_usage = 0.0f;     ///< Fraction of KV blocks in use.
  int steps = 0;                   ///< Engine steps executed.
  int scheduled_tokens_total = 0;  ///< Tokens scheduled across all steps.
  int num_finished = 0;            ///< Requests finished, any reason.
  int num_rejected = 0;            ///< Admissions refused (queue full).
  int num_preempted = 0;           ///< Running requests preempted for KV space.

  /// Token-granular prefix-cache accounting, recorded at admission (vLLM
  /// PrefixCacheStats analog). Queried tokens are the prompts of admissions
  /// that consulted the cache; hit tokens are what an adoption returned.
  int prefix_queries = 0;          ///< Admissions that consulted the cache.
  int prefix_hits = 0;             ///< Admissions with a non-empty adoption.
  int64_t prefix_queried_tokens = 0;  ///< Prompt tokens behind prefix_queries.
  int64_t prefix_hit_tokens = 0;      ///< Prompt tokens served from cache.
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
/// from CacheManager over the pool shared with the model runner; the scheduler never
/// touches tensors.
class Scheduler {
 public:
  /// \brief Wires the scheduler to the KV block pool.
  ///
  /// \param config       Policy knobs.
  /// \param cache        Cache policy; enables prefix caching when
  ///                     cache.enable_prefix_caching is set.
  /// \param pool         The KV block pool to allocate from; must outlive the
  ///                     scheduler.
  /// \param eos_token_ids Model's declared end-of-sequence tokens used for
  ///                     finish detection; generation halts on any (requests
  ///                     may override via ignore_eos).
  Scheduler(SchedulerConfig config, const CacheConfig& cache, KvBlockPool* pool,
            std::vector<TokenId> eos_token_ids);

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
  SchedulerConfig config_;
  CacheManager cache_manager_;
  std::vector<TokenId> eos_token_ids_;

  RequestQueue waiting_;
  /// Running requests in admission order.
  std::vector<Request> running_;
  absl::flat_hash_map<RequestId, size_t> running_index_;

  /// \brief Preempts the newest running request after `keep` so the request
  ///        at `keep` can grow.
  ///
  /// vLLM V1 semantics: victims are newest-first (lowest priority under
  /// FCFS), only requests not yet scheduled this step, and their cached
  /// prefix blocks survive in the index -- resumption re-adopts them. The
  /// victim requeues at the waiting head with a zeroed watermark.
  ///
  /// \param keep     The index of the request that needs the space.
  /// \param preempted Output ids of this step's victims, appended here.
  /// \return          True when a victim was preempted.
  bool PreemptNewestAfter(size_t keep, std::vector<RequestId>* preempted);

  /// Finished requests awaiting pickup, oldest first.
  std::deque<Request> finished_;
  /// Finishes gathered by UpdateFromOutput, delivered by the next Schedule().
  std::vector<RequestId> pending_finish_notifications_;

  SchedulerStats stats_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_SCHEDULER_H_
