#ifndef INFERX_ENGINE_SCHEDULER_OUTPUT_H_
#define INFERX_ENGINE_SCHEDULER_OUTPUT_H_

#include <cstdint>
#include <optional>
#include <array>
#include <vector>

#include "absl/types/span.h"
#include "inferx/engine/request.h"

namespace inferx {

/// \file
/// \brief The scheduler's plan for one engine step, and the model runner's
/// answer.
///
/// A step message carries only what is new or changed: the model runner keeps
/// per-request state, so a request appears exactly once per step, either in
/// `scheduled_new_reqs` (full state, first time the runner sees it) or among
/// `scheduled_cached_reqs` (a diff against what the runner already holds).
///
/// `num_new_tokens` is the token-budget allocation, not a phase label: a
/// large value on first schedule is (chunked) prefill and 1 on a cached
/// request is decode. There is no prefill/decode phase split anywhere in the
/// struct.

/// \brief One request's allocation in a scheduler step.
struct ScheduledRequest {
  RequestId request_id = 0;  ///< The request being scheduled.
  int num_new_tokens = 0;    ///< Tokens allocated to this request this step.
};

/// \brief Full state for a request the model runner has never seen.
///
/// `prompt_token_ids` borrows from the Request: the scheduler holds the
/// Request until UpdateFromOutput returns, so the borrow outlives the step.
/// For a resumed preemption victim this is prompt plus retained accepted
/// output -- the replay sequence whose KV must be (re)computed.
struct NewRequestData {
  RequestId request_id = 0;  ///< The request being introduced.
  absl::Span<const TokenId> prompt_token_ids;  ///< Replay sequence, borrowed.
  sampling::SamplingParams sampling_params;  ///< The request's sampling parameters.
  int num_computed_tokens = 0;     ///< Sequence tokens already computed.
};

/// \brief The diff for a request the model runner has seen before.
struct CachedRequestUpdate {
  RequestId request_id = 0;  ///< The request being updated.
  int num_computed_tokens = 0;  ///< End position including this step's allocation.
};

/// \brief The scheduler's plan for one engine step.
/// \brief Flattened per-group block tables for the scheduled batch, in
///        batch order: the scheduler's immutable snapshot at publication
///        (docs/attention.md §9). The runner never reads the live facade
///        for a step it is executing.
inline constexpr int kMaxSchedulerGroups = 8;

struct SchedulerOutput {
  /// \brief This step's allocations, in batch order.
  std::vector<ScheduledRequest> scheduled;
  /// \brief Per cache group: every scheduled request's whole table,
  ///        concatenated in batch order.
  std::array<std::vector<int32_t>, kMaxSchedulerGroups> group_block_ids;
  /// \brief Requests the model runner meets for the first time this step.
  std::vector<NewRequestData> scheduled_new_reqs;
  /// \brief Requests already known to the runner, with their diffs.
  std::vector<CachedRequestUpdate> scheduled_cached_reqs;
  /// \brief Sum of scheduled[].num_new_tokens.
  int total_num_scheduled_tokens = 0;
  /// \brief Requests finished since the last step; the runner frees its
  /// per-request state for these.
  std::vector<RequestId> finished_request_ids;
  /// \brief Requests preempted while building this step; the model runner
  /// drops their state too -- they come back later as fresh requests whose
  /// prefix-cache hits stand in for the recomputation they avoid.
  std::vector<RequestId> preempted_request_ids;

  /// \brief True when the step neither runs work nor reports state changes.
  bool IsEmpty() const {
    return scheduled.empty() && finished_request_ids.empty() &&
           preempted_request_ids.empty();
  }
};

/// \brief Tokens sampled for one scheduled request.
struct SampledTokens {
  RequestId request_id = 0;        ///< The request the tokens belong to.
  std::vector<TokenId> token_ids;  ///< Sampled token ids, in order.

  /// \brief Set when this step ended the request; absent otherwise.
  ///
  /// The scheduler owns finish accounting; this is the runner's observation
  /// (e.g. a stop token surfaced during sampling), which the scheduler
  /// reconciles in UpdateFromOutput.
  std::optional<FinishReason> finish_reason;
};

/// \brief The model runner's result for one engine step.
///
/// Order matches SchedulerOutput::scheduled: samples[i] answers scheduled[i].
struct ModelRunnerOutput {
  std::vector<SampledTokens> samples;  ///< Per scheduled request.
};

}  // namespace inferx

#endif  // INFERX_ENGINE_SCHEDULER_OUTPUT_H_
