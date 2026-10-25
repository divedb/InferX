#ifndef INFERX_ENGINE_REQUEST_H_
#define INFERX_ENGINE_REQUEST_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/status.h"
#include "inferx/engine/request_types.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

/// \brief A single generation request and its mutable per-request state.///        The request vocabulary (RequestId, RequestStatus, FinishReason,
///        StopReason) lives in request_types.h, shared with the serving
///        boundary.
///
/// Token storage: prompt and generated tokens live in ONE contiguous vector
/// (`all_token_ids_`), with the prompt occupying `[0, num_prompt_tokens())`
/// and generated tokens following. This gives the cache manager, prefix
/// hashing and speculative verification a single sequence to work on, while
/// prompt() / output() expose cheap zero-copy views.
///
/// Threading: a Request is not thread-safe. It is owned and mutated by the
/// engine/scheduler thread only.
class Request {
 public:
  /// \brief Creates a request in the kWaiting state.
  ///
  /// \param id           Engine-assigned unique id.
  /// \param prompt       Prompt token ids; moved in and treated as immutable.
  /// \param params       Sampling parameters. `params.max_tokens` is the
  ///                     single source of truth for the generation bound.
  /// \param arrival_time Monotonic arrival timestamp in seconds, used for
  ///                     FCFS ordering, latency metrics and timeouts.
  Request(RequestId id, std::vector<TokenId> prompt, sampling::SamplingParams params = {},
          double arrival_time = 0.0)
      : id_(id),
        all_token_ids_(std::move(prompt)),
        num_prompt_tokens_(static_cast<int>(all_token_ids_.size())),
        sampling_params_(std::move(params)),
        arrival_time_(arrival_time) {}

  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
  Request(Request&&) = default;
  Request& operator=(Request&&) = default;

  /// \brief Returns the unique request id assigned by the engine.
  ///
  /// \return The request id.
  RequestId GetRequestId() const noexcept { return id_; }

  /// \brief Returns the sampling parameters for this request.
  ///
  /// \return The sampling parameters.
  const sampling::SamplingParams& GetSamplingParams() const noexcept {
    return sampling_params_;
  }

  /// \brief Returns the bound on generated tokens (`GetSamplingParams().max_tokens`).
  ///
  /// \return The maximum number of generated tokens.
  int MaxNewTokens() const noexcept { return GetSamplingParams().max_tokens; }

  /// \brief Monotonic arrival time in seconds.
  ///
  /// Intended for FCFS tie-breaks, latency metrics, and timeouts. Callers
  /// that omit it get 0.0; equal timestamps fall back to queue order.
  ///
  /// \return The arrival time in seconds.
  double ArrivalTime() const noexcept { return arrival_time_; }

  /// \brief Scheduling priority; reserved for priority policies (backlog
  ///        A10). No policy reads it yet; lower values would schedule first.
  ///
  /// \return The scheduling priority.
  int Priority() const noexcept { return priority_; }

  /// \brief Sets the priority; only meaningful before admission.
  ///
  /// \param priority The new priority value.
  void SetPriority(int priority) { priority_ = priority; }

  /// \brief Per-request EOS override, reserved for future stop checking.
  ///
  /// No caller reads or sets it today: the scheduler checks its engine-wide
  /// EOS token list against every request.
  ///
  /// \return The EOS token id.
  std::optional<TokenId> EosTokenId() const noexcept { return eos_token_id_; }

  /// \brief Sets the EOS token id; only meaningful before admission.
  ///
  /// \param eos The new EOS token id.
  void SetEosTokenId(std::optional<TokenId> eos) { eos_token_id_ = eos; }

  /// \brief Tenant-scoped prefix-cache salt.
  ///
  /// Salted and unsalted versions of one prompt share no cached blocks: the
  /// salt is folded into the hash chain's seed, defeating cross-tenant
  /// timing side channels. Empty means unsalted.
  ///
  /// \return The cache salt.
  const std::string& CacheSalt() const noexcept { return cache_salt_; }

  /// \brief Sets the salt; only meaningful before admission.
  ///
  /// \param salt The new cache salt.
  void SetCacheSalt(std::string salt) { cache_salt_ = std::move(salt); }

  /// \brief Returns the prompt token ids (view; invalidated by AppendOutput).
  ///
  /// \return The prompt token ids.
  absl::Span<const TokenId> Prompt() const noexcept {
    return absl::MakeConstSpan(all_token_ids_)
        .subspan(0, static_cast<std::size_t>(num_prompt_tokens_));
  }

  /// \brief Returns the tokens generated so far (view; invalidated by AppendOutput).
  ///
  /// \return The generated token ids.
  absl::Span<const TokenId> Output() const noexcept {
    return absl::MakeConstSpan(all_token_ids_)
        .subspan(static_cast<std::size_t>(num_prompt_tokens_));
  }

  /// \brief Returns prompt followed by generated tokens as one sequence
  ///        (view; invalidated by AppendOutput).
  ///
  /// \return The prompt + generated token ids.
  absl::Span<const TokenId> AllTokenIds() const noexcept {
    return absl::MakeConstSpan(all_token_ids_);
  }

  /// \brief Number of prompt tokens. Fixed for the lifetime of the request.
  ///
  /// \return The number of prompt tokens.
  int NumPromptTokens() const noexcept { return num_prompt_tokens_; }

  /// \brief Number of generated tokens so far.
  ///
  /// \return The number of generated tokens.
  int NumOutputTokens() const noexcept { return NumTokens() - num_prompt_tokens_; }

  /// \brief Total tokens (prompt + generated).
  ///
  /// \return The total number of tokens.
  int NumTokens() const noexcept { return static_cast<int>(all_token_ids_.size()); }

  /// \brief Appends sampled tokens to the output.
  ///
  /// \param token_ids Newly sampled token ids, in order.
  /// \return          OK, or FailedPrecondition unless the request is
  ///                  running (waiting, preempted, and finished requests
  ///                  cannot accept tokens).
  Status AppendOutput(absl::Span<const TokenId> token_ids);

  /// \brief Tokens whose KV is already computed (prefix-cache hits count).
  ///        Monotonically advances while running; reset by Preempt().
  ///
  /// \return The number of computed tokens.
  int NumComputedTokens() const noexcept { return num_computed_tokens_; }

  /// \brief Updates the computed-token watermark. Scheduler-only.
  ///
  /// \param computed The new watermark; must be >= the current value.
  ///                 Preempt()'s reset to zero is the sole exception.
  void SetNumComputedTokens(int computed) { num_computed_tokens_ = computed; }

  /// \brief Prompt tokens served from the prefix cache at first scheduling,
  ///        or -1 if the request has not been scheduled yet.
  ///
  /// Unlike NumComputedTokens(), this is recorded once and never overwritten, so it stays valid
  /// for hit-rate metrics after the request progresses or is preempted.
  ///
  /// \return The number of cached tokens, or -1 if not scheduled yet.
  int NumCachedTokens() const noexcept { return num_cached_tokens_; }

  /// \brief Records the prefix-cache hit count. Scheduler-only; no-op after
  ///        the first call. Values below zero are ignored.
  ///
  /// \param cached The number of prompt tokens served from the prefix cache.
  void SetNumCachedTokens(int cached) {
    if (cached >= 0 && num_cached_tokens_ < 0) num_cached_tokens_ = cached;
  }

  /// \brief Returns the current lifecycle state.
  ///
  /// \return The current request status.
  RequestStatus GetStatus() const noexcept { return status_; }

  /// \brief True once the request reached the terminal state.
  ///
  /// \return True if the request is finished.
  bool IsFinished() const noexcept { return status_ == RequestStatus::kFinished; }

  /// \brief Admits the request to run (scheduler-only).
  ///
  /// \return OK, or FailedPrecondition when not in kWaiting.
  Status MarkRunning();

  /// \brief Returns the request to the waiting pool (scheduler-only).
  ///
  /// Valid from kPreempted (rescheduling after preemption).
  ///
  /// \return OK, or FailedPrecondition from any other state.
  Status MarkWaiting();

  /// \brief Preempts a running request (scheduler-only).
  ///
  /// Its KV blocks are expected to be released by the CacheManager. This call
  /// resets the computed-token watermark to zero and increments NumPreemptions(), so the
  /// request recomputes from its prefix-cache hits when rescheduled. Generated tokens are kept.
  ///
  /// \return OK, or FailedPrecondition when not running.
  Status Preempt();

  /// \brief Number of times this request has been preempted.
  ///
  /// \return The number of preemptions.
  int NumPreemptions() const noexcept { return num_preemptions_; }

  /// \brief Transitions to the terminal state. Cache release belongs to CacheManager.
  ///
  /// \param reason      Why the request ended.
  /// \param stop_reason Which stop token/string fired; only meaningful when
  ///                    `reason` is kStopped.
  /// \return            OK, or FailedPrecondition when already finished.
  Status Finish(FinishReason reason, StopReason stop_reason = {});

  /// \brief Why the request ended; meaningful once finished.
  ///
  /// \return The finish reason.
  FinishReason GetFinishReason() const noexcept { return finish_reason_; }

  /// \brief Which stop condition fired; monostate if none or not finished.
  ///
  /// \return The stop reason.
  const StopReason& GetStopReason() const noexcept { return stop_reason_; }

 private:
  // The four members below are logically immutable after construction; they
  // are deliberately not const so the defaulted move assignment stays
  // viable (a const member would silently delete it).
  RequestId id_ = 0;

  /// Prompt followed by generated tokens.
  /// \note Declaration order matters: num_prompt_tokens_ is initialized from
  ///       all_token_ids_.size() in the constructor.
  std::vector<TokenId> all_token_ids_;

  /// Prompt length; immutable after construction.
  int num_prompt_tokens_ = 0;
  sampling::SamplingParams sampling_params_;
  double arrival_time_ = 0.0;

  int priority_ = 0;
  std::optional<TokenId> eos_token_id_;
  std::string cache_salt_;

  RequestStatus status_ = RequestStatus::kWaiting;
  FinishReason finish_reason_ = FinishReason::kStopped;
  StopReason stop_reason_;
  int num_preemptions_ = 0;

  // KV / prefix-cache progress.
  int num_computed_tokens_ = 0;
  int num_cached_tokens_ = -1;  ///< -1 until first scheduled.
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_H_