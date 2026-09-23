#ifndef INFERX_ENGINE_REQUEST_H_
#define INFERX_ENGINE_REQUEST_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/status.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

enum class RequestStatus : std::uint8_t {
  kWaiting = 0,    ///< Queued and eligible for scheduling.
  kWaitingForFsm,  ///< Queued but blocked on FSM/grammar compilation.
  kRunning,        ///< Admitted by the scheduler and currently active.
  kPreempted,      ///< Preempted; KV resources released and awaiting rescheduling.
  kFinished,       ///< Terminal state. See FinishReason for the cause.
};

enum class FinishReason : std::uint8_t {
  kStopped = 0,   ///< Generation stopped normally (EOS, stop token/string, etc.).
  kLengthCapped,  ///< Reached the configured generation/token limit.
  kAborted,       ///< Explicitly cancelled or aborted by the caller.
  kError,         ///< Terminated because of an execution/runtime error.
};

using RequestId = uint64_t;

class Request {
 public:
  Request(uint64_t id, std::vector<TokenId> prompt, const sampling::SamplingParams& params = {})
      : id_(id),
        prompt_(std::move(prompt)),
        sampling_params_(params),
        max_new_tokens_(params.max_tokens) {}

  /// \brief Returns the engine-assigned request id.
  uint64_t id() const { return id_; }
  /// \brief Returns the current lifecycle state.
  RequestStatus status() const { return status_; }
  /// \brief Returns the prompt token ids.
  const std::vector<int>& prompt() const { return prompt_; }
  /// \brief Returns the tokens generated so far.
  const std::vector<int>& output() const { return output_; }
  /// \brief Returns the bound on generated tokens.
  int max_new_tokens() const { return max_new_tokens_; }
  /// \brief Returns the request's sampling parameters.
  const sampling::SamplingParams& sampling_params() const { return sampling_params_; }

  /// \brief Prompt tokens whose KV is already computed (prefix hits count).
  int num_computed_tokens() const { return num_computed_tokens_; }
  /// \brief Updates the computed-token watermark. Scheduler-only.
  void set_num_computed_tokens(int computed) { num_computed_tokens_ = computed; }

  /// \brief True when the scheduler has granted this request cache blocks.
  bool has_blocks() const { return block_table_.has_value(); }
  /// \brief Returns the granted block table; must have_blocks().
  const BlockTable& block_table() const { return *block_table_; }
  /// \brief Returns the granted block table for mutation; scheduler-only.
  BlockTable* mutable_block_table() { return &*block_table_; }
  /// \brief Attaches a block table. Scheduler-only, on cache grant.
  void GrantBlocks(BlockTable table) { block_table_ = std::move(table); }

  /// \brief Appends sampled tokens; refuses when finished.
  Status AppendOutput(absl::Span<const int> token_ids);

  /// \brief Transitions to the terminal state and drops the block table.
  ///
  /// \param reason Why the request ended.
  /// \return       OK, or FailedPrecondition when already finished.
  Status Finish(FinishReason reason);

  /// \brief Why the request ended; meaningful once finished.
  FinishReason finish_reason() const { return finish_reason_; }

 private:
  RequestId id_ = 0;
  std::vector<TokenId> prompt_;
  std::vector<TokenId> output_;
  sampling::SamplingParams sampling_params_;

  RequestStatus status_ = RequestStatus::kWaiting;
  FinishReason finish_reason_ = FinishReason::kStopped;

  int max_new_tokens_ = 0;
  int num_computed_tokens_ = 0;
  std::optional<BlockTable> block_table_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_H_
