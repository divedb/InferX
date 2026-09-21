#ifndef INFERX_ENGINE_REQUEST_H_
#define INFERX_ENGINE_REQUEST_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/status.h"

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

using TokenId = int;

/// \brief Parameters controlling token sampling and generation termination.
struct SamplingParams {
  /// \brief Maximum number of tokens to generate.
  ///
  /// This limit applies only to generated tokens and does not include prompt
  /// tokens. Generation terminates with FinishReason::kLengthCapped when this
  /// limit is reached.
  ///
  /// EXAMPLE:
  /// A request with 100 prompt tokens and max_tokens = 32 may contain up to
  /// 132 tokens in total.
  std::uint32_t max_tokens = 16;

  /// \brief Minimum number of tokens to generate before normal stop conditions
  ///        are allowed to terminate generation.
  ///
  /// EOS and other normal stop conditions should be suppressed until at least
  /// this many tokens have been generated. Explicit cancellation and errors
  /// are not affected by this setting.
  std::uint32_t min_tokens = 0;

  /// \brief Temperature applied to logits before sampling.
  ///
  /// A value of 0 enables greedy decoding. Values greater than 0 enable
  /// stochastic sampling after scaling logits by the temperature.
  ///
  /// Higher values generally produce a flatter probability distribution,
  /// while lower positive values make sampling more concentrated on
  /// high-probability tokens.
  ///
  /// Must be greater than or equal to 0.
  ///
  /// EXAMPLE:
  ///   temperature = 0.0f  // Greedy decoding.
  ///   temperature = 1.0f  // No temperature scaling.
  float temperature = 1.0f;

  /// \brief Number of highest-probability candidate tokens retained for
  ///        sampling.
  ///
  /// A value of 0 disables top-k filtering.
  std::uint32_t top_k = 0;

  /// \brief Cumulative probability threshold used for nucleus (top-p)
  ///        sampling.
  ///
  /// The smallest set of highest-probability tokens whose cumulative
  /// probability reaches this threshold is retained.
  ///
  /// A value of 1.0 disables top-p filtering.
  ///
  /// Must be in the range (0, 1].
  float top_p = 1.0f;

  /// \brief Minimum probability threshold relative to the most probable token.
  ///
  /// Tokens whose probability is below min_p times the probability of the
  /// most probable token are excluded from sampling.
  ///
  /// A value of 0 disables min-p filtering.
  ///
  /// Must be in the range [0, 1].
  float min_p = 0.0f;

  /// \brief Penalty applied when a token has already appeared in the generated
  ///        output.
  ///
  /// The penalty depends only on whether a token has appeared, not on how many
  /// times it has appeared. Positive values discourage repeated tokens;
  /// negative values encourage them.
  ///
  /// A value of 0 disables the presence penalty.
  ///
  /// EXAMPLE:
  ///   presence_penalty = 0.0f  // Disabled.
  float presence_penalty = 0.0f;

  /// \brief Penalty based on how frequently a token has appeared in the
  ///        generated output.
  ///
  /// Positive values increasingly penalize tokens as their occurrence count
  /// grows. Negative values encourage repeated occurrences.
  ///
  /// A value of 0 disables the frequency penalty.
  float frequency_penalty = 0.0f;

  /// \brief Multiplicative penalty applied to previously observed tokens.
  ///
  /// A value of 1.0 disables the repetition penalty. Values greater than 1.0
  /// discourage repetition.
  ///
  /// This penalty is distinct from presence_penalty and frequency_penalty.
  float repetition_penalty = 1.0f;

  /// \brief Whether EOS tokens should be ignored as generation stop
  ///        conditions.
  ///
  /// When false, generation normally terminates when an EOS token configured
  /// by the model/tokenizer is generated. When true, EOS tokens do not
  /// terminate generation.
  ///
  /// This setting does not disable stop_token_ids or stop_strings.
  ///
  /// EXAMPLE:
  ///   ignore_eos = false  // Stop normally when EOS is generated.
  bool ignore_eos = false;

  /// \brief Additional token IDs that terminate generation.
  ///
  /// These are request-specific stop tokens and are separate from the model's
  /// EOS token IDs.
  ///
  /// Stop-token termination is suppressed while the number of generated
  /// tokens is less than min_tokens.
  std::vector<TokenId> stop_token_ids;

  /// \brief Text sequences that terminate generation when encountered.
  ///
  /// Stop strings are matched against incrementally detokenized output rather
  /// than directly against token IDs. A stop string may therefore span
  /// multiple tokens.
  ///
  /// Stop-string termination is suppressed while the number of generated
  /// tokens is less than min_tokens.
  std::vector<std::string> stop_strings;

  /// \brief Optional random seed used for stochastic sampling.
  ///
  /// When specified, the engine should use the seed to initialize the
  /// request's sampling RNG. The exact reproducibility guarantees may depend
  /// on the execution backend, batching, and sampling implementation.
  ///
  /// This value has no effect on greedy decoding.
  std::optional<std::uint64_t> seed;

  /// \brief Number of token log-probabilities to return for each generated
  ///        token.
  ///
  /// A value of 0 disables log-probability output. A positive value requests
  /// log-probability information for the sampled token and up to the requested
  /// number of highest-probability candidates.
  ///
  /// The implementation may impose an upper bound on this value.
  std::uint32_t logprobs = 0;
};

using RequestId = uint64_t;

class Request {
 public:
  Request(uint64_t id, std::vector<TokenId> prompt, const SamplingParams& params = {})
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
  const SamplingParams& sampling_params() const { return sampling_params_; }

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
  SamplingParams sampling_params_;

  RequestStatus status_ = RequestStatus::kWaiting;
  FinishReason finish_reason_ = FinishReason::kStopped;

  int max_new_tokens_ = 0;
  int num_computed_tokens_ = 0;
  std::optional<BlockTable> block_table_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_H_
