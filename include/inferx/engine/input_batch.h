#ifndef INFERX_ENGINE_INPUT_BATCH_H_
#define INFERX_ENGINE_INPUT_BATCH_H_

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/sampling/sampler.h"

namespace inferx {

/// \brief Persistent, row-oriented per-request state for one rank.
///
/// Replaces the runner's unordered_map<RequestId, RequestState>. Each live
/// request owns one row; token ids, counters, and sampling knobs are stored
/// in flat arrays so per-step preparation reads contiguous memory. The batch
/// survives across steps and is mutated only by the diff in SchedulerOutput
/// (add new, update cached, remove finished).
///
/// Block tables are NOT stored here; they live in the attention runtime's
/// cache facade (docs/attention.md §8.3).
///
/// Not thread-safe; owned and used by a single ModelRunner.
class InputBatch {
 public:
  struct NewRequest {
    RequestId id;
    std::vector<TokenId> prompt_tokens;
    int num_computed = 0;
    sampling::SamplingParams params;
  };

  InputBatch(int max_num_reqs, int max_model_len);

  /// \brief Adds a request in the lowest free row. Fails if the batch is
  ///        full, the id is already present, or the prompt exceeds
  ///        max_model_len.
  Status Add(const NewRequest& req);

  /// \brief Frees the request's row, leaving a hole until Condense().
  ///        Unknown ids are ignored.
  void Remove(RequestId id);

  /// \brief Moves rows from the tail into holes so live rows are
  ///        contiguous [0, NumReqs()). Row indices change.
  void Condense();

  /// \brief Appends an accepted token (sampled locally or received from
  ///        the driver rank after a group execution).
  Status AppendToken(RequestId id, TokenId token);
  Status SetNumComputed(RequestId id, int num_computed);

  std::optional<int> RowOf(RequestId id) const;

  int NumReqs() const { return static_cast<int>(id_to_row_.size()); }
  /// \brief High-water row count including holes; bound for row loops.
  int NumRows() const { return static_cast<int>(row_to_id_.size()); }
  int MaxNumReqs() const { return max_num_reqs_; }
  int MaxModelLen() const { return max_model_len_; }

  // Row accessors; `row` must be a live row.
  RequestId IdAt(int row) const { return row_to_id_[row]; }
  const TokenId* TokensAt(int row) const {
    return token_ids_.data() + static_cast<int64_t>(row) * max_model_len_;
  }
  int NumTokens(int row) const { return num_tokens_[row]; }
  int NumPromptTokens(int row) const { return num_prompt_tokens_[row]; }
  int NumComputed(int row) const { return num_computed_[row]; }
  const sampling::SamplingParams& ParamsAt(int row) const { return params_[row]; }
  int64_t RngOffset(int row) const { return rng_offset_[row]; }
  void AdvanceRngOffset(int row, int64_t n = 1) { rng_offset_[row] += n; }

 private:
  static constexpr RequestId kFreeRow = static_cast<RequestId>(-1);

  int max_num_reqs_;
  int max_model_len_;

  std::unordered_map<RequestId, int> id_to_row_;
  std::vector<RequestId> row_to_id_;  ///< kFreeRow marks a hole.
  std::vector<int> free_rows_;        ///< Holes, reused lowest-first.

  std::vector<TokenId> token_ids_;  ///< [max_num_reqs, max_model_len]
  std::vector<int> num_tokens_;
  std::vector<int> num_prompt_tokens_;
  std::vector<int> num_computed_;
  std::vector<sampling::SamplingParams> params_;
  std::vector<int64_t> rng_offset_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_INPUT_BATCH_H_
