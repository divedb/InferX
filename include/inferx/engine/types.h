#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"

namespace inferx {

/// \brief One rank: its global index, device, and parallel coordinates.
///
/// The topology itself lives in `ParallelConfig` (sizes plus this rank's tp/pp/dp
/// coordinates, derived EP size); nothing is duplicated here.
struct RankInfo {
  int rank = 0;             ///< Global rank in [0, parallel.WorldSize()).
  ParallelConfig parallel;  ///< Sizes + this rank's coordinates.
  Device device;

  int world_size() const { return parallel.WorldSize(); }
  bool is_first_pp_stage() const { return parallel.pipeline_parallel_rank == 0; }
  bool is_last_pp_stage() const {
    return parallel.pipeline_parallel_rank == parallel.pipeline_parallel_size - 1;
  }
  /// \brief True for the one rank whose sampled tokens go back to the scheduler: TP rank 0 of
  ///        the last pipeline stage (rank 0 when pp == 1).
  bool returns_output() const {
    return is_last_pp_stage() && parallel.tensor_parallel_rank == 0;
  }
};

/// \brief Per-sequence slice of a step.
struct SequenceStep {
  int64_t request_id = 0;
  int32_t num_new_tokens = 0;        ///< Tokens fed this step (prefill chunk, or 1 for decode).
  int32_t context_len = 0;           ///< Tokens already in the KV cache before this step.
  int32_t num_generated_tokens = 0;  ///< Used as the sampling RNG offset.
  /// KV block ids covering [0, context_len + num_new_tokens). Owned (not a span) because a
  /// StepInput is shared between rank threads and outlives the scheduler's buffers.
  std::vector<int32_t> block_table;
  SamplingParams sampling;
};

/// \brief Everything a step needs. Built by the scheduler, immutable afterwards.
struct StepInput {
  uint64_t step_id = 0;
  std::vector<int32_t> token_ids;  ///< Flattened new tokens of all sequences.
  std::vector<int32_t> positions;  ///< Flattened positions; same length as token_ids.
  std::vector<SequenceStep> sequences;
  bool is_pure_decode = false;  ///< Every sequence has num_new_tokens == 1.
};

/// \brief Result of a step as seen by the scheduler (rank 0 only).
struct StepOutput {
  uint64_t step_id = 0;
  std::vector<int32_t> sampled_tokens;  ///< One per sequence, in StepInput order.
};

/// \brief Sizes of one packed step. Padding (for graph buckets) is included in the totals.
struct StepShape {
  int32_t num_seqs = 0;     ///< Real sequences.
  int32_t padded_seqs = 0;  ///< == num_seqs unless padded up to a graph bucket.
  int32_t num_tokens = 0;   ///< Real + padding tokens.
  int32_t max_query_len = 0;
  int32_t max_seq_len = 0;
};

/// \brief Device-resident views of a staged step. Pointers stay valid until the next Fill().
struct StepViews {
  StepShape shape;
  Tensor token_ids;   ///< int32 [num_tokens]
  Tensor positions;   ///< int32 [num_tokens]
  Tensor logit_rows;  ///< int32 [padded_seqs]: row of the token to sample, per sequence
  const int32_t* slot_mapping = nullptr;     ///< [num_tokens]
  const int32_t* query_start_loc = nullptr;  ///< [padded_seqs + 1]
  const int32_t* seq_lens = nullptr;         ///< [padded_seqs]
  const int32_t* block_table = nullptr;      ///< [padded_seqs, block_table_stride]
  int32_t block_table_stride = 0;
};


}  // namespace inferx