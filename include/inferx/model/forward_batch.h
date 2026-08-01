#pragma once

#include <cstdint>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::model {

/// \brief One step's worth of work, flattened across every sequence in it.
///
/// The precursor to §5's `BatchPlan`, and shaped like it on purpose: the
/// scheduler decides all of this and the model does none of it. Prefill and
/// decode are the same structure with different proportions -- prefill is many
/// tokens from one sequence, decode is one token from each of many -- which is
/// what lets a single forward path serve both and, later, a mixed batch (§8.1).
///
/// Everything is per-token and parallel: index `i` of `token_ids`, `positions`,
/// `seq_of_token` and `slots` all describe the same token.
struct ForwardBatch {
  /// Token ids to run, concatenated across sequences.
  std::vector<int32_t> token_ids;
  /// Absolute position of each token in its own sequence. Also how many keys
  /// precede it, which is what makes the attention causal without a mask.
  std::vector<int32_t> positions;
  /// Which sequence each token belongs to: a row index into `block_table`.
  std::vector<int32_t> seq_of_token;
  /// Where each token's K/V is written: `block · block_size + offset`.
  std::vector<int32_t> slots;

  /// Row-major `[num_seqs, max_blocks_per_seq]`. Unused entries may be
  /// anything; attention never reads past a token's own position.
  std::vector<int32_t> block_table;
  int64_t num_seqs = 0;
  int64_t max_blocks_per_seq = 0;

  /// When true, `token_ids` is ignored and the device buffer is used as it
  /// stands -- because the previous step's sampler already wrote this step's
  /// tokens into it.
  ///
  /// This is what closes §5.2's loop. Uploading token ids from the host would
  /// mean the host had to know them, which would mean waiting for the previous
  /// step to finish, which is the synchronization the pipeline exists to
  /// remove. A prefill still supplies its tokens normally; only a
  /// device-sampled continuation sets this.
  bool tokens_from_device = false;

  /// Which token indices need logits. Usually one per sequence -- the last --
  /// because computing the LM head over every prefill token costs a
  /// `[tokens, 151936]` GEMM to throw nearly all of it away.
  std::vector<int32_t> logits_indices;

  /// Per-logits-row sampling parameters, parallel to `logits_indices`.
  ///
  /// They ride on the batch rather than being configured on the model because
  /// they are per *request*: one sequence in a step may be greedy while another
  /// samples at 0.8. A model-wide setting could not express that, and the
  /// scheduler is the only component that knows which request each row belongs
  /// to.
  ///
  /// Empty means greedy for every row, which is what every caller predating
  /// sampling gets without changing.
  std::vector<float> temperature;
  std::vector<float> top_p;
  std::vector<uint64_t> seeds;

  int64_t num_tokens() const {
    return static_cast<int64_t>(token_ids.size());
  }

  /// \brief Checks the internal consistency the model would otherwise trust.
  Status Validate(int64_t vocab_size, int64_t total_slots) const;
};

}  // namespace inferx::model
