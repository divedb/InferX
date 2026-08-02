#include "inferx/model/forward_batch.h"

namespace inferx::model {

Status ForwardBatch::Validate(int64_t vocab_size, int64_t total_slots) const {
  const int64_t n = num_tokens();

  if (n == 0) return InvalidArgumentError("batch has no tokens");

  const auto same_size = [&](const std::vector<int32_t>& v, const char* name) {
    return static_cast<int64_t>(v.size()) == n
               ? OkStatus()
               : InvalidArgumentError(name, " has ", v.size(), " entries but "
                                      "there are ", n, " tokens");
  };

  INFERX_RETURN_IF_ERROR(same_size(positions, "positions"));
  INFERX_RETURN_IF_ERROR(same_size(seq_of_token, "seq_of_token"));
  INFERX_RETURN_IF_ERROR(same_size(slots, "slots"));

  if (num_seqs <= 0 || max_blocks_per_seq <= 0) {
    return InvalidArgumentError("batch has num_seqs=", num_seqs,
                                " max_blocks_per_seq=", max_blocks_per_seq);
  }

  if (static_cast<int64_t>(block_table.size()) !=
      num_seqs * max_blocks_per_seq) {
    return InvalidArgumentError("block_table has ", block_table.size(),
                                " entries, expected ",
                                num_seqs * max_blocks_per_seq);
  }

  if (!seq_lens.empty() &&
      static_cast<int64_t>(seq_lens.size()) != num_seqs) {
    return InvalidArgumentError("seq_lens has ", seq_lens.size(),
                                " entries, expected ", num_seqs);
  }

  for (int64_t i = 0; i < n; ++i) {
    const size_t u = static_cast<size_t>(i);

    // Not checked when they come from the device: the value in the vector is a
    // placeholder the model never reads.
    if (!tokens_from_device &&
        (token_ids[u] < 0 || token_ids[u] >= vocab_size)) {
      return InvalidArgumentError("token_ids[", i, "] = ", token_ids[u],
                                  " is outside the vocabulary");
    }
    if (positions[u] < 0) {
      return InvalidArgumentError("positions[", i, "] = ", positions[u],
                                  " is negative");
    }
    if (seq_of_token[u] < 0 || seq_of_token[u] >= num_seqs) {
      return InvalidArgumentError("seq_of_token[", i, "] = ", seq_of_token[u],
                                  " is outside [0, ", num_seqs, ")");
    }
    // A token writes its key at its own position, so the sequence must be
    // declared long enough to contain it. Catches a chunk whose token count
    // and declared length disagree -- attention would then read a key the
    // plan says is not there.
    if (!seq_lens.empty() &&
        positions[u] >= seq_lens[static_cast<size_t>(seq_of_token[u])]) {
      return InvalidArgumentError("positions[", i, "] = ", positions[u],
                                  " is past the end of sequence ",
                                  seq_of_token[u], ", declared ",
                                  seq_lens[static_cast<size_t>(seq_of_token[u])],
                                  " tokens long");
    }
    // A slot outside the pool would scatter this token's keys into whatever
    // follows the cache in device memory.
    if (slots[u] < 0 || slots[u] >= total_slots) {
      return InvalidArgumentError("slots[", i, "] = ", slots[u],
                                  " is outside [0, ", total_slots, ")");
    }
  }

  // An empty `logits_indices` used to be rejected here as obviously useless
  // work. Chunked prefill makes it the ordinary shape of a step that carries
  // several thousand prompt tokens and finishes none of them: the batch is all
  // keys and values, and there is nothing to sample until a prompt's last
  // chunk lands. The scheduler is what guarantees a finished prompt gets a
  // row, and its own tests are where that is checked.

  for (size_t i = 0; i < logits_indices.size(); ++i) {
    if (logits_indices[i] < 0 || logits_indices[i] >= n) {
      return InvalidArgumentError("logits_indices[", i, "] = ",
                                  logits_indices[i], " is outside [0, ", n,
                                  ")");
    }
  }

  return OkStatus();
}

}  // namespace inferx::model
