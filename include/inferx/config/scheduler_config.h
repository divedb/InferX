#ifndef INFERX_CONFIG_SCHEDULER_CONFIG_H_
#define INFERX_CONFIG_SCHEDULER_CONFIG_H_

namespace inferx {

/// \brief Tuning knobs of the scheduling policy.
struct SchedulerConfig {
  /// Token budget per step, across all scheduled requests.
  /// EXAMPLE: --max-num-batched-tokens 8192
  int max_num_batched_tokens = 4096;

  /// Upper bound on concurrently running requests.
  /// EXAMPLE: --max-num-seqs 64
  int max_num_seqs = 32;

  /// Bound on the waiting queue; admissions beyond it are rejected. Not
  /// exposed as a CLI flag yet.
  int queue_capacity = 64;

  /// Whether a long prompt may be split across steps within the token budget. If false, a
  /// request with a prompt longer than the token budget is rejected.
  /// EXAMPLE: --no-enable-chunked-prefill
  bool chunked_prefill = true;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_SCHEDULER_CONFIG_H_
