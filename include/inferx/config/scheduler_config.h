/// \file
/// \brief Scheduler tuning knobs (vLLM SchedulerConfig analogue).

#ifndef INFERX_CONFIG_SCHEDULER_CONFIG_H_
#define INFERX_CONFIG_SCHEDULER_CONFIG_H_

namespace inferx {

/// \brief Tuning knobs of the scheduling policy.
struct SchedulerConfig {
  /// \brief Token budget per step, across all scheduled requests.
  int max_num_batched_tokens = 4096;
  /// \brief Upper bound on concurrently running requests.
  int max_num_seqs = 32;
  /// \brief Bound on the waiting queue; admissions beyond it are rejected.
  int queue_capacity = 64;
  /// \brief Whether a long prompt may be split across steps within the token
  ///        budget (the scheduler's native mode; CLI contract for vLLM's
  ///        --enable-chunked-prefill).
  bool chunked_prefill = true;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_SCHEDULER_CONFIG_H_
