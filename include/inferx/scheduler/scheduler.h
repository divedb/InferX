#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/forward_batch.h"

namespace inferx::scheduler {

using RequestId = uint64_t;

/// \brief What a request wants generated.
struct SamplingParams {
  /// Hard cap on generated tokens. Reached is a normal finish, not an error.
  int32_t max_tokens = 16;
  /// Any of these ends the sequence. Empty means only `max_tokens` stops it.
  std::vector<int32_t> stop_tokens;

  /// Softmax temperature. Zero or below is greedy, which is the limit of the
  /// distribution rather than a separate mode.
  float temperature = 0.0f;

  /// Nucleus threshold. At or above 1 disables truncation.
  float top_p = 1.0f;

  /// Fixes the draw. A request that pins its seed reproduces exactly, which is
  /// what makes a sampled server debuggable.
  uint64_t seed = 0;
};

enum class FinishReason {
  kNotFinished,
  kStopToken,
  kMaxTokens,
  kCancelled,
  /// Admitted, then the pool ran out while it was growing. Distinct from
  /// rejection at admission, because the caller has partial output.
  kOutOfMemory,
};

const char* FinishReasonName(FinishReason reason);

/// \brief One token appended to one running sequence by a single step.
///
/// A `Completion` arrives when a sequence is over, which is too late for
/// streaming -- SSE has to emit each token as it is produced. This is the
/// per-step view of the same thing: `CommitStep` reports which request each
/// sampled token belongs to, so the layer above can route it to a client
/// without having to reconstruct the batch's ordering for itself.
///
/// It carries no text. Detokenization is stateful across tokens (a character
/// can span two of them) and is the caller's business, not the scheduler's --
/// §3.1 keeps the scheduler to plain integers.
struct TokenDelta {
  RequestId id = 0;
  int32_t token = 0;

  /// Set when this token also ended the sequence, so a streaming caller can
  /// close the stream on the same step rather than waiting for `TakeCompleted`.
  FinishReason finish = FinishReason::kNotFinished;
};

/// \brief A finished sequence, handed back to whatever submitted it.
struct Completion {
  RequestId id = 0;
  std::vector<int32_t> output_tokens;
  FinishReason reason = FinishReason::kNotFinished;
};

struct SchedulerConfig {
  /// Sequences resident at once. Bounds the batch and the block table's width.
  int64_t max_running = 8;
  /// Tokens per step across the whole batch (§8.1). A prefill longer than this
  /// is not yet chunked -- that is M5 -- so for now it bounds admission.
  int64_t max_batch_tokens = 2048;
  /// Longest sequence, prompt plus generation. Bounds `max_blocks_per_seq`.
  int64_t max_seq_len = 2048;
};

/// \brief FCFS request scheduling over a paged KV cache. Single-threaded.
///
/// The component §3.1 cares most about: it holds *all* the policy and touches
/// *no* CUDA. Everything it produces is a `ForwardBatch` of plain integers and
/// everything it consumes is a vector of sampled token ids, which is what makes
/// it unit-testable on a machine with no GPU -- the pool can be host-allocated
/// and the whole lifecycle exercised without a device present.
///
/// M3 scope: FCFS admission (§8.3), one prefill per step, synchronous stepping.
/// No chunked prefill, no preemption, no prefix cache -- those are M5, and the
/// interface here is the one they extend rather than replace. Notably
/// `PrepareStep` already returns a batch that can mix a prefill with decodes,
/// because that is the shape chunked prefill needs.
///
/// The step loop is deliberately the depth-0 case of §5.2: prepare, execute,
/// commit, with nothing in flight across the boundary. T6 keeps this path as
/// the correctness reference that the overlapped version is diffed against, so
/// it stays even after M6 makes it slow.
class Scheduler {
 public:
  /// \brief Creates a scheduler over `pool`, which must outlive it.
  static StatusOr<Scheduler> Create(const SchedulerConfig& config,
                                    KvBlockPool* pool);

  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;
  Scheduler(Scheduler&&) noexcept;
  Scheduler& operator=(Scheduler&&) noexcept;

  /// \brief Queues a request. Admission happens later, in `PrepareStep`.
  ///
  /// \param id     Caller's identifier, returned with the completion.
  /// \param prompt Prompt tokens. Must be non-empty.
  /// \param params What to generate.
  Status AddRequest(RequestId id, std::vector<int32_t> prompt,
                    SamplingParams params);

  /// \brief Marks a request finished without generating further tokens.
  ///
  /// Safe to call for a request that is queued, running, or already gone --
  /// a disconnect races with everything, so it cannot be an error (§4, step 10).
  void Cancel(RequestId id);

  /// \brief Builds the next step's batch, admitting waiting requests if it can.
  ///
  /// \param out_batch  Receives the batch. Left with no tokens when there is
  ///                   nothing to run, which is not an error.
  /// \return           OK, or the reason admission could not proceed at all.
  Status PrepareStep(model::ForwardBatch* out_batch);

  /// \brief Applies one sampled token per sequence in the last prepared batch.
  ///
  /// \param sampled     One token per entry of the batch's `logits_indices`, in
  ///                    the same order.
  /// \param out_deltas  Optional. Receives one entry per sampled token, naming
  ///                    the request it belongs to. Cleared first. Streaming
  ///                    callers need this; batch callers can ignore it and read
  ///                    `TakeCompleted` instead.
  Status CommitStep(const std::vector<int32_t>& sampled,
                    std::vector<TokenDelta>* out_deltas = nullptr);

  /// \brief Removes and returns everything that finished since the last call.
  std::vector<Completion> TakeCompleted();

  /// \brief True while any request is queued or running.
  bool HasWork() const;

  int64_t num_running() const;
  int64_t num_waiting() const;

  /// \brief Blocks currently held by running sequences.
  int64_t blocks_in_use() const;

 private:
  struct Impl;

  explicit Scheduler(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::scheduler
