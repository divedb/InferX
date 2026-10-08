/// \file
/// \brief Runner-private execution mirror of the scheduler's requests:
///        stable slots, generations guarding ID reuse, committed token
///        watermarks, RNG positions, and the recurrent-slot free list.
///        The scheduler stays authoritative for admission and output
///        history; this table never becomes a second scheduler.

#ifndef INFERX_MODELS_EXECUTION_REQUEST_STATE_H_
#define INFERX_MODELS_EXECUTION_REQUEST_STATE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/status.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/sampling/sampling_params.h"

namespace inferx::models::execution {

/// \brief One active request's execution record.
struct RequestRecord {
  RequestId request_id = 0;
  /// Increments on slot reuse; a delayed result carrying an older
  /// generation must not update this record.
  uint64_t generation = 0;
  std::vector<int32_t> block_ids;  ///< Logical KV block table, in order.
  std::vector<TokenId> prompt;     ///< Prompt tokens, copied at admission.
  sampling::SamplingParams params;
  /// Tokens sampled so far; the request's RNG offset. Advances only when a
  /// step actually samples this request -- never for prefill compute.
  uint64_t generated = 0;
  /// Last successfully completed token watermark (committed, not planned).
  int committed = 0;
  TokenId last_sampled = -1;  ///< Feed-back token for decode steps.
  /// Recurrent-state slot; -1 while unassigned or when unneeded.
  int recurrent_slot = -1;
};

/// \brief Slot-indexed request table. Slots are stable across batch
///        reordering; each step builds a compact batch-to-slot mapping.
class RequestStateTable {
 public:
  explicit RequestStateTable(int max_slots);

  /// \brief Admits a request the runner has not seen (or re-admits a reused
  ///        ID at a fresh generation). Fails on duplicate live IDs and when
  ///        no slot is free.
  StatusOr<RequestRecord*> Admit(const NewRequestData& data);

  /// \brief The live record for `id`, or nullptr.
  RequestRecord* Find(RequestId id);

  /// \brief Retires a finished request: frees its recurrent slot and its
  ///        table slot. The ID may be admitted again later.
  Status Retire(RequestId id);

  /// \brief Fills the recurrent free list when the model has recurrent
  ///        layers; capacity is then visible to admission.
  void InitRecurrentSlots(int slots);

  /// \brief Acquires a recurrent-state slot; errors when exhausted.
  StatusOr<int> AcquireRecurrentSlot();
  void ReleaseRecurrentSlot(int slot);

  /// \brief True when at least one recurrent slot is free.
  bool has_recurrent_capacity() const { return !free_recurrent_.empty(); }

  int max_slots() const { return max_slots_; }
  size_t live() const { return index_.size(); }

 private:
  int max_slots_;
  std::vector<std::optional<RequestRecord>> slots_;
  std::vector<size_t> free_slots_;  ///< LIFO.
  absl::flat_hash_map<RequestId, size_t> index_;
  /// LIFO free list; empty when the model has no recurrent layers.
  std::vector<int32_t> free_recurrent_;
};

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_REQUEST_STATE_H_
