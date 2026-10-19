#ifndef INFERX_CACHE_RECURRENT_STATE_POOL_H_
#define INFERX_CACHE_RECURRENT_STATE_POOL_H_

#include <cstdint>
#include <vector>

#include "inferx/cache/cache_spec.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"

namespace inferx {

/// \brief Fixed-size pool of per-sequence recurrent state, one slot per
/// concurrent sequence.
///
/// Built from the recurrent group of a CachePlan. The pool stores bytes and
/// never interprets them: what a component means (a delta-rule matrix, a
/// causal-conv ring, an SSM state) and its dtype are the model's business, so
/// no mixer-specific geometry lives here. Like KvBlockPool it is allocated
/// once at startup and never grows.
///
/// Memory layout: `[layer position][component][slot][bytes_per_slot]`. Each
/// component of each layer is one contiguous [max_slots, bytes_per_slot]
/// region, so kernels index a sequence's state by slot with a fixed stride.
/// "Layer position" is the layer's index within `group.layers`.
///
/// Slots are handed out by AllocateSlot and returned by ReleaseSlot. Contents
/// are not cleared on allocation, because clearing needs a stream: the owner
/// calls ResetSlot before the step that reads the slot, since a fresh sequence
/// must start from zero state. (Paged KV needs no reset because it is fully
/// overwritten.)
///
/// Not thread-safe: call from the scheduler thread only.
class RecurrentStatePool {
 public:
  /// \brief Allocates state for every layer of a recurrent group.
  ///
  /// \param group      `CachePlan::RecurrentGroup()`. Its spec must use
  ///                   RecurrentRetention with fixed-slot components only.
  ///                   Copied; the plan need not outlive the pool.
  /// \param max_slots  Concurrent sequences the pool can serve.
  /// \param device     Where the state lives.
  /// \return The pool; InvalidArgument for a non-recurrent or empty group or
  ///         a non-positive `max_slots`; ResourceExhausted if it does not fit.
  static StatusOr<RecurrentStatePool> Create(const CacheGroupSpec& group, int64_t max_slots,
                                             Device device);

  // -- slot ledger ----------------------------------------------------------

  /// \brief Takes a free slot. Its contents are stale until ResetSlot runs.
  /// \return ResourceExhausted when all slots are in use; callers preempt.
  StatusOr<int64_t> AllocateSlot();

  /// \brief Returns `slot` to the free list. Releasing a slot that is not in
  /// use is an error (double free).
  Status ReleaseSlot(int64_t slot);

  // -- kernel views ----------------------------------------------------------

  /// \brief One component of one layer as a [max_slots, bytes_per_slot] uint8
  /// view. The owning model layer reshapes or bitcasts it into kernel
  /// geometry; the pool does not interpret the bytes. `layer` is a model layer
  /// index belonging to the group, and `component` indexes
  /// `group().spec.components`.
  StatusOr<Tensor> ComponentView(int64_t layer, int64_t component) const;

  /// \brief Zeroes one slot's state across every layer and component, on the
  /// context's stream. Call when a slot is assigned to a new request, before
  /// the step's kernels read it; stream ordering makes that safe under graph
  /// capture. The slot must be allocated.
  Status ResetSlot(ops::OpContext& ctx, int64_t slot);

  // -- geometry --------------------------------------------------------------

  const CacheGroupSpec& group() const { return group_; }
  int64_t NumLayers() const { return group_.NumLayers(); }
  int64_t NumComponents() const { return static_cast<int64_t>(group_.spec.components.size()); }
  int64_t MaxSlots() const { return max_slots_; }

  /// \brief Bytes one sequence's state occupies across every layer and
  /// component.
  int64_t SlotBytes() const { return slot_bytes_; }
  /// \brief Total device bytes held.
  int64_t Bytes() const { return slot_bytes_ * max_slots_; }

  int64_t NumFreeSlots() const { return static_cast<int64_t>(free_slots_.size()); }
  int64_t NumUsedSlots() const { return max_slots_ - NumFreeSlots(); }

  /// \brief Position of model layer `layer` within `group().layers`, or -1 if
  /// the layer is not in the group.
  int64_t PositionOfLayer(int64_t layer) const;

 private:
  RecurrentStatePool() = default;

  /// Byte offset of one component region inside the allocation.
  int64_t RegionOffset(int64_t position, int64_t component) const {
    return region_offsets_[static_cast<size_t>(position * NumComponents() + component)];
  }

  Device device_{};
  DeviceBuffer storage_;
  CacheGroupSpec group_;
  int64_t max_slots_ = 0;
  int64_t slot_bytes_ = 0;

  /// Offset of each (layer position, component) region, row-major.
  std::vector<int64_t> region_offsets_;
  /// Model layer index -> position in `group_.layers`, or -1.
  std::vector<int32_t> layer_position_;

  /// Free-slot stack and an in-use table for O(1) double-free detection.
  std::vector<int64_t> free_slots_;
  std::vector<uint8_t> in_use_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_RECURRENT_STATE_POOL_H_