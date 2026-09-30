#ifndef INFERX_CACHE_RECURRENT_STATE_POOL_H_
#define INFERX_CACHE_RECURRENT_STATE_POOL_H_

#include <cstdint>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/state.h"
#include "inferx/ops/op_context.h"

namespace inferx {

/// \brief Fixed-size pool of recurrent mixer state, one region per layer.
///
/// Mirrors KvBlockPool's lifecycle rules: one allocation at startup, slots
/// handed out by the scheduler-adjacent owner, never grown. Each slot holds
/// one sequence's per-layer state: the delta-rule matrix
/// [value_heads, key_dim, value_dim] float32 and the causal-conv input ring
/// [conv_dim, kernel - 1]. Slots are assigned per request and reset (zeroed,
/// stream-ordered) when (re)assigned, since a fresh sequence must start from
/// zero state; paged KV needs no reset because it is fully overwritten.
class RecurrentStatePool {
 public:
  /// \brief Allocates state for every recurrent layer.
  ///
  /// \param specs  One entry per recurrent layer, in model layer order.
  /// \param max_slots  Concurrent sequences the pool can serve.
  /// \param device  Where the state lives.
  static StatusOr<RecurrentStatePool> Create(const std::vector<RecurrentStateSpec>& specs,
                                             int64_t max_slots, DeviceId device);

  /// \brief The delta-rule state of one layer: [slots, v, dk, dv] float32.
  StatusOr<Tensor> State(int64_t layer) const;

  /// \brief The conv-input ring of one layer: [slots, conv_dim, k-1] float32.
  StatusOr<Tensor> ConvState(int64_t layer) const;

  /// \brief Zeroes one slot's state across every layer, on the given stream.
  ///
  /// Called when a slot is assigned to a new request, before the step's
  /// kernels read it; stream ordering makes that safe under graph capture.
  Status ResetSlot(ops::OpContext& ctx, int64_t slot);

  int64_t NumLayers() const { return num_layers_; }
  int64_t MaxSlots() const { return max_slots_; }
  size_t Bytes() const { return storage_.size(); }

 private:
  RecurrentStatePool() = default;

  DeviceBuffer storage_;
  std::vector<RecurrentStateSpec> specs_;
  int64_t num_layers_ = 0;
  int64_t max_slots_ = 0;
  /// Byte offset of each layer's region within the allocation.
  std::vector<int64_t> layer_offsets_;
  DeviceId device_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_RECURRENT_STATE_POOL_H_
