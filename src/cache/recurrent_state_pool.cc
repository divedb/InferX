#include "inferx/cache/recurrent_state_pool.h"

#include <algorithm>
#include <utility>

#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"

namespace inferx {

StatusOr<RecurrentStatePool> RecurrentStatePool::Create(const CacheGroupSpec& group,
                                                        int64_t max_slots, Device device) {
  if (max_slots <= 0) {
    return InvalidArgumentError("recurrent state pool needs a positive max_slots, got ",
                                max_slots);
  }
  if (group.layers.empty()) {
    return InvalidArgumentError("recurrent state pool needs a non-empty group");
  }
  if (!std::holds_alternative<RecurrentRetention>(group.spec.retention)) {
    return InvalidArgumentError("group ", group.group_index, " is not recurrent");
  }
  for (const CacheComponent& component : group.spec.components) {
    if (component.growth != CacheComponent::Growth::kFixedSlot) {
      return InvalidArgumentError("recurrent group ", group.group_index,
                                  " has a kPerToken component; recurrent state is fixed-slot");
    }
    if (component.bytes_per_slot <= 0) {
      return InvalidArgumentError("recurrent component is degenerate: bytes_per_slot=",
                                  component.bytes_per_slot);
    }
  }

  RecurrentStatePool pool;
  pool.device_ = device;
  pool.group_ = group;
  pool.max_slots_ = max_slots;
  pool.slot_bytes_ = group.NumLayers() * group.spec.StateBytesPerLayer();

  // [layer position][component][slot][bytes_per_slot]: one contiguous
  // [max_slots, bytes_per_slot] region per (position, component), row-major.
  const int64_t num_components = pool.NumComponents();
  const size_t num_regions =
      static_cast<size_t>(group.NumLayers()) * static_cast<size_t>(num_components);
  pool.region_offsets_.assign(num_regions, 0);
  int64_t offset = 0;
  for (int64_t position = 0; position < group.NumLayers(); ++position) {
    for (int64_t component = 0; component < num_components; ++component) {
      pool.region_offsets_[static_cast<size_t>(position * num_components + component)] = offset;
      offset += max_slots * group.spec.components[static_cast<size_t>(component)].bytes_per_slot;
    }
  }

  INFERX_ASSIGN_OR_RETURN(pool.storage_,
                          DeviceBuffer::Allocate(static_cast<size_t>(offset), device));

  // Model layer -> position within the group; -1 for layers it does not own.
  const int64_t max_layer = *std::max_element(group.layers.begin(), group.layers.end());
  pool.layer_position_.assign(static_cast<size_t>(max_layer) + 1, -1);
  for (int64_t position = 0; position < group.NumLayers(); ++position) {
    pool.layer_position_[static_cast<size_t>(group.layers[static_cast<size_t>(position)])] =
        static_cast<int32_t>(position);
  }

  // Descending, so popping from the back hands out slot 0 first.
  pool.free_slots_.reserve(static_cast<size_t>(max_slots));
  pool.in_use_.assign(static_cast<size_t>(max_slots), 0);
  for (int64_t slot = max_slots - 1; slot >= 0; --slot) {
    pool.free_slots_.push_back(slot);
  }

  return pool;
}

StatusOr<int64_t> RecurrentStatePool::AllocateSlot() {
  if (free_slots_.empty()) {
    return ResourceExhaustedError("state pool is full: all ", max_slots_, " slots are in use");
  }

  const int64_t slot = free_slots_.back();
  free_slots_.pop_back();
  in_use_[static_cast<size_t>(slot)] = 1;
  return slot;
}

Status RecurrentStatePool::ReleaseSlot(int64_t slot) {
  if (slot < 0 || slot >= max_slots_) {
    return InvalidArgumentError("slot ", slot, " is outside [0, ", max_slots_, ")");
  }
  if (!in_use_[static_cast<size_t>(slot)]) {
    return InvalidArgumentError("slot ", slot, " is already free");
  }

  in_use_[static_cast<size_t>(slot)] = 0;
  free_slots_.push_back(slot);
  return OkStatus();
}

StatusOr<Tensor> RecurrentStatePool::ComponentView(int64_t layer, int64_t component) const {
  const int64_t position = PositionOfLayer(layer);
  if (position < 0) {
    return InvalidArgumentError("layer ", layer, " does not belong to recurrent group ",
                                group_.group_index);
  }
  if (component < 0 || component >= NumComponents()) {
    return InvalidArgumentError("component ", component, " is outside [0, ", NumComponents(),
                                ")");
  }

  const CacheComponent& spec =
      group_.spec.components[static_cast<size_t>(component)];
  return Tensor::FromBlob(
      const_cast<std::byte*>(storage_.data()) +
          RegionOffset(position, component),
      DataType::kUInt8, Shape({max_slots_, spec.bytes_per_slot}), device_);
}

Status RecurrentStatePool::ResetSlot(ops::OpContext& ctx, int64_t slot) {
  if (slot < 0 || slot >= max_slots_) {
    return InvalidArgumentError("slot ", slot, " is outside [0, ", max_slots_, ")");
  }
  if (!in_use_[static_cast<size_t>(slot)]) {
    return InvalidArgumentError("slot ", slot, " is not allocated");
  }

  // A fresh sequence must start from zero state, and zeroing needs a stream;
  // stream ordering makes this safe to call right before the reading step.
  const int64_t num_components = NumComponents();
  for (int64_t position = 0; position < NumLayers(); ++position) {
    for (int64_t component = 0; component < num_components; ++component) {
      const CacheComponent& spec =
          group_.spec.components[static_cast<size_t>(component)];
      const int64_t offset =
          RegionOffset(position, component) + slot * spec.bytes_per_slot;
      INFERX_RETURN_IF_ERROR(ctx.Runtime().Memory().MemsetAsync(
          storage_.data() + offset, 0, static_cast<size_t>(spec.bytes_per_slot),
          ctx.GetStream()));
    }
  }
  return OkStatus();
}

int64_t RecurrentStatePool::PositionOfLayer(int64_t layer) const {
  if (layer < 0 || layer >= static_cast<int64_t>(layer_position_.size())) return -1;
  return layer_position_[static_cast<size_t>(layer)];
}

}  // namespace inferx
