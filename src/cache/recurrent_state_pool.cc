#include "inferx/cache/recurrent_state_pool.h"

#include <numeric>

#include "absl/strings/str_cat.h"
#include "inferx/core/shape.h"
#include "inferx/ops/execution_context.h"

namespace inferx {

StatusOr<RecurrentStatePool> RecurrentStatePool::Create(
    const std::vector<RecurrentStateSpec>& specs, int64_t max_slots, DeviceId device) {
  if (specs.empty() || max_slots <= 0) {
    return InvalidArgumentError("recurrent pool needs layers and positive slots");
  }
  for (const auto& spec : specs) {
    if (spec.key_heads <= 0 || spec.value_heads <= 0 || spec.key_dim <= 0 ||
        spec.value_dim <= 0 || spec.conv_kernel_size <= 0 ||
        spec.value_heads % spec.key_heads != 0) {
      return InvalidArgumentError("degenerate recurrent state geometry");
    }
  }
  RecurrentStatePool pool;
  pool.specs_ = specs;
  pool.num_layers_ = static_cast<int64_t>(specs.size());
  pool.max_slots_ = max_slots;
  pool.device_ = device;
  pool.layer_offsets_.reserve(specs.size());

  // Per layer: delta-rule state then the conv ring, slot-major so one
  // slot's reset touches contiguous bytes per region.
  int64_t offset = 0;
  for (const auto& spec : specs) {
    pool.layer_offsets_.push_back(offset);
    const int64_t state_elems = spec.value_heads * spec.key_dim * spec.value_dim;
    const int64_t conv_elems =
        (2 * spec.key_heads * spec.key_dim + spec.value_heads * spec.value_dim) *
        (spec.conv_kernel_size - 1);
    offset += max_slots * (state_elems + conv_elems) * static_cast<int64_t>(sizeof(float));
  }
  INFERX_ASSIGN_OR_RETURN(pool.storage_,
                          DeviceBuffer::Allocate(static_cast<size_t>(offset), device));
  return pool;
}

namespace {

StatusOr<Tensor> ViewAt(const DeviceBuffer& storage, int64_t offset, const Shape& shape,
                        DeviceId device) {
  return Tensor::FromBlob(const_cast<std::byte*>(storage.data()) + offset, DataType::kFloat32,
                          shape, device);
}

}  // namespace

StatusOr<Tensor> RecurrentStatePool::State(int64_t layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return InvalidArgumentError("recurrent layer ", layer, " is out of range");
  }
  const auto& spec = specs_[layer];
  return ViewAt(storage_, layer_offsets_[layer],
                Shape({max_slots_, spec.value_heads, spec.key_dim, spec.value_dim}), device_);
}

StatusOr<Tensor> RecurrentStatePool::ConvState(int64_t layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return InvalidArgumentError("recurrent layer ", layer, " is out of range");
  }
  const auto& spec = specs_[layer];
  const int64_t conv_dim =
      2 * spec.key_heads * spec.key_dim + spec.value_heads * spec.value_dim;
  const int64_t state_bytes = max_slots_ * spec.value_heads * spec.key_dim *
                              spec.value_dim * static_cast<int64_t>(sizeof(float));
  return ViewAt(storage_, layer_offsets_[layer] + state_bytes,
                Shape({max_slots_, conv_dim, spec.conv_kernel_size - 1}), device_);
}

Status RecurrentStatePool::ResetSlot(ops::ExecutionContext& ctx, int64_t slot) {
  if (slot < 0 || slot >= max_slots_) {
    return InvalidArgumentError("recurrent slot ", slot, " is out of range");
  }
  int64_t offset = 0;
  for (size_t layer = 0; layer < specs_.size(); ++layer) {
    const auto& spec = specs_[layer];
    const int64_t state_elems = spec.value_heads * spec.key_dim * spec.value_dim;
    const int64_t conv_elems =
        (2 * spec.key_heads * spec.key_dim + spec.value_heads * spec.value_dim) *
        (spec.conv_kernel_size - 1);
    const int64_t bytes = (state_elems + conv_elems) * static_cast<int64_t>(sizeof(float));
    // cudaMemsetAsync per region per slot; slots are rare (per admission).
    auto* base = const_cast<std::byte*>(storage_.data()) + offset +
                 slot * bytes;
    INFERX_RETURN_IF_ERROR(ctx.runtime().MemsetAsync(base, bytes, ctx.stream()));
    offset += max_slots_ * bytes;
  }
  return OkStatus();
}

}  // namespace inferx
