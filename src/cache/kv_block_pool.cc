#include "inferx/cache/kv_block_pool.h"

#include <algorithm>
#include <utility>

#include "absl/strings/str_cat.h"
#include "inferx/core/shape.h"

namespace inferx {

StatusOr<KvBlockPool> KvBlockPool::Create(int64_t num_layers, int64_t num_blocks,
                                          int64_t block_size, const KvLayout& layout,
                                          DeviceId device) {
  if (num_layers <= 0 || num_blocks <= 0 || block_size <= 0) {
    return InvalidArgumentError("KV pool needs positive dimensions, got layers=", num_layers,
                                " blocks=", num_blocks, " block_size=", block_size);
  }

  if (layout.kv_heads <= 0 || layout.head_dim <= 0 || layout.entries_per_token <= 0) {
    return InvalidArgumentError("KV layout is degenerate: entries=", layout.entries_per_token,
                                " kv_heads=", layout.kv_heads, " head_dim=", layout.head_dim);
  }

  if (DataTypeIsSubByte(layout.dtype)) {
    // Sub-byte KV would need the block stride to be a bit count rather than a
    // byte count, and nothing needs that yet. FP8 KV (§6.4) is one byte and
    // works here unchanged.
    return InvalidArgumentError("KV dtype ", DataTypeName(layout.dtype),
                                " is not supported by the block pool");
  }

  KvBlockPool pool;
  pool.device_ = device;
  pool.layout_ = layout;
  pool.num_layers_ = num_layers;
  pool.num_blocks_ = num_blocks;
  pool.block_size_ = block_size;

  // One entry (K or V) of one block: block_size tokens x kv_heads x head_dim.
  const int64_t entry_elems = block_size * layout.kv_heads * layout.head_dim;
  pool.entry_stride_ = DataTypeByteSize(layout.dtype, entry_elems);

  const int64_t block_bytes = pool.entry_stride_ * layout.entries_per_token;
  pool.layer_stride_ = block_bytes * num_blocks;

  const int64_t total = pool.layer_stride_ * num_layers;

  INFERX_ASSIGN_OR_RETURN(pool.storage_,
                          DeviceBuffer::Allocate(static_cast<size_t>(total), device));

  // Descending, so that popping from the back hands out block 0 first. Only
  // cosmetic, but it makes a fresh pool's allocations readable in a dump.
  pool.free_list_.resize(static_cast<size_t>(num_blocks));
  for (int64_t i = 0; i < num_blocks; ++i) {
    pool.free_list_[static_cast<size_t>(i)] = static_cast<int32_t>(num_blocks - 1 - i);
  }

  return pool;
}

StatusOr<KvBlockPool> KvBlockPool::Create(int64_t num_blocks, int64_t block_size,
                                          const std::vector<KvLayout>& layouts,
                                          DeviceId device) {
  if (layouts.empty()) {
    return InvalidArgumentError("KV pool needs at least one layer layout");
  }
  for (const auto& layout : layouts) {
    if (layout.kv_heads <= 0 || layout.head_dim <= 0 || layout.entries_per_token <= 0) {
      return InvalidArgumentError("KV layout is degenerate: entries=", layout.entries_per_token,
                                  " kv_heads=", layout.kv_heads, " head_dim=", layout.head_dim);
    }
    if (DataTypeIsSubByte(layout.dtype)) {
      return InvalidArgumentError("KV dtype ", DataTypeName(layout.dtype),
                                  " is not supported by the block pool");
    }
  }
  // First validate through the uniform path for the shared checks, then
  // rebuild the geometry per layer.
  INFERX_ASSIGN_OR_RETURN(auto probe, Create(static_cast<int64_t>(layouts.size()), num_blocks,
                                             block_size, layouts.front(), device));
  probe.layouts_ = layouts;
  probe.layer_offsets_.assign(layouts.size() + 1, 0);
  int64_t offset = 0;
  for (size_t layer = 0; layer < layouts.size(); ++layer) {
    probe.layer_offsets_[layer] = offset;
    const int64_t entry_elems = block_size * layouts[layer].kv_heads * layouts[layer].head_dim;
    const int64_t entry_bytes = DataTypeByteSize(layouts[layer].dtype, entry_elems);
    offset += entry_bytes * layouts[layer].entries_per_token * num_blocks;
  }
  probe.layer_offsets_[layouts.size()] = offset;
  INFERX_ASSIGN_OR_RETURN(probe.storage_, DeviceBuffer::Allocate(static_cast<size_t>(offset), device));
  return probe;
}

StatusOr<int32_t> KvBlockPool::AllocateBlock() {
  if (free_list_.empty()) {
    return ResourceExhaustedError("KV pool is full: all ", num_blocks_, " blocks are in use");
  }

  const int32_t block = free_list_.back();
  free_list_.pop_back();

  return block;
}

Status KvBlockPool::FreeBlock(int32_t block) {
  if (block < 0 || block >= num_blocks_) {
    return InvalidArgumentError("block ", block, " is outside [0, ", num_blocks_, ")");
  }

  // Linear, and deliberately so: the free list is at most `num_blocks` long and
  // this only runs when a sequence finishes, not per step. Catching a double
  // free is worth far more here than the scan costs -- the failure it prevents
  // is two sequences writing to the same KV block, which surfaces as garbled
  // output from an unrelated request.
  if (std::find(free_list_.begin(), free_list_.end(), block) != free_list_.end()) {
    return InvalidArgumentError("block ", block, " is already free");
  }

  free_list_.push_back(block);
  return OkStatus();
}

Status KvBlockPool::FreeBlocks(const std::vector<int32_t>& blocks) {
  for (const int32_t b : blocks) INFERX_RETURN_IF_ERROR(FreeBlock(b));
  return OkStatus();
}

namespace {

StatusOr<Tensor> ViewAt(const DeviceBuffer& storage, int64_t offset, const Shape& shape,
                        DataType dtype, DeviceId device) {
  return Tensor::FromBlob(const_cast<std::byte*>(storage.data()) + offset, dtype, shape,
                          device);
}

}  // namespace

StatusOr<Tensor> KvBlockPool::KeyCache(int64_t layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return InvalidArgumentError("layer ", layer, " is outside [0, ", num_layers_, ")");
  }
  const KvLayout& layout = LayoutFor(layer);
  const int64_t offset = layouts_.empty() ? layer * layer_stride_ : layer_offsets_[layer];
  return ViewAt(storage_, offset,
                Shape({num_blocks_, block_size_, layout.kv_heads, layout.head_dim}),
                layout.dtype, device_);
}

StatusOr<Tensor> KvBlockPool::ValueCache(int64_t layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return InvalidArgumentError("layer ", layer, " is outside [0, ", num_layers_, ")");
  }
  const KvLayout& layout = LayoutFor(layer);
  if (layout.entries_per_token < 2) {
    return FailedPreconditionError(
        "this KV layout has ", layout.entries_per_token,
        " entry per token and therefore no separate value cache; an MLA latent "
        "is read through KeyCache()");
  }
  const int64_t entry_elems = block_size_ * layout.kv_heads * layout.head_dim;
  const int64_t entry_bytes = DataTypeByteSize(layout.dtype, entry_elems);
  const int64_t base =
      layouts_.empty() ? layer * layer_stride_ : layer_offsets_[layer];
  // K and V for a layer are adjacent: [blocks][K|V][...] would interleave them
  // per block, which is worse for the append kernel, so the entry dimension is
  // outermost within a layer.
  return ViewAt(storage_, base + entry_bytes * num_blocks_,
                Shape({num_blocks_, block_size_, layout.kv_heads, layout.head_dim}),
                layout.dtype, device_);
}

}  // namespace inferx
