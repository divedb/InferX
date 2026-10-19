#include "inferx/cache/kv_block_pool.h"

#include <algorithm>
#include <utility>

#include "inferx/core/shape.h"

namespace inferx {

namespace {

// The pool pages tokens, so only per-token components describe its geometry.
// Fixed-slot state belongs to the recurrent state pool, not to blocks.
Status ValidateComponent(const CacheComponent& component) {
  if (component.growth != CacheComponent::Growth::kPerToken) {
    return InvalidArgumentError(
        "KV pool needs a kPerToken component, got a kFixedSlot one; per-sequence "
        "state belongs to the recurrent state pool");
  }

  if (component.bytes_per_token <= 0) {
    return InvalidArgumentError("KV component is degenerate: bytes_per_token=",
                                component.bytes_per_token);
  }

  // dtype is informational (kvcache.md §6.1): storage is opaque bytes, so
  // sub-byte dtypes are fine as long as bytes_per_token already accounts for
  // packing (DataTypeByteSize rounds up).
  return OkStatus();
}

}  // namespace

StatusOr<KvBlockPool> KvBlockPool::Create(int64_t num_layers, int64_t num_blocks,
                                          int64_t block_size,
                                          const CacheComponent& component, Device device) {
  if (num_layers <= 0 || num_blocks <= 0 || block_size <= 0) {
    return InvalidArgumentError("KV pool needs positive dimensions, got layers=", num_layers,
                                " blocks=", num_blocks, " block_size=", block_size);
  }

  INFERX_RETURN_IF_ERROR(ValidateComponent(component));

  KvBlockPool pool;
  pool.device_ = device;
  pool.component_ = component;
  pool.num_layers_ = num_layers;
  pool.num_blocks_ = num_blocks;
  pool.block_size_ = block_size;

  // One block of one layer: block_size tokens x bytes_per_token bytes. The
  // null block consumes a row of its own so id->offset math stays uniform and
  // kernels can skip block-0 rows without a bounds special case.
  const int64_t block_bytes = block_size * component.bytes_per_token;
  pool.layer_stride_ = block_bytes * (num_blocks + 1);

  const int64_t total = pool.layer_stride_ * num_layers;

  INFERX_ASSIGN_OR_RETURN(pool.storage_,
                          DeviceBuffer::Allocate(static_cast<size_t>(total), device));

  pool.records_.assign(static_cast<size_t>(num_blocks) + 2, Record{});
  const int32_t sent = pool.Sentinel();
  pool.records_[static_cast<size_t>(sent)].lru_prev = sent;
  pool.records_[static_cast<size_t>(sent)].lru_next = sent;
  // Descending, so that popping from the back hands out block 1 first. Only
  // cosmetic, but it makes a fresh pool's allocations readable in a dump.
  for (int64_t id = num_blocks; id >= 1; --id) {
    pool.FreePushBack(static_cast<BlockId>(id));
  }

  return pool;
}

StatusOr<KvBlockPool> KvBlockPool::Create(int64_t num_blocks, int64_t block_size,
                                          const std::vector<CacheComponent>& components,
                                          Device device) {
  if (components.empty()) {
    return InvalidArgumentError("KV pool needs at least one layer component");
  }
  for (const auto& component : components) {
    INFERX_RETURN_IF_ERROR(ValidateComponent(component));
  }
  // First validate through the uniform path for the shared checks, then
  // rebuild the geometry per layer.
  INFERX_ASSIGN_OR_RETURN(auto probe, Create(static_cast<int64_t>(components.size()), num_blocks,
                                             block_size, components.front(), device));
  probe.components_ = components;
  probe.layer_offsets_.assign(components.size() + 1, 0);
  int64_t offset = 0;
  for (size_t layer = 0; layer < components.size(); ++layer) {
    probe.layer_offsets_[layer] = offset;
    offset += block_size * components[layer].bytes_per_token * (num_blocks + 1);
  }
  probe.layer_offsets_[components.size()] = offset;
  INFERX_ASSIGN_OR_RETURN(probe.storage_, DeviceBuffer::Allocate(static_cast<size_t>(offset), device));
  return probe;
}

// -- intrusive queues ---------------------------------------------------------

void KvBlockPool::FreePushBack(BlockId block) {
  Record& r = records_[static_cast<size_t>(block)];
  r.prev_free = free_tail_;
  r.next_free = kFree;
  if (free_tail_ != kFree) {
    records_[static_cast<size_t>(free_tail_)].next_free = block;
  } else {
    free_head_ = block;
  }
  free_tail_ = block;
  ++free_depth_;
}

BlockId KvBlockPool::FreePopBack() {
  if (free_tail_ == kFree) return kNullBlockId;
  const BlockId block = free_tail_;
  Record& r = records_[static_cast<size_t>(block)];
  free_tail_ = r.prev_free;
  if (free_tail_ != kFree) {
    records_[static_cast<size_t>(free_tail_)].next_free = kFree;
  } else {
    free_head_ = kFree;
  }
  r.prev_free = r.next_free = kFree;
  --free_depth_;
  return block;
}

void KvBlockPool::LruPushBack(BlockId block) {
  const int32_t sent = Sentinel();
  Record& r = records_[static_cast<size_t>(block)];
  const int32_t prev = records_[static_cast<size_t>(sent)].lru_prev;
  r.lru_prev = prev;
  r.lru_next = sent;
  records_[static_cast<size_t>(prev)].lru_next = block;
  records_[static_cast<size_t>(sent)].lru_prev = block;
}

void KvBlockPool::LruErase(BlockId block) {
  Record& r = records_[static_cast<size_t>(block)];
  records_[static_cast<size_t>(r.lru_prev)].lru_next = r.lru_next;
  records_[static_cast<size_t>(r.lru_next)].lru_prev = r.lru_prev;
  r.lru_prev = r.lru_next = -1;
}

BlockId KvBlockPool::LruPopFront() {
  const int32_t sent = Sentinel();
  const int32_t front = records_[static_cast<size_t>(sent)].lru_next;
  if (front == sent) return kNullBlockId;
  LruErase(front);
  return front;
}

void KvBlockPool::Reclaim(BlockId block) {
  Record& r = records_[static_cast<size_t>(block)];
  index_.erase(r.key);
  r.key = BlockKey{};
  r.cached = false;
  r.discoverable = false;
  FreePushBack(block);
}

// -- allocation and ledger ----------------------------------------------------

StatusOr<BlockId> KvBlockPool::AllocateBlock() {
  BlockId block = FreePopBack();
  if (block == kNullBlockId) {
    // Free queue empty: evict the least-recently-used cached entry. Pinned
    // blocks are never on the LRU, so victims are always safe to reuse.
    const BlockId victim = LruPopFront();
    if (victim == kNullBlockId) {
      return ResourceExhaustedError("KV pool is full: all ", num_blocks_,
                                    " blocks are in use or pinned");
    }
    index_.erase(records_[static_cast<size_t>(victim)].key);
    Record& vr = records_[static_cast<size_t>(victim)];
    vr.key = BlockKey{};
    vr.cached = false;
    vr.discoverable = false;
    block = victim;
  }

  Record& r = records_[static_cast<size_t>(block)];
  r.ref_cnt = 1;
  r.key = BlockKey{};
  r.cached = false;
  r.discoverable = false;
  r.prev_free = r.next_free = kFree;
  return block;
}

StatusOr<std::vector<BlockId>> KvBlockPool::AllocateBlocks(int64_t n) {
  if (n < 0) {
    return InvalidArgumentError("cannot allocate a negative number of blocks: ", n);
  }

  std::vector<BlockId> blocks;
  blocks.reserve(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    auto block = AllocateBlock();
    if (!block.ok()) {
      // All-or-nothing: unwind what was taken. Fresh blocks are uncached
      // with one reference, so releasing returns them to the free queue.
      const std::vector<BlockId> taken(blocks.rbegin(), blocks.rend());
      static_cast<void>(ReleaseBlocks(taken));
      return block.status();
    }
    blocks.push_back(*block);
  }
  return blocks;
}

Status KvBlockPool::AcquireBlock(BlockId block) {
  if (!Usable(block)) {
    return InvalidArgumentError("block ", block, " is outside [1, ", num_blocks_, "]");
  }

  Record& r = records_[static_cast<size_t>(block)];
  if (r.ref_cnt == 0 && !r.cached) {
    return InvalidArgumentError("block ", block, " is free");
  }

  if (r.ref_cnt == 0) LruErase(block);  // idle cached entry becomes pinned
  ++r.ref_cnt;
  return OkStatus();
}

Status KvBlockPool::ReleaseBlock(BlockId block) {
  if (!Usable(block)) {
    return InvalidArgumentError("block ", block, " is outside [1, ", num_blocks_, "]");
  }
  if (block == kNullBlockId) {
    return InvalidArgumentError("cannot release the null block");
  }

  Record& r = records_[static_cast<size_t>(block)];
  if (r.ref_cnt == 0) {
    return InvalidArgumentError("block ", block, " holds no reference");
  }

  --r.ref_cnt;
  if (r.ref_cnt > 0) return OkStatus();

  // Last reference gone. Discoverable entries stay in the index and become
  // eviction capacity; tombstones and uncached blocks go back to the pool.
  if (r.discoverable) {
    LruPushBack(block);
  } else {
    r.key = BlockKey{};
    r.cached = false;
    FreePushBack(block);
  }
  return OkStatus();
}

Status KvBlockPool::ReleaseBlocks(const std::vector<BlockId>& blocks) {
  for (const BlockId b : blocks) INFERX_RETURN_IF_ERROR(ReleaseBlock(b));
  return OkStatus();
}

int64_t KvBlockPool::EvictCached(int64_t count) {
  int64_t evicted = 0;
  for (; evicted < count; ++evicted) {
    const BlockId victim = LruPopFront();
    if (victim == kNullBlockId) break;
    Reclaim(victim);
  }
  return evicted;
}

uint32_t KvBlockPool::RefCount(BlockId block) const {
  return Usable(block) ? records_[static_cast<size_t>(block)].ref_cnt : 0;
}

// -- prefix index --------------------------------------------------------------

void KvBlockPool::PublishBlock(const BlockKey& key, BlockId block) {
  if (!Usable(block)) return;
  Record& r = records_[static_cast<size_t>(block)];
  if (r.ref_cnt == 0) return;  // only a holder can publish

  if (const auto it = index_.find(key); it != index_.end()) {
    if (it->second == block && r.discoverable) return;  // re-publish: no-op
    // The key moves to this block; the previous one loses its entry.
    Record& old = records_[static_cast<size_t>(it->second)];
    if (old.discoverable) {
      // Displaced: no entry and no tombstone, whether held or idle.
      old.discoverable = false;
      old.key = BlockKey{};
      old.cached = false;
      if (old.ref_cnt == 0) {
        LruErase(it->second);
        FreePushBack(it->second);
      }
    }
  }

  r.key = key;
  r.cached = true;
  r.discoverable = true;
  index_[key] = block;
}

std::optional<BlockId> KvBlockPool::LookupCached(const BlockKey& key) const {
  const auto it = index_.find(key);
  if (it == index_.end()) return std::nullopt;
  const Record& r = records_[static_cast<size_t>(it->second)];
  if (!r.discoverable) return std::nullopt;
  return it->second;
}

void KvBlockPool::ForgetBlock(BlockId block) {
  if (!Usable(block)) return;
  Record& r = records_[static_cast<size_t>(block)];
  if (!r.cached) return;

  index_.erase(r.key);
  r.discoverable = false;
  if (r.ref_cnt == 0) {
    // Nothing holds it and it can no longer be found: reclaim now.
    LruErase(block);
    r.key = BlockKey{};
    r.cached = false;
    FreePushBack(block);
  }
}

void KvBlockPool::ClearCachedEntries() {
  for (int64_t id = 1; id <= num_blocks_; ++id) {
    Record& r = records_[static_cast<size_t>(id)];
    if (!r.cached) continue;
    if (r.ref_cnt == 0) {
      // Evictable: reclaim immediately.
      LruErase(static_cast<BlockId>(id));
      r.key = BlockKey{};
      r.cached = false;
      r.discoverable = false;
      FreePushBack(static_cast<BlockId>(id));
    } else {
      // Pinned: ownership-only tombstone; it frees when the last holder
      // releases (ReleaseBlock sends undiscoverable blocks to the free queue).
      r.discoverable = false;
    }
  }
  index_.clear();
}

Status KvBlockPool::ResetPrefixCache() {
  // No tombstones: while anything is held the entries must stay live, so
  // draining requests first is the caller's job.
  for (int64_t id = 1; id <= num_blocks_; ++id) {
    if (records_[static_cast<size_t>(id)].ref_cnt > 0) {
      return FailedPreconditionError("cannot reset the prefix cache: block ", id,
                                     " is still held");
    }
  }
  ClearCachedEntries();
  return OkStatus();
}

bool KvBlockPool::Contains(BlockId block) const {
  return Usable(block) && records_[static_cast<size_t>(block)].cached;
}

int64_t KvBlockPool::NumCachedBlocks() const {
  int64_t count = 0;
  for (int64_t id = 1; id <= num_blocks_; ++id) {
    if (records_[static_cast<size_t>(id)].discoverable) ++count;
  }
  return count;
}

int64_t KvBlockPool::EvictableBlocks() const {
  int64_t count = 0;
  for (int64_t id = 1; id <= num_blocks_; ++id) {
    const Record& r = records_[static_cast<size_t>(id)];
    if (r.discoverable && r.ref_cnt == 0) ++count;
  }
  return count;
}

// -- kernel views ---------------------------------------------------------------

namespace {

StatusOr<Tensor> ViewAt(const DeviceBuffer& storage, int64_t offset, const Shape& shape,
                        DataType dtype, Device device) {
  return Tensor::FromBlob(const_cast<std::byte*>(storage.data()) + offset, dtype, shape,
                          device);
}

}  // namespace

StatusOr<Tensor> KvBlockPool::LayerCache(int64_t layer) const {
  if (layer < 0 || layer >= num_layers_) {
    return InvalidArgumentError("layer ", layer, " is outside [0, ", num_layers_, ")");
  }
  const int64_t offset = components_.empty() ? layer * layer_stride_ : layer_offsets_[layer];
  const CacheComponent& component = ComponentFor(layer);
  // uint8 with bytes_per_token as the innermost extent: the raw byte region
  // the owning layer reshapes into kernel geometry (Tensor::Reshape/Bitcast).
  // One row per block id, the null block's included.
  return ViewAt(storage_, offset,
                Shape({num_blocks_ + 1, block_size_, component.bytes_per_token}),
                DataType::kUInt8, device_);
}

}  // namespace inferx
