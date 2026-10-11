#include "inferx/models/scratch_arena.h"

#include <algorithm>
#include <utility>

#include "absl/container/flat_hash_map.h"

namespace inferx {
namespace models {

namespace {

/// First chunk the measuring phase grabs; later chunks at least double, so
/// the chunk count stays logarithmic in the workload.
constexpr size_t kFirstChunkBytes = 32u << 20;  // 32 MiB

size_t AlignUp(size_t bytes, size_t alignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

}  // namespace

ScratchArena ScratchArena::Measuring(Device device) {
  ScratchArena arena;
  arena.device_ = device;
  return arena;
}

ScratchArena::ScratchArena(ScratchArena&& other) noexcept = default;
ScratchArena& ScratchArena::operator=(ScratchArena&& other) noexcept = default;

ScratchArena::~ScratchArena() = default;

Status ScratchArena::Grow(size_t bytes) {
  // Chunk sizes grow with demand; a single huge request gets its own chunk.
  size_t want = chunks_.empty() ? kFirstChunkBytes
                                : chunks_.back().buffer.size() * 2;
  want = std::max(want, AlignUp(bytes, kTensorAlignment));
  Chunk chunk;
  INFERX_ASSIGN_OR_RETURN(chunk.buffer, DeviceBuffer::Allocate(want, device_));
  chunk.blocks.push_back({0, chunk.buffer.size(), true});
  chunks_.push_back(std::move(chunk));
  high_water_bytes_ += static_cast<int64_t>(want);
  return OkStatus();
}

StatusOr<void*> ScratchArena::AllocateMeasured(size_t bytes, size_t alignment) {
  if (bytes == 0) return static_cast<void*>(nullptr);
  // Carves are rounded to the requested alignment (and the tensor floor),
  // so every block offset stays aligned however the splits stack up.
  const size_t alignment_floor = std::max(alignment, kTensorAlignment);
  const size_t carve = AlignUp(bytes, alignment_floor);
  for (size_t c = 0; c < chunks_.size(); ++c) {
    Chunk& chunk = chunks_[c];
    for (size_t b = 0; b < chunk.blocks.size(); ++b) {
      // Splitting below can reallocate blocks; keep a value, never a
      // reference into the vector, while constructing the allocation.
      const Chunk::Block block = chunk.blocks[b];
      if (!block.free || block.size < carve) continue;
      const size_t slack = block.size - carve;
      const size_t offset = block.offset;
      chunk.blocks[b].free = false;
      chunk.blocks[b].size = carve;
      void* ptr = chunk.buffer.data() + offset;
      live_[ptr] = {c, offset};
      if (slack >= alignment_floor) {
        chunk.blocks.insert(chunk.blocks.begin() + static_cast<long>(b) + 1,
                            {offset + carve, slack, true});
      }
      return ptr;
    }
  }
  INFERX_RETURN_IF_ERROR(Grow(carve));
  return AllocateMeasured(bytes, alignment);
}

StatusOr<void*> ScratchArena::AllocateFrozen(size_t bytes, size_t alignment) {
  if (frozen_allocator_ == nullptr) {
    return FailedPreconditionError("scratch arena allocation before Freeze");
  }
  return frozen_allocator_->Allocate(bytes, alignment);
}

StatusOr<void*> ScratchArena::Allocate(size_t bytes, size_t alignment) {
  if (frozen_) return AllocateFrozen(bytes, alignment);
  return AllocateMeasured(bytes, alignment);
}

Status ScratchArena::Deallocate(void* ptr) {
  if (ptr == nullptr) return OkStatus();
  if (frozen_) return frozen_allocator_->Deallocate(ptr);

  const auto it = live_.find(ptr);
  if (it == live_.end()) {
    return InvalidArgumentError("scratch arena asked to free a foreign pointer");
  }
  const auto [chunk_index, block_offset] = it->second;
  live_.erase(it);
  Chunk& chunk = chunks_[chunk_index];
  // Blocks stay sorted by offset; locate this one without trusting any
  // index captured before earlier coalesces.
  auto& blocks = chunk.blocks;
  size_t b = 0;
  while (b < blocks.size() && blocks[b].offset != block_offset) ++b;
  if (b == blocks.size()) {
    return InternalError("scratch arena block bookkeeping lost an allocation");
  }
  blocks[b].free = true;
  // Coalesce neighbours within the chunk, right then left.
  if (b + 1 < blocks.size() && blocks[b + 1].free) {
    blocks[b].size += blocks[b + 1].size;
    blocks.erase(blocks.begin() + static_cast<long>(b) + 1);
  }
  if (b > 0 && blocks[b - 1].free) {
    blocks[b - 1].size += blocks[b].size;
    blocks.erase(blocks.begin() + static_cast<long>(b));
  }
  return OkStatus();
}

Device ScratchArena::GetDevice() const { return device_; }

Status ScratchArena::Freeze(Device device, double margin) {
  if (frozen_) return FailedPreconditionError("scratch arena is already frozen");
  if (!live_.empty()) {
    return FailedPreconditionError("cannot freeze: ", live_.size(),
                                   " scratch allocations are still live");
  }
  if (margin < 0.0) {
    return InvalidArgumentError("freeze margin must be non-negative, got ", margin);
  }

  const int64_t region_bytes =
      static_cast<int64_t>(static_cast<double>(high_water_bytes_) * (1.0 + margin));
  auto region = std::make_unique<DeviceBuffer>();
  INFERX_ASSIGN_OR_RETURN(*region,
                          DeviceBuffer::Allocate(static_cast<size_t>(region_bytes), device));
  auto allocator = std::make_unique<CachingAllocator>(
      region->data(), region->size(), device, kTensorAlignment);

  // The measuring chunks die here; the frozen region takes over.
  chunks_.clear();
  live_.clear();
  frozen_region_ = std::move(region);
  frozen_allocator_ = std::move(allocator);
  device_ = device;
  frozen_ = true;
  return OkStatus();
}

Status ScratchArena::Reset() {
  if (frozen_) {
    frozen_allocator_->Reset();
    return OkStatus();
  }
  for (Chunk& chunk : chunks_) {
    chunk.blocks.clear();
    chunk.blocks.push_back({0, chunk.buffer.size(), true});
  }
  live_.clear();
  return OkStatus();
}

}  // namespace models
}  // namespace inferx
