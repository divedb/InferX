#ifndef INFERX_MODELS_SCRATCH_ARENA_H_
#define INFERX_MODELS_SCRATCH_ARENA_H_

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/allocator.h"
#include "inferx/core/caching_allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/status.h"

namespace inferx {
namespace models {

/// \brief The model layer's step workspace: measured, then frozen.
///
/// Sizing scratch by formula couples the runner to every family's buffer
/// count; this arena replaces the formula with a measurement. Start in
/// measuring mode: allocations are backed by growable device chunks while
/// the arena records the high-water mark -- the bytes one contiguous
/// region would have needed to serve the same allocation sequence,
/// alignment included. After a profile step at the largest shape the
/// scheduler can produce, Freeze() swaps the backing for a single
/// contiguous region of `high_water * (1 + margin)` bytes and serves every
/// later allocation from it.
///
/// Blocks are recycled through a free list in both phases, so a step's
/// per-layer buffers reuse each other's memory. Because a step executes
/// one deterministic allocation/free sequence, addresses are a pure
/// function of that sequence once frozen -- the stability CUDA-graph
/// replay needs.
///
/// Not thread-safe; one arena belongs to one runner's step loop, and
/// Reset() requires the stream to have drained.
class ScratchArena final : public Allocator {
 public:
  /// \brief An arena that measures while it serves, growing as needed on
  ///        `device` (the frozen region lands there too).
  static ScratchArena Measuring(Device device);

  ScratchArena(ScratchArena&& other) noexcept;
  ScratchArena& operator=(ScratchArena&& other) noexcept;
  ScratchArena(const ScratchArena&) = delete;
  ScratchArena& operator=(const ScratchArena&) = delete;
  ~ScratchArena() override;

  /// \brief Replaces the measuring backing with one contiguous region of
  ///        `high_water * (1 + margin)` bytes on `device`.
  ///
  /// Must be called in measuring mode, after the profiling workload, with
  /// every measured allocation already freed. After Freeze the arena never
  /// grows; an allocation that does not fit returns ResourceExhausted.
  Status Freeze(Device device, double margin);

  /// \brief The measured bytes one contiguous region must hold.
  int64_t high_water_bytes() const { return high_water_bytes_; }

  /// \brief Returns every block to the free list.
  ///
  /// Only legal between steps, after the stream has completed: outstanding
  /// tensors would dangle. The ordinary alloc/free cycle already recycles
  /// per-step buffers; this is the explicit reset for capture and tests.
  Status Reset();

  bool frozen() const { return frozen_; }

  // Allocator interface.
  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override;
  /// The alignment-defaulting overload lives on the base; re-expose it so
  /// callers of the concrete type see both.
  using Allocator::Allocate;
  Status Deallocate(void* ptr) override;
  Device GetDevice() const override;
  std::string_view Name() const override { return "scratch-arena"; }

 private:
  ScratchArena() = default;

  /// One growable chunk of the measuring phase: a device buffer plus its
  /// block list. Chunks (and blocks within them) are 256-byte aligned.
  struct Chunk {
    DeviceBuffer buffer;
    /// offset, size, free -- sorted by offset for coalescing.
    struct Block {
      size_t offset;
      size_t size;
      bool free;
    };
    std::vector<Block> blocks;
  };

  StatusOr<void*> AllocateMeasured(size_t bytes, size_t alignment);
  StatusOr<void*> AllocateFrozen(size_t bytes, size_t alignment);
  Status Grow(size_t bytes);

  bool frozen_ = false;
  Device device_{};
  /// Measuring-phase chunks; empty once frozen.
  std::vector<Chunk> chunks_;
  /// Live allocations -> (chunk index, block offset). Offsets, not block
  /// indices: coalescing erases blocks and would shift stale indices.
  absl::flat_hash_map<void*, std::pair<size_t, size_t>> live_;
  /// Total device bytes the measuring chunks reached; the freeze size is
  /// derived from it, so the frozen region is never smaller than what the
  /// same sequence consumed.
  int64_t high_water_bytes_ = 0;
  /// Frozen-phase allocator over one contiguous region.
  std::unique_ptr<CachingAllocator> frozen_allocator_;
  std::unique_ptr<DeviceBuffer> frozen_region_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_SCRATCH_ARENA_H_
