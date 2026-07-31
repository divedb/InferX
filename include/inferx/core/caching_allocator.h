#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/allocator.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx {

/// \brief A caching allocator that manages a pre-allocated memory region.
///
/// The caching allocator maintains a free list of memory blocks and reuses
/// freed blocks to satisfy future allocation requests. It does not own the
/// memory region; the caller must ensure that the backing region outlives the
/// allocator. The allocator is not thread-safe; callers must synchronize access
/// if multiple threads may allocate from the same allocator concurrently.
class CachingAllocator final : public Allocator {
 public:
  /// \brief Statistics for the caching allocator.
  struct Stats {
    /// The total capacity of the allocator in bytes.
    size_t capacity = 0;

    /// The total number of bytes currently in use by allocated blocks.
    size_t in_use = 0;

    /// The peak number of bytes in use by allocated blocks since the allocator
    /// was created or last reset.
    size_t peak_in_use = 0;

    /// The size of the largest free block in bytes.
    size_t largest_free_block = 0;

    /// The number of free blocks currently available in the allocator.
    size_t free_block_count = 0;

    /// The number of allocated blocks currently in use by the allocator.
    uint64_t num_allocations = 0;

    /// The number of deallocations performed by the allocator.
    uint64_t num_deallocations = 0;

    /// The number of times a free block was split to satisfy an allocation
    uint64_t num_splits = 0;

    /// The number of times two adjacent free blocks were coalesced into a
    /// single free block.
    uint64_t num_coalesces = 0;

    /// The number of allocation failures due to insufficient free memory or
    /// fragmentation.
    uint64_t num_failures = 0;

    /// \brief Computes the fragmentation of the allocator.
    ///
    /// \return A value between 0.0 and 1.0 representing the fragmentation
    ///         level.
    double Fragmentation() const;
  };

  /// \brief Default constructor.
  CachingAllocator() = default;

  /// \brief Constructs a caching allocator with a pre-allocated memory region.
  ///
  /// \param base      Pointer to the pre-allocated memory region.
  /// \param capacity  The size of the memory region in bytes.
  /// \param device    The device ID associated with the allocator.
  /// \param alignment The alignment requirement for allocations.
  CachingAllocator(std::byte* base, size_t capacity, DeviceId device,
                   size_t alignment = kTensorAlignment);

  CachingAllocator(const CachingAllocator&) = delete;
  CachingAllocator& operator=(const CachingAllocator&) = delete;
  CachingAllocator(CachingAllocator&&) = default;
  CachingAllocator& operator=(CachingAllocator&&) = default;

  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override;

  // Declaring the override above would otherwise hide the base's one-argument
  // overload for anyone holding a CachingAllocator by its concrete type.
  using Allocator::Allocate;

  /// \brief Allocates a TensorView with the specified TensorSpec.
  ///
  /// \param spec The TensorSpec describing the desired data type and shape of
  ///             the tensor.
  /// \return     A StatusOr containing the allocated TensorView or an error
  ///             status.
  StatusOr<TensorView> AllocateView(const TensorSpec& spec);

  /// \brief Deallocates a previously allocated memory region.
  ///
  /// \param ptr Pointer to the memory region to deallocate.
  /// \return    A status indicating the success or failure of the deallocation.
  Status Deallocate(void* ptr) override;

  /// \brief Resets the allocator, releasing all allocated blocks and returning
  ///        to a single free block.
  void Reset();

  Stats GetStats() const;
  size_t Capacity() const { return capacity_; }

  /// \brief Returns the alignment requirement for allocations.
  ///
  /// \return The alignment requirement for allocations in bytes.
  size_t Granularity() const { return alignment_; }

  DeviceId Device() const override { return device_; }
  std::string_view Name() const override { return "workspace"; }

 private:
  /// \brief Represents a block of memory managed by the caching allocator.
  struct Block {
    /// The offset of the block within the allocator's memory region.
    size_t offset = 0;

    /// The size of the block in bytes.
    size_t size = 0;

    /// Indicates whether the block is free or allocated.
    bool free = true;

    /// Pointer to the previous block in address order, for coalescing.
    Block* prev = nullptr;

    /// Pointer to the next block in address order, for coalescing.
    Block* next = nullptr;
  };

  /// Keyed by (size, offset): lower_bound({n, 0}) gives the smallest block that
  /// fits, and the offset tiebreak makes allocation order deterministic, which
  /// matters for reproducing bugs.
  using FreeKey = std::pair<size_t, size_t>;

  /// \brief Allocates a new block from the pool of pre-allocated memory.
  ///
  /// \return A pointer to the newly allocated block, or nullptr if allocation
  ///         fails.
  Block* NewBlock();

  /// \brief Recycles a block back into the pool of available blocks.
  ///
  /// \param b The block to recycle.
  void RecycleBlock(Block* b);

  /// \brief Inserts a block into the set of free blocks.
  ///
  /// \param b The block to insert.
  void InsertFree(Block* b);

  /// \brief Removes a block from the set of free blocks.
  ///
  /// \param b The block to remove.
  void RemoveFree(Block* b);

  /// \brief Validates the arguments for an allocation request.
  ///
  /// \param bytes     The number of bytes to allocate.
  /// \param alignment The alignment requirement for the allocation.
  /// \return          A status indicating whether the arguments are valid.
  Status ValidateAllocateArgs(size_t bytes, size_t alignment) const;

  /// \brief A free block selected for an allocation, plus the leading bytes
  ///        skipped to reach the requested alignment (zero on the common,
  ///        already-aligned path).
  struct FoundBlock {
    Block* block = nullptr;
    size_t pad = 0;
  };

  /// \brief Selects a free block for allocation, considering alignment
  ///        requirements.
  ///
  /// \param need      The number of bytes needed for the allocation.
  /// \param alignment The alignment requirement for the allocation.
  /// \return          A FoundBlock structure containing the selected block and
  ///                  any leading padding.
  FoundBlock TakeFreeBlock(size_t need, size_t alignment);

  /// \brief Selects the best-fit free block for allocation.
  ///
  /// \param need The number of bytes needed for the allocation.
  /// \return     A FoundBlock structure containing the selected block and any
  ///             leading padding.
  FoundBlock TakeBestFit(size_t need);

  /// \brief Selects a free block for allocation that satisfies an alignment
  ///        requirement stronger than the allocator's default alignment.
  ///
  /// \param need      The number of bytes needed for the allocation.
  /// \param alignment The alignment requirement for the allocation.
  /// \return          A FoundBlock structure containing the selected block and
  ///                  any leading padding.
  FoundBlock TakeOverAligned(size_t need, size_t alignment);

  /// \brief Splits a block into two parts, with the first part satisfying an
  ///        allocation request and the second part remaining free.
  ///
  /// \param b   The block to split.
  /// \param pad The size of the first part of the block.
  void SplitFront(Block* b, size_t pad);

  /// \brief Splits a block into two parts, with the first part satisfying an
  ///        allocation request and the second part remaining free.
  ///
  /// \param b    The block to split.
  /// \param need The size of the first part of the block.
  void SplitTail(Block* b, size_t need);

  /// \brief Commits a block to an allocation, marking it as in use and updating
  ///        the allocator's internal state.
  ///
  /// \param b The block to commit.
  /// \return  A pointer to the allocated memory region corresponding to the
  ///          committed block.
  void* Commit(Block* b);

  std::byte* base_ = nullptr;
  size_t capacity_ = 0;
  size_t alignment_ = kTensorAlignment;
  DeviceId device_;

  std::vector<std::unique_ptr<Block>> pool_;
  std::vector<Block*> recycled_;
  Block* head_ = nullptr;

  std::map<FreeKey, Block*> free_blocks_;
  absl::flat_hash_map<void*, Block*> allocated_;

  size_t in_use_ = 0;
  size_t peak_in_use_ = 0;
  uint64_t num_allocations_ = 0;
  uint64_t num_deallocations_ = 0;
  uint64_t num_splits_ = 0;
  uint64_t num_coalesces_ = 0;
  uint64_t num_failures_ = 0;
};

}  // namespace inferx
