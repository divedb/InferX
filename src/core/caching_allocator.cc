#include "inferx/core/caching_allocator.h"

#include <bit>

#include "inferx/common/align.h"
#include "inferx/core/tensor_spec.h"

namespace inferx {

double CachingAllocator::Stats::Fragmentation() const {
  const size_t total_free = capacity - in_use;
  if (total_free == 0) return 0.0;
  return 1.0 - static_cast<double>(largest_free_block) /
                   static_cast<double>(total_free);
}

CachingAllocator::CachingAllocator(std::byte* base, size_t capacity,
                                   DeviceId device, size_t alignment)
    : device_(device) {
  if (alignment == 0 || !std::has_single_bit(alignment)) {
    alignment = kTensorAlignment;
  }
  alignment_ = alignment;

  // Align the region start, then round its length down, so that every block
  // offset is a multiple of alignment_ *and* every base_+offset address is too.
  // Establishing both invariants once here is what lets Allocate() stay free of
  // per-allocation alignment arithmetic.
  size_t skew = 0;
  if (base != nullptr) {
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base);
    const uintptr_t aligned_addr = AlignUp(addr, alignment_);
    skew = static_cast<size_t>(aligned_addr - addr);
  }
  base_ = (base == nullptr) ? nullptr : base + skew;
  capacity_ = (capacity > skew ? capacity - skew : 0) & ~(alignment_ - 1);

  Reset();
}

CachingAllocator::Block* CachingAllocator::NewBlock() {
  if (!recycled_.empty()) {
    Block* b = recycled_.back();
    recycled_.pop_back();
    *b = Block{};

    return b;
  }

  pool_.push_back(std::make_unique<Block>());

  return pool_.back().get();
}

void CachingAllocator::RecycleBlock(Block* b) {
  *b = Block{};
  recycled_.push_back(b);
}

void CachingAllocator::InsertFree(Block* b) {
  b->free = true;
  free_blocks_.emplace(FreeKey{b->size, b->offset}, b);
}

void CachingAllocator::RemoveFree(Block* b) {
  free_blocks_.erase(FreeKey{b->size, b->offset});
}

void CachingAllocator::Reset() {
  free_blocks_.clear();
  allocated_.clear();
  recycled_.clear();
  pool_.clear();
  head_ = nullptr;
  in_use_ = 0;

  if (base_ != nullptr && capacity_ > 0) {
    Block* b = NewBlock();
    b->offset = 0;
    b->size = capacity_;
    head_ = b;
    InsertFree(b);
  }
}

StatusOr<void*> CachingAllocator::Allocate(size_t bytes, size_t alignment) {
  INFERX_RETURN_IF_ERROR(ValidateAllocateArgs(bytes, alignment));

  const size_t need = AlignUp(bytes, alignment_);

  if (need < bytes) {
    return InvalidArgumentError("allocation size ", bytes, " overflows");
  }

  const FoundBlock found = TakeFreeBlock(need, alignment);

  if (found.block == nullptr) {
    ++num_failures_;
    const Stats s = GetStats();

    return ResourceExhaustedError(
        "workspace allocator on ", device_.ToString(), " cannot satisfy ", need,
        " bytes aligned to ", alignment, ": ", s.capacity - s.in_use,
        " free across ", s.free_block_count, " blocks, largest ",
        s.largest_free_block, " bytes (fragmentation ", s.Fragmentation(), ")");
  }

  if (found.pad > 0) SplitFront(found.block, found.pad);

  SplitTail(found.block, need);

  return Commit(found.block);
}

Status CachingAllocator::ValidateAllocateArgs(size_t bytes,
                                              size_t alignment) const {
  if (base_ == nullptr) {
    return FailedPreconditionError("CachingAllocator is not initialized");
  }

  if (bytes == 0) {
    return InvalidArgumentError("zero-byte allocation requested");
  }

  if (alignment == 0 || !std::has_single_bit(alignment)) {
    return InvalidArgumentError("alignment ", alignment,
                                " is not a non-zero power of two");
  }

  return OkStatus();
}

CachingAllocator::FoundBlock CachingAllocator::TakeFreeBlock(size_t need,
                                                             size_t alignment) {
  // Every block offset is a multiple of alignment_ and base_ is aligned to it,
  // so any request no stronger than alignment_ is satisfied by best-fit alone;
  // a stronger alignment must scan for a block whose aligned start still fits.
  return alignment <= alignment_ ? TakeBestFit(need)
                                 : TakeOverAligned(need, alignment);
}

CachingAllocator::FoundBlock CachingAllocator::TakeBestFit(size_t need) {
  auto it = free_blocks_.lower_bound(FreeKey{need, 0});

  if (it == free_blocks_.end()) return {nullptr, 0};

  Block* b = it->second;
  free_blocks_.erase(it);

  return {b, 0};
}

CachingAllocator::FoundBlock CachingAllocator::TakeOverAligned(
    size_t need, size_t alignment) {
  // Rare path -- the common workspace request uses the default alignment.
  // Linear in the number of candidates large enough to hold `need`.
  const uintptr_t base_addr = reinterpret_cast<uintptr_t>(base_);

  for (auto it = free_blocks_.lower_bound(FreeKey{need, 0});
       it != free_blocks_.end(); ++it) {
    Block* cand = it->second;
    const uintptr_t addr = base_addr + cand->offset;
    const size_t skip = static_cast<size_t>(AlignUp(addr, alignment) - addr);

    if (skip <= cand->size && need <= cand->size - skip) {
      free_blocks_.erase(it);

      return {cand, skip};
    }
  }

  return {nullptr, 0};
}

void CachingAllocator::SplitFront(Block* b, size_t pad) {
  // Split the alignment padding off the front so it stays allocatable. `pad` is
  // a multiple of alignment_ (both the block start and the aligned target are),
  // so the offset invariant survives. The new free block cannot merge
  // backwards: adjacent free blocks are always coalesced on release, so a free
  // block's predecessor is never free.
  Block* head_pad = NewBlock();
  head_pad->offset = b->offset;
  head_pad->size = pad;
  head_pad->prev = b->prev;
  head_pad->next = b;

  if (b->prev != nullptr) {
    b->prev->next = head_pad;
  } else {
    head_ = head_pad;
  }

  b->prev = head_pad;
  b->offset += pad;
  b->size -= pad;
  InsertFree(head_pad);
  ++num_splits_;
}

void CachingAllocator::SplitTail(Block* b, size_t need) {
  // Split the tail only when the remainder is itself usable. A remainder
  // smaller than alignment_ can never be handed out, so leaving it attached
  // costs nothing and saves a block object plus a later coalesce.
  if (b->size - need < alignment_) return;

  Block* rest = NewBlock();
  rest->offset = b->offset + need;
  rest->size = b->size - need;
  rest->prev = b;
  rest->next = b->next;
  if (b->next != nullptr) b->next->prev = rest;
  b->next = rest;
  b->size = need;
  InsertFree(rest);
  ++num_splits_;
}

void* CachingAllocator::Commit(Block* b) {
  b->free = false;
  void* ptr = base_ + b->offset;
  allocated_.emplace(ptr, b);
  in_use_ += b->size;

  if (in_use_ > peak_in_use_) peak_in_use_ = in_use_;

  ++num_allocations_;

  return ptr;
}

StatusOr<TensorView> CachingAllocator::AllocateView(const TensorSpec& spec) {
  INFERX_RETURN_IF_ERROR(CheckRankFitsView(spec.GetShape()));
  INFERX_RETURN_IF_ERROR(spec.Verify());

  const int64_t bytes = spec.NBytes();
  INFERX_ASSIGN_OR_RETURN(void* p, Allocate(static_cast<size_t>(bytes)));

  return TensorView::Create(p, spec.GetDataType(), spec.GetShape(), device_);
}

Status CachingAllocator::Deallocate(void* ptr) {
  if (ptr == nullptr) return OkStatus();

  auto it = allocated_.find(ptr);
  if (it == allocated_.end()) {
    return InvalidArgumentError(
        "Deallocate called on a pointer this allocator did not hand out, or "
        "which was already freed");
  }
  Block* b = it->second;
  allocated_.erase(it);

  in_use_ -= b->size;
  b->free = true;
  ++num_deallocations_;

  // Coalesce backwards, then forwards. Both neighbours must leave the free
  // index before their size or offset changes, since those form the key.
  if (b->prev != nullptr && b->prev->free) {
    Block* prev = b->prev;
    RemoveFree(prev);
    prev->size += b->size;
    prev->next = b->next;
    if (b->next != nullptr) b->next->prev = prev;
    RecycleBlock(b);
    b = prev;
    ++num_coalesces_;
  }

  if (b->next != nullptr && b->next->free) {
    Block* next = b->next;
    RemoveFree(next);
    b->size += next->size;
    b->next = next->next;
    if (next->next != nullptr) next->next->prev = b;
    RecycleBlock(next);
    ++num_coalesces_;
  }

  InsertFree(b);
  return OkStatus();
}

CachingAllocator::Stats CachingAllocator::GetStats() const {
  Stats s;
  s.capacity = capacity_;
  s.in_use = in_use_;
  s.peak_in_use = peak_in_use_;
  s.free_block_count = free_blocks_.size();
  s.largest_free_block =
      free_blocks_.empty() ? 0 : std::prev(free_blocks_.end())->first.first;
  s.num_allocations = num_allocations_;
  s.num_deallocations = num_deallocations_;
  s.num_splits = num_splits_;
  s.num_coalesces = num_coalesces_;
  s.num_failures = num_failures_;

  return s;
}

}  // namespace inferx
