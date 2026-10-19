#ifndef INFERX_CACHE_KV_BLOCK_POOL_H_
#define INFERX_CACHE_KV_BLOCK_POOL_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/cache/block_hash.h"
#include "inferx/cache/cache_spec.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx {

/// Block identifier as it appears in block tables and kernel indices.
using BlockId = int32_t;

/// \brief A prefix-index key: a block digest namespaced by its group.
///
/// Two groups may compute the same digest for different content (different
/// geometry, different layers), so the index key carries the group.
struct BlockKey {
  BlockHash hash;
  int32_t group_index = 0;

  bool operator==(const BlockKey&) const = default;

  template <typename H>
  friend H AbslHashValue(H h, const BlockKey& k) {
    return H::combine(std::move(h), k.hash, k.group_index);
  }
};

/// \brief The single device allocation holding all KV blocks, plus the
/// ledger that hands them out.
///
/// One `cudaMalloc` at startup, never freed and never grown: §6.1's rule is
/// that a `cudaFree` in the steady state implicitly synchronizes the device and
/// destroys the overlap pipeline, so the pool is sized once and hands out
/// indices thereafter.
///
/// Storage is byte-denominated (kvcache.md §6.1): each layer's region is
/// `block_size * bytes_per_token` bytes per block, sized from the layer's
/// `CacheComponent`. The pool stops understanding geometry — GQA's K and V,
/// MLA latents, and quantized payloads are all just bytes; the owning model
/// layer reshapes `LayerCache` views into kernel geometry.
///
/// Blocks are identified by an `int32` index rather than a pointer, because
/// that is what a block table has to contain for a kernel to walk it, and
/// because indices survive being copied to the device. Id 0 is
/// `kNullBlockId`: reserved, never allocated, never written. Window-rolled
/// table entries point there and kernels skip block-0 rows; usable ids run
/// 1..num_blocks.
///
/// The ledger (kvcache.md §4.9) tracks, per block, a reference count, a
/// prefix-index entry, and queue membership. Membership is exclusive: the
/// free queue (never cached, ref 0), the LRU queue (cached and discoverable,
/// ref 0), or neither (ref > 0). `AllocateBlock` evicts from the LRU when
/// the free queue is empty, so cached-but-unreferenced blocks are capacity.
///
/// Not thread-safe: the scheduler is the only thing that allocates or frees
/// blocks (§5.1), and making it lock-free by topology is the whole point.
class KvBlockPool {
 public:
  /// Reserved, never allocated, never written. Window-rolled table entries
  /// point here; kernels skip block 0 rows.
  static constexpr BlockId kNullBlockId = 0;

  /// \brief Carves `num_blocks` usable blocks (ids 1..num_blocks) out of one
  /// allocation.
  ///
  /// \param num_layers  Layers to cache. Each gets its own region.
  /// \param num_blocks  Usable blocks, shared across all sequences. Block 0
  ///                    is reserved as the null block and consumes storage.
  /// \param block_size  Tokens per block. 16 by default (T10).
  /// \param component   Per-token storage of every layer, in bytes.
  ///                    \see CacheComponent.
  /// \param device      Where the pool lives. `Cpu()` exists so the scheduler's
  ///                    block bookkeeping can be unit-tested on a machine with
  ///                    no device at all, which §3.1 calls the
  ///                    highest-leverage testability decision in the design.
  ///                    The ledger is the same code either way.
  /// \return            The pool, or ResourceExhausted if it will not fit.
  static StatusOr<KvBlockPool> Create(int64_t num_layers, int64_t num_blocks,
                                      int64_t block_size, const CacheComponent& component,
                                      Device device);

  /// \brief Same, with one component per layer.
  ///
  /// Hybrid models (Qwen3-Next) and decompressed MLA layers vary per-layer
  /// storage; the pool keeps one free queue across layers and sizes each
  /// layer's region from its own component.
  static StatusOr<KvBlockPool> Create(int64_t num_blocks, int64_t block_size,
                                      const std::vector<CacheComponent>& components,
                                      Device device);

  // -- allocation and ledger ------------------------------------------------

  /// \brief Takes a free block, evicting LRU cached entries first when the
  /// free queue is empty. Never returns kNullBlockId.
  ///
  /// \return A block id, or ResourceExhausted when nothing is free or
  ///         evictable. Callers are expected to act on exhaustion by
  ///         preempting (§8.2), which is why it is a status rather than a
  ///         fatal error.
  StatusOr<BlockId> AllocateBlock();

  /// \brief Takes `n` free blocks, or fails without handing out anything.
  StatusOr<std::vector<BlockId>> AllocateBlocks(int64_t n);

  /// \brief +1 reference (adopt/fork/sharing). Errors on the null block, an
  /// unknown id, or a block on the free queue; a cached-but-idle block is
  /// acquirable and leaves the LRU.
  Status AcquireBlock(BlockId block);

  /// \brief -1 reference. At zero: discoverable cached blocks stay in the
  /// index and join the LRU; everything else returns to the free queue (LIFO
  /// position).
  Status ReleaseBlock(BlockId block);

  /// \brief Releases a batch (finished sequence, fork undo).
  Status ReleaseBlocks(const std::vector<BlockId>& blocks);

  /// \brief Evicts up to `count` cached blocks, least-recently-used first.
  /// Pinned entries are never victims. Returns the number actually evicted.
  int64_t EvictCached(int64_t count);

  /// \brief Reference count of `block`; 0 for the null block, unknown ids,
  ///         and free blocks.
  uint32_t RefCount(BlockId block) const;

  // -- prefix index (absorbed PrefixCache; one binding per key) ----------------

  /// \brief Registers one fully computed block under `key`. The caller
  /// holds the resulting reference. Re-registering the same block is a
  /// no-op; a key already bound to another block moves to this one.
  void PublishBlock(const BlockKey& key, BlockId block);

  /// \brief Cached and discoverable block for `key`, if any.
  std::optional<BlockId> LookupCached(const BlockKey& key) const;

  /// \brief Drops a block's index entry without freeing the block (used when
  /// a window group rolls a published block off and its last reference
  /// drops). No-op on uncached blocks.
  void ForgetBlock(BlockId block);

  /// \brief Makes every cached entry undiscoverable. Evictable entries are
  /// reclaimed now; pinned ones become ownership-only tombstones that free
  /// when their last holder releases.
  void ClearCachedEntries();

  /// \brief Drops every prefix-index entry. Unlike ClearCachedEntries there
  /// are no tombstones: fails with FailedPrecondition while any block is
  /// still held, so callers drain their requests first.
  Status ResetPrefixCache();

  /// \brief True while the ledger tracks `block` (entry or tombstone).
  bool Contains(BlockId block) const;
  /// \brief Discoverable entries; tombstones do not count.
  int64_t NumCachedBlocks() const;
  /// \brief Cached entries with zero references; the eviction pool's depth.
  int64_t EvictableBlocks() const;

  // -- geometry and kernel views ---------------------------------------------

  /// \brief One layer's byte region, shaped for reshaping by its owner.
  ///
  /// A `[num_blocks + 1, block_size, bytes_per_token]` uint8 view including
  /// the null block's reserved row. The owning model layer reshapes it into
  /// kernel geometry (`Tensor::Reshape`/`Bitcast`); the pool never interprets
  /// component bytes. The returned Tensor borrows the pool's allocation: the
  /// pool must outlive it.
  ///
  /// \param layer Layer index.
  /// \return      The cache tensor, or an error status for an invalid layer.
  StatusOr<Tensor> LayerCache(int64_t layer) const;

  /// \brief Returns the number of layers the pool caches.
  int64_t NumLayers() const { return num_layers_; }
  /// \brief Returns the number of usable blocks (null block excluded),
  ///         shared across all sequences.
  int64_t NumBlocks() const { return num_blocks_; }
  /// \brief Returns the number of tokens per block.
  int64_t BlockSize() const { return block_size_; }
  /// \brief Returns the per-token storage of the cache.
  const CacheComponent& Component() const { return component_; }

  /// \brief Returns layer `layer`'s storage (heterogeneous pools vary).
  const CacheComponent& ComponentFor(int64_t layer) const {
    return components_.empty() ? component_ : components_[static_cast<size_t>(layer)];
  }

  /// \brief Returns the free-queue depth (excludes evictable cached blocks).
  int64_t FreeBlocks() const { return free_depth_; }
  /// \brief Returns the number of blocks not on the free queue.
  int64_t UsedBlocks() const { return num_blocks_ - free_depth_; }
  /// \brief Blocks available to Reserve: free plus evictable, since
  ///        allocation evicts from the LRU under pressure.
  int64_t NumFreeBlocks() const { return FreeBlocks() + EvictableBlocks(); }

  /// \brief Returns the size of the backing allocation in bytes.
  size_t Bytes() const { return storage_.size(); }

  /// \brief Blocks needed to hold `tokens` tokens.
  int64_t BlocksForTokens(int64_t tokens) const {
    return (tokens + block_size_ - 1) / block_size_;
  }

  /// \brief Fraction of usable blocks not on the free queue, in [0, 1].
  double Usage() const {
    return num_blocks_ == 0 ? 0.0 : static_cast<double>(num_blocks_ - free_depth_) /
                                        static_cast<double>(num_blocks_);
  }

  /// \brief Bytes one usable block occupies across all layers.
  static int64_t BlockBytes(int64_t num_layers, int64_t block_size,
                            const CacheComponent& component) {
    return num_layers * block_size * component.bytes_per_token;
  }

 private:
  KvBlockPool() = default;

  /// \brief Per-block ledger entry (kvcache.md §4.9). Index 0 is the null
  /// block's reserved record; one extra sentinel slots in as the LRU list's
  /// anchor.
  struct Record {
    /// Holders: request ownership, prefix pins, fork sharers. 0 exactly while
    /// the block is available (free queue or LRU).
    uint32_t ref_cnt = 0;

    /// Index key while cached; zeroed on eviction/forget.
    BlockKey key{};

    /// Has a ledger entry (true even for tombstones until reclaimed).
    bool cached = false;
    /// False once ClearCachedEntries() made the entry undiscoverable.
    bool discoverable = false;

    /// Intrusive free-queue links (LIFO: push/pop at the back).
    int32_t prev_free = -1;
    int32_t next_free = -1;
    /// Intrusive LRU links; valid only while on the LRU.
    int32_t lru_prev = -1;
    int32_t lru_next = -1;
  };

  static constexpr int32_t kFree = -1;

  bool Usable(BlockId block) const { return block >= 1 && block <= num_blocks_; }
  int32_t Sentinel() const { return num_blocks_ + 1; }

  void FreePushBack(BlockId block);
  BlockId FreePopBack();
  void LruPushBack(BlockId block);
  void LruErase(BlockId block);
  BlockId LruPopFront();

  /// \brief Evicts `block`'s index entry and returns the block to the free
  /// queue. Only valid for ref-0 discoverable entries.
  void Reclaim(BlockId block);

  DeviceBuffer storage_;
  CacheComponent component_;
  /// Per-layer components for heterogeneous pools; empty when uniform.
  std::vector<CacheComponent> components_;
  /// Per-layer byte offsets (heterogeneous pools).
  std::vector<int64_t> layer_offsets_;
  int64_t num_layers_ = 0;
  int64_t num_blocks_ = 0;
  int64_t block_size_ = 0;
  /// Bytes from the start of one layer's region to the next.
  int64_t layer_stride_ = 0;
  Device device_;

  /// Null block + one record per usable block + the LRU sentinel.
  std::vector<Record> records_;
  /// Discoverable index entries: key -> block.
  absl::flat_hash_map<BlockKey, BlockId> index_;
  int32_t free_head_ = kFree;
  int32_t free_tail_ = kFree;
  int64_t free_depth_ = 0;
};

/// \brief Per-sequence block table: which blocks hold which part of a sequence.
///
/// Host-side bookkeeping. The device-side copy is a flat `[max_seqs,
/// max_blocks_per_seq]` int32 tensor living in a fixed buffer so that
/// CUDA-graph-captured decode steps can read it without re-capture (§6.2); this
/// class is what fills that buffer.
class BlockTable {
 public:
  /// \brief Constructs a table whose blocks each hold `block_size` tokens.
  ///
  /// \param block_size Tokens per block; must match the pool's.
  explicit BlockTable(int64_t block_size) : block_size_(block_size) {}

  /// \brief Appends a block to the sequence.
  void Append(BlockId block) { blocks_.push_back(block); }

  /// \brief The block holding logical token `pos`, and the slot within it.
  ///
  /// \param pos   Logical token position.
  /// \param block Output: the block index holding the token.
  /// \param slot  Output: the slot of the token within the block.
  /// \return      False when `pos` is past what has been allocated.
  bool Locate(int64_t pos, BlockId* block, int64_t* slot) const {
    // Reject on `pos`, not on the divided index: truncating division maps
    // (-block_size, 0) to index 0 and would hand back a negative slot.
    if (pos < 0) return false;
    const int64_t index = pos / block_size_;
    if (index >= static_cast<int64_t>(blocks_.size())) return false;

    *block = blocks_[static_cast<size_t>(index)];
    *slot = pos % block_size_;
    return true;
  }

  /// \brief Returns the block indices of the sequence, in logical order.
  const std::vector<BlockId>& Blocks() const { return blocks_; }
  /// \brief Returns the number of blocks held by the sequence.
  int64_t Size() const { return static_cast<int64_t>(blocks_.size()); }
  /// \brief Returns the number of tokens the held blocks can store.
  int64_t CapacityTokens() const { return Size() * block_size_; }
  /// \brief Drops every block from the table.
  void Clear() { blocks_.clear(); }

 private:
  int64_t block_size_;
  std::vector<BlockId> blocks_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_KV_BLOCK_POOL_H_
