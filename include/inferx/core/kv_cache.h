#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/device_buffer.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx {

/// \brief How many bytes one token occupies in one layer's cache.
///
/// Parameterized rather than computed from `2 · kv_heads · head_dim`, because
/// that formula is a property of grouped-query attention and not of caching.
/// MLA stores a single compressed latent per token per layer and does not split
/// into K and V at all (ARCHITECTURE.md §7.3, T11), so the pool has to be told
/// its geometry rather than deriving it. Getting this wrong means rewriting the
/// block allocator later, which is exactly what T11 says to avoid.
struct KvLayout {
  /// Entries per token per layer. 2 for K and V; 1 for an MLA latent.
  int64_t entries_per_token = 2;
  /// KV heads *on this rank*. For GQA this is `num_key_value_heads / tp_size`;
  /// for MLA it does not shard at all, which is why the pool never divides it.
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  DataType dtype = DataType::kBFloat16;

  int64_t ElementsPerToken() const {
    return entries_per_token * kv_heads * head_dim;
  }

  int64_t BytesPerToken() const {
    return DataTypeByteSize(dtype, ElementsPerToken());
  }
};

/// \brief The single device allocation holding all KV blocks, plus its free
/// list.
///
/// One `cudaMalloc` at startup, never freed and never grown: §6.1's rule is
/// that a `cudaFree` in the steady state implicitly synchronizes the device and
/// destroys the overlap pipeline, so the pool is sized once and hands out
/// indices thereafter.
///
/// Layout:
///
///     kv[layer][entry][block][block_size][kv_heads][head_dim]
///
/// with `entry` being K or V for GQA. §6.2 sketches this with `entry` inside
/// `block`; it is outside here for two reasons. Attention streams every K of a
/// sequence and then every V, so keeping each entry contiguous across blocks is
/// the better read pattern. And it makes `KeyCache`/`ValueCache` plain
/// contiguous views, which matters because they are handed to kernels as
/// ordinary `TensorView`s rather than as a strided five-dimensional thing every
/// kernel would have to re-index.
///
/// Blocks are identified by an `int32` index rather than a pointer, because
/// that is what a block table has to contain for a kernel to walk it, and
/// because indices survive being copied to the device.
///
/// Not thread-safe: the scheduler is the only thing that allocates or frees
/// blocks (§5.1), and making it lock-free by topology is the whole point.
class KvBlockPool {
 public:
  /// \brief Carves `num_blocks` blocks out of one allocation.
  ///
  /// \param num_layers  Layers to cache. Each gets its own region.
  /// \param num_blocks  Total blocks, shared across all sequences.
  /// \param block_size  Tokens per block. 16 by default (T10).
  /// \param layout      Per-token geometry. \see KvLayout.
  /// \return            The pool, or ResourceExhausted if it will not fit.
  /// \param device      Where the pool lives. `Cpu()` exists so the scheduler's
  /// block bookkeeping can be
  ///                    unit-tested on a machine with no device at all, which
  ///                    §3.1 calls the highest-leverage testability decision in
  ///                    the design. The free list is the same code either way.
  static StatusOr<KvBlockPool> Create(int64_t num_layers, int64_t num_blocks,
                                      int64_t block_size,
                                      const KvLayout& layout, DeviceId device);

  /// \brief Takes a free block. The contents are whatever was there before.
  ///
  /// \return A block index, or ResourceExhausted when the pool is full. Callers
  ///         are expected to act on exhaustion by preempting (§8.2), which is
  ///         why it is a status rather than a fatal error.
  StatusOr<int32_t> AllocateBlock();

  /// \brief Returns a block to the free list.
  ///
  /// \param block A block previously returned by `AllocateBlock`. Returning one
  ///              twice is an error rather than a silent corruption of the free
  ///              list, since that would hand the same block to two sequences.
  Status FreeBlock(int32_t block);

  /// \brief Frees several blocks, e.g. a whole finished sequence.
  Status FreeBlocks(const std::vector<int32_t>& blocks);

  /// \brief The K and V regions for one layer, shaped for the kernels.
  ///
  /// Both are `[num_blocks, block_size, kv_heads, head_dim]`. They are separate
  /// views into one allocation rather than one view with an `entry` dimension,
  /// because every kernel indexes K and V independently and a five-dimensional
  /// view would just be indexed apart again at every use.
  StatusOr<TensorView> KeyCache(int64_t layer) const;
  StatusOr<TensorView> ValueCache(int64_t layer) const;

  int64_t num_layers() const { return num_layers_; }
  int64_t num_blocks() const { return num_blocks_; }
  int64_t block_size() const { return block_size_; }
  const KvLayout& layout() const { return layout_; }

  int64_t free_blocks() const {
    return static_cast<int64_t>(free_list_.size());
  }
  int64_t used_blocks() const { return num_blocks_ - free_blocks(); }

  size_t bytes() const { return storage_.size(); }

  /// \brief Blocks needed to hold `tokens` tokens.
  int64_t BlocksForTokens(int64_t tokens) const {
    return (tokens + block_size_ - 1) / block_size_;
  }

  /// \brief Bytes one block occupies across all layers. Useful for sizing.
  static int64_t BlockBytes(int64_t num_layers, int64_t block_size,
                            const KvLayout& layout) {
    return num_layers * block_size * layout.BytesPerToken();
  }

 private:
  KvBlockPool() = default;

  DeviceBuffer storage_;
  KvLayout layout_;
  int64_t num_layers_ = 0;
  int64_t num_blocks_ = 0;
  int64_t block_size_ = 0;
  /// Bytes from the start of one layer's region to the next.
  int64_t layer_stride_ = 0;
  /// Bytes from a layer's K region to its V region.
  int64_t entry_stride_ = 0;
  DeviceId device_;

  // Free blocks, taken from the back. LIFO rather than FIFO on purpose: the
  // most recently freed block is the most likely to still be in L2.
  std::vector<int32_t> free_list_;
};

/// \brief Per-sequence block table: which blocks hold which part of a sequence.
///
/// Host-side bookkeeping. The device-side copy is a flat `[max_seqs,
/// max_blocks_per_seq]` int32 tensor living in a fixed buffer so that
/// CUDA-graph-captured decode steps can read it without re-capture (§6.2); this
/// class is what fills that buffer.
class BlockTable {
 public:
  explicit BlockTable(int64_t block_size) : block_size_(block_size) {}

  /// \brief Appends a block to the sequence.
  void Append(int32_t block) { blocks_.push_back(block); }

  /// \brief The block holding logical token `pos`, and the slot within it.
  ///
  /// \return false when `pos` is past what has been allocated.
  bool Locate(int64_t pos, int32_t* block, int64_t* slot) const {
    const int64_t index = pos / block_size_;
    if (index < 0 || index >= static_cast<int64_t>(blocks_.size()))
      return false;

    *block = blocks_[static_cast<size_t>(index)];
    *slot = pos % block_size_;
    return true;
  }

  const std::vector<int32_t>& blocks() const { return blocks_; }
  int64_t size() const { return static_cast<int64_t>(blocks_.size()); }
  int64_t capacity_tokens() const { return size() * block_size_; }
  void Clear() { blocks_.clear(); }

 private:
  int64_t block_size_;
  std::vector<int32_t> blocks_;
};

}  // namespace inferx
