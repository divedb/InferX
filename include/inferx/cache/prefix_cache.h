#ifndef INFERX_CACHE_PREFIX_CACHE_H_
#define INFERX_CACHE_PREFIX_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/sha256.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

/// \brief Content digest identifying one block's KV state.
using BlockHash = std::array<uint8_t, 32>;

/// \brief Hash chain over block-aligned token runs (vLLM V1 `BlockHash`
///        analog).
///
/// `digest_i = SHA256("inferx.kv.blockhash.v1" || digest_{i-1} ||
/// u64le(token_count) || tokens as u64le)`. The chain seed folds in the
/// request's cache salt, so a salt invalidates the whole chain from block 0 —
/// the same isolation vLLM gets by hashing `("cache_salt", salt)` into the
/// first block. Only full blocks are hashed: a partial tail block is private
/// to the request computing it and never becomes cacheable.
class BlockHashChain {
 public:
  /// \brief Starts a chain. `cache_salt` namespaces every digest that follows.
  explicit BlockHashChain(std::string_view cache_salt = {});

  /// \brief Extends the chain over one full block of tokens.
  ///
  /// \param tokens Exactly one block of token ids, in logical order.
  /// \return       The block's digest.
  BlockHash Extend(absl::Span<const TokenId> tokens);

  /// \brief The chain tip; the digest of the last hashed block.
  const BlockHash& tip() const { return tip_; }

 private:
  BlockHash tip_{};
};

/// \brief Hash-keyed index over computed KV blocks, with reference-counted
///        sharing and LRU eviction (vLLM V1 `BlockHashToBlockMap` +
/// `FreeKVCacheBlockQueue` analog).
///
/// The index sits between the scheduler and the block pool: adopted blocks
/// are pinned (never eviction victims) while any request references them;
/// released blocks with live hash entries stay cached and evictable rather
/// than returning to the pool's free list; eviction reclaims the
/// least-recently-used unpinned entries first and happens lazily, on
/// allocation pressure. Duplicate registrations of one hash coexist, as in
/// vLLM — concurrent writers of an identical prefix each keep their block
/// and adoption picks the first entry.
///
/// Not thread-safe: single-threaded with the scheduler, like the pool.
class PrefixCache {
 public:
  explicit PrefixCache(KvBlockPool* pool) : pool_(pool) {}

  /// \brief Adopts the longest cached prefix of `hashes`, pinning its blocks.
  ///
  /// The walk stops at the first missing hash. `max_blocks` caps the
  /// adoption (the n-1 logits rule truncates to whole blocks here).
  ///
  /// \param hashes    Block digests in logical order, from the chain seed.
  /// \param max_blocks Stop after this many blocks; negative means unbounded.
  /// \return          The pinned block ids, in logical order.
  std::vector<int32_t> Adopt(absl::Span<const BlockHash> hashes, int64_t max_blocks = -1);

  /// \brief Registers one fully computed block under `hash`.
  ///
  /// The caller holds the resulting reference; Release() drops it. A block
  /// is registered at most once; re-registering the same block is a no-op.
  void Publish(const BlockHash& hash, int32_t block);

  /// \brief Drops one reference per block previously adopted or published.
  ///
  /// Entries whose reference count reaches zero stay discoverable and become
  /// LRU eviction candidates; their bytes are not freed.
  void Release(absl::Span<const int32_t> blocks);

  /// \brief Evicts up to `count` cached blocks, least-recently-used first.
  ///
  /// Pinned entries are never victims. Evicted entries leave the index and
  /// their blocks return to the pool's free list.
  ///
  /// \param count Number of blocks sought.
  /// \return      The number actually evicted.
  int64_t Evict(int64_t count);

  /// \brief Forgets every entry, freeing evictable blocks to the pool.
  void Clear();

  /// \brief True while `block` has a live cache entry (pinned or evictable).
  bool Contains(int32_t block) const { return by_block_.contains(block); }

  /// \brief Entries currently discoverable by hash.
  int64_t cached_blocks() const { return static_cast<int64_t>(by_block_.size()); }
  /// \brief Cached entries with no references; the eviction pool's depth.
  int64_t evictable_blocks() const { return static_cast<int64_t>(lru_.size()); }

 private:
  struct Entry {
    BlockHash hash;
    int32_t block = -1;
    uint64_t refs = 0;
    /// Position in the LRU list; meaningful only while refs == 0.
    std::list<int32_t>::iterator lru;
  };

  /// Forgets `entry`'s mappings without touching the pool. The caller
  /// decides what happens to the block.
  void Unlink(Entry* entry);

  KvBlockPool* pool_;
  /// Hash → live entries for that hash (duplicates coexist).
  absl::flat_hash_map<BlockHash, std::vector<Entry*>> by_hash_;
  /// Block id → its entry; owns all entries.
  absl::flat_hash_map<int32_t, std::unique_ptr<Entry>> by_block_;
  /// Unpinned entries, least-recently-released first.
  std::list<int32_t> lru_;
};

}  // namespace inferx

#endif  // INFERX_CACHE_PREFIX_CACHE_H_
