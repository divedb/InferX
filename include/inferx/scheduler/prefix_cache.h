#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/kv_cache.h"

namespace inferx::scheduler {

/// \brief A radix tree over token sequences, holding the KV of prefixes that
///        more than one request is likely to want (§6.3).
///
/// The workloads this exists for -- a system prompt in front of every request,
/// few-shot examples, multi-turn chat replaying the conversation, an agent
/// rebuilding its scratchpad -- all share a *token* prefix, and all of them pay
/// for it twice without this: once when the first request computes it and again
/// on every request after. Prefill is the expensive half of serving (a 2k prompt
/// is ~16 decode steps), so a prefix that is already in the cache is the
/// difference between 176 ms of time-to-first-token and none.
///
/// **A radix tree rather than vLLM's block hashing** (T4). Hashing finds a match
/// by probing at every block boundary; a tree finds the longest one in a single
/// descent and shares the interior nodes between everyone who passes through
/// them. The cost is node splitting on a partial match, which is only tolerable
/// because §5.1 makes the scheduler the sole mutator -- there is no lock here
/// and there is not meant to be one.
///
/// ### What is shareable
///
/// **Complete blocks only.** A block still being written to cannot be shared,
/// because the sequence writing it would be scribbling on someone else's
/// history. So every run of tokens in the tree is a whole number of blocks, and
/// a sequence's partial tail block is never offered. That is also what makes
/// sharing safe without copy-on-write: a request that matches a prefix reads
/// those blocks and writes only from the match onward, which begins at a block
/// boundary by construction.
///
/// The other half of the safety argument is that a block's contents are a
/// function of the tokens before it as well as the tokens in it -- attention is
/// causal, so a key depends on the whole prefix. That is exactly what the path
/// through the tree encodes: two sequences reach the same node only by agreeing
/// on every token along the way.
///
/// ### Ownership
///
/// The cache owns every block in the tree. A sequence's block table may point at
/// tree-owned blocks -- its matched prefix -- and at blocks it allocated itself.
/// `Acquire` takes a reference so the prefix cannot be evicted while it is being
/// read; `Finish` gives the reference back and offers the sequence's own
/// complete blocks to the tree, which keeps what is new and frees the rest.
///
/// Nothing in the tree is reachable from the pool's free list, so cached blocks
/// count as *used*. They are reclaimed by `Evict`, which the allocator calls on
/// its own behalf when it runs dry: an unreferenced cached block is free memory
/// that happens to still hold something useful.
class PrefixCache {
 public:
  /// \brief Creates a cache over `pool`, which must outlive it.
  PrefixCache(KvBlockPool* pool, int64_t block_size);

  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;

  /// \brief What a lookup found.
  struct Match {
    /// Tree-owned blocks covering the first `tokens` of the query, in order.
    std::vector<int32_t> blocks;
    /// Tokens covered. Always `blocks.size() * block_size`.
    int64_t tokens = 0;
  };

  /// \brief Finds the longest cached prefix of `tokens` and references it.
  ///
  /// Every returned block is pinned against eviction until the matching
  /// `Finish`. Callers that take a match and never finish it leak the pin, and
  /// the blocks become unevictable for the life of the cache.
  ///
  /// \param tokens  The full sequence being admitted, prompt and all.
  /// \param limit   Never match beyond this many tokens. The scheduler passes
  ///                `prompt_len - 1`, because a sequence whose every token is
  ///                already cached has nothing to run and would never produce
  ///                the logits it needs to generate from. One token must
  ///                always be left to compute.
  Match Acquire(const std::vector<int32_t>& tokens, int64_t limit);

  /// \brief Ends a sequence's relationship with the cache.
  ///
  /// Releases the reference `Acquire` took, then offers the blocks the sequence
  /// computed for itself. Whatever the tree does not keep -- duplicates of a
  /// path some other sequence inserted first, and the partial tail block -- goes
  /// back to the pool.
  ///
  /// \param tokens        The sequence's tokens.
  /// \param cached_tokens How many of them have KV written. Tokens past this
  ///                      have no keys and their blocks hold nothing.
  /// \param blocks        The sequence's block table.
  /// \param acquired      What `Acquire` returned for it, in tokens. Those
  ///                      blocks are the tree's already.
  void Finish(const std::vector<int32_t>& tokens, int64_t cached_tokens,
              const std::vector<int32_t>& blocks, int64_t acquired);

  /// \brief Frees unreferenced blocks, least recently used first.
  ///
  /// Only leaves are evictable, and only unreferenced ones. An interior node is
  /// a prefix of something still cached, so dropping it would strand every
  /// descendant; evicting the leaf may of course turn its parent into one, and
  /// a request for several blocks will walk back up a branch.
  ///
  /// \param blocks How many to reclaim.
  /// \return       How many actually were. Fewer means everything left is
  ///               referenced by a running sequence.
  int64_t Evict(int64_t blocks);

  /// \brief Blocks the tree is holding, referenced or not.
  int64_t cached_blocks() const { return cached_blocks_; }

  /// \brief Blocks held that nothing is currently reading, and so the amount
  ///        `Evict` could reclaim.
  int64_t evictable_blocks() const;

  /// Tokens served from the cache, and tokens that had to be computed. The
  /// ratio is the only number that says whether any of this is paying for
  /// itself on a given workload.
  int64_t hit_tokens() const { return hit_tokens_; }
  int64_t miss_tokens() const { return miss_tokens_; }

 private:
  /// One run of tokens and the blocks holding it.
  ///
  /// The run is always a whole number of blocks. Children are keyed by the first
  /// token of their own run, which is what makes the descent one map lookup per
  /// node rather than a scan.
  struct Node {
    std::vector<int32_t> tokens;
    std::vector<int32_t> blocks;

    Node* parent = nullptr;
    absl::flat_hash_map<int32_t, std::unique_ptr<Node>> children;

    /// Live sequences reading this node. Nonzero pins it against eviction.
    int64_t refs = 0;
    /// LRU stamp, from the cache's own counter.
    uint64_t last_used = 0;
  };

  /// Splits `node` so that its run is `prefix_blocks` long, moving the tail into
  /// a new child. Both halves inherit the refcount, because a sequence that
  /// referenced the whole run now references both pieces of it.
  void Split(Node* node, int64_t prefix_blocks);

  /// Walks `tokens` and applies `visit` to every node on the path, stopping
  /// where the tree does. Returns tokens covered.
  int64_t Walk(const std::vector<int32_t>& tokens, int64_t limit,
               const std::function<void(Node*)>& visit);

  KvBlockPool* pool_ = nullptr;
  int64_t block_size_ = 0;

  std::unique_ptr<Node> root_;

  int64_t cached_blocks_ = 0;
  uint64_t clock_ = 0;

  int64_t hit_tokens_ = 0;
  int64_t miss_tokens_ = 0;
};

}  // namespace inferx::scheduler
