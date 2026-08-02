#include "inferx/scheduler/prefix_cache.h"

#include <algorithm>
#include <utility>

namespace inferx::scheduler {

PrefixCache::PrefixCache(KvBlockPool* pool, int64_t block_size)
    : pool_(pool), block_size_(block_size), root_(std::make_unique<Node>()) {}

PrefixCache::~PrefixCache() {
  // The tree owns its blocks, so it has to give them back. Nothing else can:
  // a cached block is not on the free list and no sequence's table claims it,
  // so a cache that simply vanished would strand every block it held in a pool
  // that outlives it.
  const std::function<void(Node*)> release = [&](Node* node) {
    if (!node->blocks.empty()) (void)pool_->FreeBlocks(node->blocks);
    for (auto& [key, child] : node->children) release(child.get());
  };

  release(root_.get());
}

namespace {

/// How many leading tokens of `a` and `b` agree, capped at `limit`.
int64_t CommonPrefix(const std::vector<int32_t>& a, int64_t a_from,
                     const std::vector<int32_t>& b, int64_t limit) {
  int64_t n = 0;
  while (n < limit && a_from + n < static_cast<int64_t>(a.size()) &&
         n < static_cast<int64_t>(b.size()) &&
         a[static_cast<size_t>(a_from + n)] == b[static_cast<size_t>(n)]) {
    ++n;
  }
  return n;
}

}  // namespace

void PrefixCache::Split(Node* node, int64_t prefix_blocks) {
  const int64_t prefix_tokens = prefix_blocks * block_size_;

  auto tail = std::make_unique<Node>();
  tail->tokens.assign(node->tokens.begin() + prefix_tokens, node->tokens.end());
  tail->blocks.assign(node->blocks.begin() + prefix_blocks, node->blocks.end());
  tail->parent = node;

  // Both halves inherit the reference count and the stamp. A sequence that was
  // reading the whole run is now reading two nodes, and dropping the count on
  // either would let half of what it is reading be evicted out from under it.
  tail->refs = node->refs;
  tail->last_used = node->last_used;

  tail->children = std::move(node->children);
  for (auto& [key, child] : tail->children) child->parent = tail.get();

  node->tokens.resize(static_cast<size_t>(prefix_tokens));
  node->blocks.resize(static_cast<size_t>(prefix_blocks));

  node->children.clear();
  const int32_t key = tail->tokens.front();
  node->children.emplace(key, std::move(tail));
}

int64_t PrefixCache::Walk(const std::vector<int32_t>& tokens, int64_t limit,
                          const std::function<void(Node*)>& visit) {
  int64_t matched = 0;
  Node* node = root_.get();

  while (matched < limit) {
    const auto it = node->children.find(tokens[static_cast<size_t>(matched)]);
    if (it == node->children.end()) break;

    Node* child = it->second.get();

    const int64_t common = CommonPrefix(tokens, matched, child->tokens,
                                        limit - matched);

    // Partial agreement inside a node is not a match: the blocks are shared
    // whole, so half of one is worth nothing. Round down to the last block
    // boundary and stop there.
    const int64_t usable = (common / block_size_) * block_size_;

    if (usable == 0) break;

    if (usable < static_cast<int64_t>(child->tokens.size())) {
      // The query agrees with part of this node and then diverges. Split so the
      // agreed part is a node of its own, which is what the query references
      // and what the next insert branches from.
      Split(child, usable / block_size_);
    }

    visit(child);
    matched += static_cast<int64_t>(child->tokens.size());
    node = child;

    // A split just made `child`'s run exactly the agreed part, so this loop
    // continues only when the whole node matched.
  }

  return matched;
}

PrefixCache::Match PrefixCache::Acquire(const std::vector<int32_t>& tokens,
                                        int64_t limit) {
  Match match;

  // Whole blocks only, and never the whole sequence -- the caller needs at
  // least one token left to compute logits from.
  limit = std::min(limit, static_cast<int64_t>(tokens.size()));
  limit = (std::max<int64_t>(limit, 0) / block_size_) * block_size_;

  if (limit == 0) return match;

  ++clock_;

  match.tokens = Walk(tokens, limit, [&](Node* node) {
    ++node->refs;
    node->last_used = clock_;
    match.blocks.insert(match.blocks.end(), node->blocks.begin(),
                        node->blocks.end());
  });

  return match;
}

void PrefixCache::Finish(const std::vector<int32_t>& tokens,
                         int64_t cached_tokens,
                         const std::vector<int32_t>& blocks,
                         int64_t acquired) {
  // Give back what Acquire pinned. Walked by tokens rather than remembered,
  // which stays correct across the splits another sequence's insert may have
  // done in between: splitting preserves the token sequence, so the same walk
  // reaches the same blocks through more nodes than it did before.
  if (acquired > 0) {
    ++clock_;
    (void)Walk(tokens, acquired, [&](Node* node) {
      if (node->refs > 0) --node->refs;
      node->last_used = clock_;
    });
  }

  // Only complete blocks are shareable, and only tokens that actually have KV.
  const int64_t complete_blocks =
      std::min(std::max<int64_t>(cached_tokens, 0) / block_size_,
               static_cast<int64_t>(blocks.size()));
  const int64_t acquired_blocks = acquired / block_size_;

  // Everything past the shareable part belongs to nobody now. Never past the
  // acquired prefix either, even if `cached_tokens` says otherwise: those
  // blocks are the tree's, and a preempted sequence arrives here having just
  // had its `cached` reset to zero.
  const int64_t keep = std::min(std::max(complete_blocks, acquired_blocks),
                                static_cast<int64_t>(blocks.size()));

  std::vector<int32_t> discard(blocks.begin() + keep, blocks.end());

  if (complete_blocks > acquired_blocks) {
    // Insert the run this sequence computed for itself. The walk re-descends
    // the part the tree already has -- the acquired prefix, plus anything
    // another sequence contributed while this one was running -- and whatever
    // is left becomes a new node hanging off the deepest one it reached.
    Node* deepest = root_.get();

    const int64_t have = Walk(tokens, complete_blocks * block_size_,
                              [&](Node* node) { deepest = node; });

    const int64_t have_blocks = std::min(have / block_size_, complete_blocks);

    // The tree reaches further than this sequence acquired, which means some
    // other sequence cached the same prefix while this one was running. Our
    // copies of those blocks are duplicates -- the tree's hold the same keys
    // and are already shared -- so they go back to the pool. Missing this
    // stranded them: cached by nobody, freed by nobody, owned by a sequence
    // that no longer exists.
    for (int64_t b = acquired_blocks; b < have_blocks; ++b) {
      discard.push_back(blocks[static_cast<size_t>(b)]);
    }

    if (have_blocks < complete_blocks) {
      const int32_t key = tokens[static_cast<size_t>(have)];

      // The slot may already be taken. `Walk` stops short of a child whose run
      // agrees on its first token but diverges before the end of the block, and
      // blocks are shared whole or not at all, so there is genuinely nothing to
      // share down that branch -- but the key is spoken for.
      //
      // Inserting anyway is what `emplace` makes look harmless: it returns
      // false, keeps the existing child, and quietly destroys the node handed
      // to it along with every block inside. The count had already been
      // incremented, so the tree reported blocks it did not have and the pool
      // never got them back.
      const bool taken = deepest->children.find(key) != deepest->children.end();

      if (taken) {
        for (int64_t b = have_blocks; b < complete_blocks; ++b) {
          discard.push_back(blocks[static_cast<size_t>(b)]);
        }
      } else {
        auto node = std::make_unique<Node>();

        node->tokens.assign(tokens.begin() + have,
                            tokens.begin() + complete_blocks * block_size_);
        node->blocks.assign(blocks.begin() + have_blocks,
                            blocks.begin() + complete_blocks);

        node->parent = deepest;
        node->last_used = ++clock_;

        cached_blocks_ += static_cast<int64_t>(node->blocks.size());

        deepest->children.emplace(key, std::move(node));
      }
    }
  }

  if (!discard.empty()) (void)pool_->FreeBlocks(discard);
}

int64_t PrefixCache::evictable_blocks() const {
  int64_t total = 0;

  const std::function<void(const Node*)> count = [&](const Node* node) {
    if (node->children.empty()) {
      if (node->refs == 0) total += static_cast<int64_t>(node->blocks.size());
      return;
    }
    for (const auto& [key, child] : node->children) count(child.get());
  };

  count(root_.get());
  return total;
}

int64_t PrefixCache::Evict(int64_t blocks) {
  int64_t freed = 0;

  while (freed < blocks) {
    // The least recently used unreferenced leaf. A linear walk, which T4 says
    // to revisit only if it shows up in a profile: the tree has one node per
    // distinct cached prefix, not one per block, and eviction runs on
    // allocation failure rather than per step.
    Node* victim = nullptr;

    const std::function<void(Node*)> find = [&](Node* node) {
      if (node->children.empty()) {
        if (node != root_.get() && node->refs == 0 &&
            (victim == nullptr || node->last_used < victim->last_used)) {
          victim = node;
        }
        return;
      }
      for (auto& [key, child] : node->children) find(child.get());
    };

    find(root_.get());

    // Everything left is being read by a running sequence.
    if (victim == nullptr) break;

    // Trimmed from the tail rather than dropped whole. The end of a prefix is
    // the least useful part of it, and what is left after trimming is still a
    // prefix somebody can match -- so asking for one block costs one block,
    // not the entire cached prompt it happened to be sitting in. Evicting whole
    // nodes made a single preemption reclaim its own donation immediately and
    // reduced the hit rate to zero under exactly the pressure the cache is
    // there for.
    // Taken before the trim, which empties `tokens` along with `blocks`.
    const int32_t key = victim->tokens.front();

    while (freed < blocks && !victim->blocks.empty()) {
      (void)pool_->FreeBlock(victim->blocks.back());
      victim->blocks.pop_back();
      victim->tokens.resize(victim->tokens.size() -
                            static_cast<size_t>(block_size_));
      ++freed;
      --cached_blocks_;
      ++evicted_blocks_;
    }

    if (!victim->blocks.empty()) continue;

    // Emptied, so it goes. That may turn its parent into a leaf, which the next
    // pass will consider -- how a request for many blocks walks back up a
    // branch rather than stopping at its tip.
    victim->parent->children.erase(key);
  }

  return freed;
}

}  // namespace inferx::scheduler
