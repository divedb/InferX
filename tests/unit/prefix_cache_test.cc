// The radix prefix cache (§6.3), on a host pool with no device in sight.
//
// Everything here is block bookkeeping and tree surgery, which is exactly the
// kind of thing §3.1's scheduler/executor split exists to make testable without
// a GPU. The properties that matter are ownership properties -- a block is in
// the tree, or held by a sequence, or free, and never two of those -- and they
// are checkable by counting.

#include "inferx/scheduler/prefix_cache.h"

#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/kv_cache.h"

namespace inferx::scheduler {
namespace {

constexpr int64_t kBlockSize = 4;

StatusOr<KvBlockPool> HostPool(int64_t blocks) {
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = 1;
  layout.head_dim = 8;
  layout.dtype = DataType::kBFloat16;

  return KvBlockPool::Create(/*num_layers=*/1, blocks, kBlockSize, layout,
                             DeviceId::Cpu());
}

class PrefixCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto p = HostPool(64);
    ASSERT_TRUE(p.ok()) << p.status();
    pool_ = std::make_unique<KvBlockPool>(*std::move(p));
    cache_ = std::make_unique<PrefixCache>(pool_.get(), kBlockSize);
  }

  /// Tokens 0..n-1, so a prefix of one sequence is a prefix of the other.
  std::vector<int32_t> Seq(int64_t n, int32_t base = 0) {
    std::vector<int32_t> out(static_cast<size_t>(n));
    for (size_t i = 0; i < out.size(); ++i) {
      out[i] = base + static_cast<int32_t>(i);
    }
    return out;
  }

  /// Allocates `n` blocks the way a sequence would.
  std::vector<int32_t> Allocate(int64_t n) {
    std::vector<int32_t> blocks;
    for (int64_t i = 0; i < n; ++i) {
      auto b = pool_->AllocateBlock();
      EXPECT_TRUE(b.ok()) << b.status();
      if (!b.ok()) break;
      blocks.push_back(*b);
    }
    return blocks;
  }

  /// Runs a whole sequence past the cache: match, allocate the rest, hand it
  /// all back. Returns how many tokens the cache supplied.
  int64_t RunOnce(const std::vector<int32_t>& tokens) {
    const int64_t limit = static_cast<int64_t>(tokens.size()) - 1;
    const PrefixCache::Match match = cache_->Acquire(tokens, limit);

    std::vector<int32_t> blocks = match.blocks;

    const int64_t total_blocks =
        (static_cast<int64_t>(tokens.size()) + kBlockSize - 1) / kBlockSize;

    for (const int32_t b :
         Allocate(total_blocks - static_cast<int64_t>(blocks.size()))) {
      blocks.push_back(b);
    }

    // What the scheduler does once admission commits.
    cache_->RecordAdmission(match.tokens, static_cast<int64_t>(tokens.size()));

    cache_->Finish(tokens, static_cast<int64_t>(tokens.size()), blocks,
                   match.tokens);

    return match.tokens;
  }

  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<PrefixCache> cache_;
};

TEST_F(PrefixCacheTest, AnEmptyCacheMatchesNothing) {
  const PrefixCache::Match match = cache_->Acquire(Seq(16), 15);

  EXPECT_EQ(match.tokens, 0);
  EXPECT_TRUE(match.blocks.empty());
  EXPECT_EQ(cache_->cached_blocks(), 0);
}

// The point of the whole structure: the second request for the same prefix does
// not compute it.
TEST_F(PrefixCacheTest, ASecondRequestMatchesTheFirstsPrefix) {
  const std::vector<int32_t> tokens = Seq(16);

  EXPECT_EQ(RunOnce(tokens), 0) << "nothing was cached yet";

  // 16 tokens is 4 blocks; the last token must be left to compute, so the
  // match is capped at 15 tokens and rounds down to 3 blocks.
  EXPECT_EQ(RunOnce(tokens), 12);
}

// A prefix is shared, not a whole sequence. Two requests agreeing on their
// first half should share exactly that half.
TEST_F(PrefixCacheTest, ADivergingSequenceMatchesOnlyTheSharedPart) {
  std::vector<int32_t> first = Seq(16);
  ASSERT_EQ(RunOnce(first), 0);

  // Same first 8 tokens, different after.
  std::vector<int32_t> second = Seq(8);
  for (int32_t i = 0; i < 8; ++i) second.push_back(900 + i);

  EXPECT_EQ(RunOnce(second), 8);
}

// Divergence inside a cached run has to split the node, and the split must not
// lose or duplicate a block.
TEST_F(PrefixCacheTest, DivergenceSplitsANodeWithoutLosingBlocks) {
  ASSERT_EQ(RunOnce(Seq(20)), 0);
  const int64_t after_first = cache_->cached_blocks();
  ASSERT_GT(after_first, 0);

  std::vector<int32_t> second = Seq(8);
  for (int32_t i = 0; i < 12; ++i) second.push_back(500 + i);

  EXPECT_EQ(RunOnce(second), 8) << "should share the first two blocks";

  // The first sequence's run was split rather than copied, so the shared part
  // exists once. Total blocks = shared 2 + first's tail + second's tail.
  EXPECT_GT(cache_->cached_blocks(), after_first);

  // Nothing escaped: every block is either in the tree or free.
  EXPECT_EQ(pool_->used_blocks(), cache_->cached_blocks())
      << "blocks are held by neither the tree nor the free list";
}

// The invariant the whole design rests on: a block is in the tree, held by a
// live sequence, or free -- never lost and never in two places.
TEST_F(PrefixCacheTest, EveryBlockIsAccountedForAcrossManySequences) {
  for (int32_t base = 0; base < 6; ++base) {
    // Overlapping families of prompts, so the tree branches and splits.
    std::vector<int32_t> tokens = Seq(8);
    for (int32_t i = 0; i < 8; ++i) tokens.push_back(base * 100 + i);
    RunOnce(tokens);
  }

  EXPECT_EQ(pool_->used_blocks(), cache_->cached_blocks())
      << "the tree and the pool disagree about who owns what";
  EXPECT_GT(cache_->cached_blocks(), 0);
}

// A referenced prefix cannot be evicted, because a running sequence is reading
// it. Evicting it would hand those blocks to someone else to overwrite.
TEST_F(PrefixCacheTest, EvictionSkipsWhatIsBeingRead) {
  const std::vector<int32_t> tokens = Seq(16);
  ASSERT_EQ(RunOnce(tokens), 0);

  const int64_t cached = cache_->cached_blocks();
  ASSERT_GT(cached, 0);
  EXPECT_EQ(cache_->evictable_blocks(), cached) << "nothing is reading it yet";

  // A live sequence pins the prefix.
  const PrefixCache::Match held = cache_->Acquire(tokens, 15);
  ASSERT_GT(held.tokens, 0);

  EXPECT_EQ(cache_->evictable_blocks(), cached - held.tokens / kBlockSize);
  EXPECT_EQ(cache_->Evict(100), cached - held.tokens / kBlockSize);

  // Released, it becomes evictable again.
  cache_->Finish(tokens, held.tokens, held.blocks, held.tokens);
  EXPECT_EQ(cache_->Evict(100), held.tokens / kBlockSize);

  EXPECT_EQ(cache_->cached_blocks(), 0);
  EXPECT_EQ(pool_->free_blocks(), pool_->num_blocks()) << "eviction leaked";
}

// Eviction is LRU, and it walks back up a branch: taking a leaf can turn its
// parent into one, which is how a request for several blocks is satisfied.
TEST_F(PrefixCacheTest, EvictionTakesTheLeastRecentlyUsedAndClimbs) {
  const std::vector<int32_t> older = Seq(16, /*base=*/0);
  const std::vector<int32_t> newer = Seq(16, /*base=*/400);

  ASSERT_EQ(RunOnce(older), 0);
  ASSERT_EQ(RunOnce(newer), 0);

  // Touch the older one so it becomes the more recently used of the two.
  const PrefixCache::Match touch = cache_->Acquire(older, 15);
  cache_->Finish(older, touch.tokens, touch.blocks, touch.tokens);

  const int64_t before = cache_->cached_blocks();
  ASSERT_GE(before, 6);

  // One block: it must come from `newer`, which is now the stale branch.
  ASSERT_EQ(cache_->Evict(1), 1);

  // `older` still matches in full; `newer` has lost its tail.
  const PrefixCache::Match still = cache_->Acquire(older, 15);
  EXPECT_EQ(still.tokens, 12) << "the recently used branch was evicted";
  cache_->Finish(older, still.tokens, still.blocks, still.tokens);

  // And everything still balances.
  EXPECT_EQ(pool_->used_blocks(), cache_->cached_blocks());
}

// Asking for one block costs one block, not the whole prefix it belonged to.
//
// Evicting whole nodes looks harmless until it meets the case the cache exists
// for. A preemption donates its history and then immediately needs one block
// back; if that one block took the entire donation with it, the sequence's
// prefix is gone by the time it is re-admitted and the hit rate under memory
// pressure is exactly zero. Trimming from the tail leaves a shorter prefix,
// which is still a prefix, and still matches.
TEST_F(PrefixCacheTest, EvictingOneBlockLeavesTheRestOfThePrefixMatchable) {
  const std::vector<int32_t> tokens = Seq(20);
  ASSERT_EQ(RunOnce(tokens), 0);

  // 20 tokens is 5 blocks.
  ASSERT_EQ(cache_->cached_blocks(), 5);

  ASSERT_EQ(cache_->Evict(1), 1) << "evicting one block freed more than one";
  EXPECT_EQ(cache_->cached_blocks(), 4);

  // Four blocks of the prefix survive and still match.
  const PrefixCache::Match match = cache_->Acquire(tokens, 19);
  EXPECT_EQ(match.tokens, 16);
  cache_->Finish(tokens, match.tokens, match.blocks, match.tokens);

  // And the accounting still balances.
  EXPECT_EQ(pool_->used_blocks(), cache_->cached_blocks());
}

// Two prompts that agree on the first token of a block and diverge inside it.
//
// There is nothing shareable there -- blocks go in whole or not at all -- but
// the child slot under that token is already taken, so the second sequence's
// run cannot be attached. It has to be freed instead, and getting that wrong is
// invisible from the outside: `emplace` on an occupied key reports failure by
// returning false and destroying the node it was handed, blocks and all, while
// the block count has already been raised to include them. The tree then claims
// memory the pool never gets back.
//
// Repetitive prompts make this the common case rather than a corner, which is
// how it survived every test whose token streams were all distinct.
TEST_F(PrefixCacheTest, ADivergenceInsideTheFirstBlockLeaksNothing) {
  // Block size is 4. Both start with token 0, then differ at token 1.
  const std::vector<int32_t> first = {0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<int32_t> second = {0, 77, 2, 3, 4, 5, 6, 7};

  ASSERT_EQ(RunOnce(first), 0);
  const int64_t after_first = cache_->cached_blocks();
  ASSERT_GT(after_first, 0);

  // Shares the first token, so the walk finds a child under that key -- and
  // stops, because agreement runs out before the block does.
  EXPECT_EQ(RunOnce(second), 0) << "a partial block was shared";

  // The second sequence's blocks went back to the pool rather than into a tree
  // that cannot hold them.
  EXPECT_EQ(cache_->cached_blocks(), after_first);
  EXPECT_EQ(pool_->used_blocks(), cache_->cached_blocks())
      << "blocks vanished between the tree and the free list";
}

// Nothing is left holding memory once every sequence has finished and the cache
// has been drained.
TEST_F(PrefixCacheTest, DrainingTheCacheReturnsEveryBlock) {
  for (int32_t base = 0; base < 5; ++base) {
    std::vector<int32_t> tokens = Seq(4);
    for (int32_t i = 0; i < 12; ++i) tokens.push_back(base * 50 + i);
    RunOnce(tokens);
  }

  ASSERT_GT(cache_->cached_blocks(), 0);

  cache_->Evict(1000);

  EXPECT_EQ(cache_->cached_blocks(), 0);
  EXPECT_EQ(pool_->free_blocks(), pool_->num_blocks());
}

// Two sequences admitted before either finished compute the same prefix
// independently. Only one copy may end up in the tree; the other's blocks have
// to go back to the pool rather than being cached twice or leaked.
TEST_F(PrefixCacheTest, ConcurrentDuplicatesAreNotCachedTwice) {
  const std::vector<int32_t> tokens = Seq(16);

  // Both miss, because neither has finished.
  const PrefixCache::Match a = cache_->Acquire(tokens, 15);
  const PrefixCache::Match b = cache_->Acquire(tokens, 15);
  ASSERT_EQ(a.tokens, 0);
  ASSERT_EQ(b.tokens, 0);

  const std::vector<int32_t> a_blocks = Allocate(4);
  const std::vector<int32_t> b_blocks = Allocate(4);
  ASSERT_EQ(pool_->used_blocks(), 8);

  cache_->Finish(tokens, 16, a_blocks, 0);
  const int64_t after_a = cache_->cached_blocks();
  EXPECT_EQ(after_a, 4);

  cache_->Finish(tokens, 16, b_blocks, 0);

  EXPECT_EQ(cache_->cached_blocks(), after_a)
      << "the same prefix was cached twice";
  EXPECT_EQ(pool_->used_blocks(), 4) << "the duplicate's blocks were not freed";
}

// A sequence whose every token is already cached would have nothing to run and
// no logits to sample from, so the match is always short of the full sequence.
TEST_F(PrefixCacheTest, AFullMatchAlwaysLeavesATokenToCompute) {
  const std::vector<int32_t> tokens = Seq(16);
  ASSERT_EQ(RunOnce(tokens), 0);

  const PrefixCache::Match match =
      cache_->Acquire(tokens, static_cast<int64_t>(tokens.size()) - 1);

  EXPECT_LT(match.tokens, static_cast<int64_t>(tokens.size()));
  EXPECT_EQ(match.tokens, 12);

  cache_->Finish(tokens, match.tokens, match.blocks, match.tokens);
}

// A sequence shorter than one block has nothing shareable in it at all.
TEST_F(PrefixCacheTest, SequencesShorterThanABlockAreNeverCached) {
  const std::vector<int32_t> tokens = Seq(3);

  EXPECT_EQ(RunOnce(tokens), 0);
  EXPECT_EQ(cache_->cached_blocks(), 0) << "a partial block was shared";
  EXPECT_EQ(RunOnce(tokens), 0);

  EXPECT_EQ(pool_->used_blocks(), 0);
}

// An admission that never happened is not a miss.
//
// A lookup is not an admission: matching can succeed and the admission still
// fail, because there may be no room for the *rest* of the prompt, and the
// scheduler then rolls back and retries on a later step. Counting inside
// `Acquire` charged every retry, so under memory pressure -- where retries are
// most frequent, and where the numbers are most worth trusting -- the reported
// hit rate fell towards zero while the cache was working perfectly well. A
// throughput benchmark showed a rising hit rate and a falling one at the same
// time, which is how this was noticed.
TEST_F(PrefixCacheTest, ARetriedAdmissionIsCountedOnce) {
  const std::vector<int32_t> tokens = Seq(16);
  ASSERT_EQ(RunOnce(tokens), 0);

  const int64_t hits = cache_->hit_tokens();
  const int64_t misses = cache_->miss_tokens();

  // Three lookups that come to nothing, as a request deferred for three steps
  // would produce.
  for (int attempt = 0; attempt < 3; ++attempt) {
    const PrefixCache::Match match = cache_->Acquire(tokens, 15);
    ASSERT_GT(match.tokens, 0);
    cache_->Finish(tokens, match.tokens, match.blocks, match.tokens);
  }

  EXPECT_EQ(cache_->hit_tokens(), hits) << "a failed admission counted as a hit";
  EXPECT_EQ(cache_->miss_tokens(), misses)
      << "a failed admission counted as a miss";
}

// Hit and miss counts are what say whether any of this is paying for itself.
TEST_F(PrefixCacheTest, HitAndMissTokensAreCounted) {
  const std::vector<int32_t> tokens = Seq(16);

  RunOnce(tokens);
  EXPECT_EQ(cache_->hit_tokens(), 0);
  EXPECT_EQ(cache_->miss_tokens(), 16);

  RunOnce(tokens);
  EXPECT_EQ(cache_->hit_tokens(), 12);
  EXPECT_EQ(cache_->miss_tokens(), 20) << "16 the first time, 4 the second";
}

}  // namespace
}  // namespace inferx::scheduler
