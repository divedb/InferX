// CacheManager-level prefix-caching tests: adoption through release, the
// n-1 logits cap, salt isolation, pressure eviction, clear ownership, and
// the disabled path.

#include "inferx/cache/cache_manager.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "inferx/core/dtype.h"

namespace inferx {
namespace {

constexpr int64_t kBlockSize = 4;

struct Fixture {
  std::unique_ptr<KvBlockPool> pool;
  std::unique_ptr<CacheManager> manager;

  explicit Fixture(bool enable_caching, int64_t blocks = 32) {
    KvLayout layout;
    layout.entries_per_token = 2;
    layout.kv_heads = 1;
    layout.head_dim = 8;
    layout.dtype = DataType::kBFloat16;
    pool = std::make_unique<KvBlockPool>(std::move(KvBlockPool::Create(
                                              /*num_layers=*/2, blocks, kBlockSize, layout,
                                              Device::Cpu())
                                              .value()));
    manager = std::make_unique<CacheManager>(pool.get(), enable_caching);
  }
};

std::vector<TokenId> Tokens(int64_t count) {
  std::vector<TokenId> tokens;
  tokens.reserve(count);
  for (int64_t i = 0; i < count; ++i) tokens.push_back(static_cast<TokenId>(100 + i));
  return tokens;
}

TEST(CacheManagerPrefixTest, SecondPromptSkipsCachedPrefixBlocks) {
  Fixture f(true);
  const std::vector<TokenId> prompt = Tokens(12);  // Three full blocks.

  // First request: nothing cached, full admission.
  auto adoption = f.manager->AdoptPrefix(1, prompt, "", /*max_cached_tokens=*/11);
  EXPECT_EQ(adoption.cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 12).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, {}).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 3);

  const std::vector<int32_t> first_blocks(f.manager->Blocks(1).begin(),
                                         f.manager->Blocks(1).end());
  const int64_t free_after_first = f.pool->FreeBlocks();
  ASSERT_TRUE(f.manager->Release(1).ok());
  // Cached blocks keep their bytes: only the pool's raw accounting moves
  // through eviction, so nothing is freed by the release itself.
  EXPECT_EQ(f.pool->FreeBlocks(), free_after_first);
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 3);

  // Second request with the identical prompt: the cap keeps the last prompt
  // token recomputing, so the hit covers exactly two blocks.
  adoption = f.manager->AdoptPrefix(2, prompt, "", /*max_cached_tokens=*/11);
  EXPECT_EQ(adoption.cached_tokens, 8);
  ASSERT_EQ(adoption.block_ids.size(), 2u);
  EXPECT_EQ(adoption.block_ids[0], first_blocks[0]);
  EXPECT_EQ(adoption.block_ids[1], first_blocks[1]);
  // Growth appends only the remaining blocks.
  ASSERT_TRUE(f.manager->Reserve(2, 12).ok());
  const std::vector<int32_t> second_blocks(f.manager->Blocks(2).begin(),
                                          f.manager->Blocks(2).end());
  ASSERT_EQ(second_blocks.size(), 3u);
  EXPECT_EQ(second_blocks[0], first_blocks[0]);
  EXPECT_EQ(second_blocks[1], first_blocks[1]);
  EXPECT_NE(second_blocks[2], first_blocks[2]);
  EXPECT_EQ(f.pool->FreeBlocks(), free_after_first - 1);
}

TEST(CacheManagerPrefixTest, UncappedAdoptionCoversAllFullBlocks) {
  Fixture f(true);
  const std::vector<TokenId> prompt = Tokens(9);  // Two full + partial tail.

  EXPECT_EQ(f.manager->AdoptPrefix(1, prompt, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 9).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, {}).ok());
  ASSERT_TRUE(f.manager->Release(1).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);  // Partial tail stays private.

  const auto adoption = f.manager->AdoptPrefix(2, prompt, "", -1);
  EXPECT_EQ(adoption.cached_tokens, 8);
}

TEST(CacheManagerPrefixTest, PublishesDecodeBlocksAsTheyComplete) {
  Fixture f(true);
  const std::vector<TokenId> prompt = Tokens(4);
  std::vector<TokenId> output;

  ASSERT_TRUE(f.manager->AdoptPrefix(1, prompt, "", 3).cached_tokens == 0);
  ASSERT_TRUE(f.manager->Reserve(1, 4).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, output).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 1);

  // Decode: one output token at a time. The block spanning the prompt and
  // the first three output tokens completes when the fourth arrives.
  for (int step = 0; step < 3; ++step) {
    output.push_back(static_cast<TokenId>(200 + step));
    ASSERT_TRUE(f.manager->Reserve(1, prompt.size() + output.size()).ok());
    ASSERT_TRUE(f.manager->PublishComputed(1, prompt, output).ok());
    EXPECT_EQ(f.manager->CachedPrefixBlocks(), 1);
  }
  output.push_back(203);
  ASSERT_TRUE(f.manager->Reserve(1, prompt.size() + output.size()).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, output).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);
}

TEST(CacheManagerPrefixTest, SaltIsolatesChains) {
  Fixture f(true);
  const std::vector<TokenId> prompt = Tokens(8);

  EXPECT_EQ(f.manager->AdoptPrefix(1, prompt, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 8).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, {}).ok());
  ASSERT_TRUE(f.manager->Release(1).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);

  // Same tokens, different salt: shares nothing with the cached blocks.
  const auto adoption = f.manager->AdoptPrefix(2, prompt, "tenant-b", -1);
  EXPECT_EQ(adoption.cached_tokens, 0);
  EXPECT_TRUE(adoption.block_ids.empty());
}

TEST(CacheManagerPrefixTest, ReserveEvictsBeforeFailing) {
  // Room for exactly one request's blocks plus nothing else.
  Fixture f(true, /*blocks=*/4);
  const std::vector<TokenId> prompt_a = Tokens(8);
  EXPECT_EQ(f.manager->AdoptPrefix(1, prompt_a, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 8).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt_a, {}).ok());
  ASSERT_TRUE(f.manager->Release(1).ok());
  ASSERT_EQ(f.manager->CachedPrefixBlocks(), 2);
  ASSERT_EQ(f.pool->FreeBlocks(), 2);

  // A fresh request needing all four blocks succeeds by evicting the two
  // cached ones -- pressure reclaims before it fails.
  const std::vector<TokenId> prompt_b = Tokens(16);
  ASSERT_TRUE(f.manager->Reserve(2, 16).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 0);
  EXPECT_EQ(f.pool->FreeBlocks(), 0);

  // With nothing evictable left, further growth fails cleanly.
  EXPECT_FALSE(f.manager->Reserve(2, 20).ok());
}

TEST(CacheManagerPrefixTest, DisabledManagerKeepsLegacyBehavior) {
  Fixture f(false);
  const std::vector<TokenId> prompt = Tokens(8);

  const auto adoption = f.manager->AdoptPrefix(1, prompt, "", -1);
  EXPECT_EQ(adoption.cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 8).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, {}).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 0);

  const int64_t free_before = f.pool->FreeBlocks();
  ASSERT_TRUE(f.manager->Release(1).ok());
  EXPECT_EQ(f.pool->FreeBlocks(), free_before + 2);  // Immediate free, no cache.
}

TEST(CacheManagerPrefixTest, ClearPreservesSharedBlockOwnership) {
  // C01: two requests share the first two blocks through the prefix index.
  // Clearing the index must not destroy that shared ownership: releasing one
  // sharer may not free blocks the other still reads, and a fresh request
  // must never be handed a live sharer's block.
  Fixture f(true, /*blocks=*/8);
  const std::vector<TokenId> shared = Tokens(8);  // Two full blocks.
  std::vector<TokenId> prompt_a = shared;
  prompt_a.push_back(900);  // Private partial tail.
  std::vector<TokenId> prompt_b = shared;
  prompt_b.push_back(901);  // A different private tail.

  // Request 1 computes and publishes the shared prefix.
  EXPECT_EQ(f.manager->AdoptPrefix(1, prompt_a, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 9).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt_a, {}).ok());
  const std::vector<int32_t> blocks_a(f.manager->Blocks(1).begin(), f.manager->Blocks(1).end());
  ASSERT_EQ(blocks_a.size(), 3u);

  // Request 2 adopts the shared blocks and grows a private tail of its own.
  const auto adoption_b = f.manager->AdoptPrefix(2, prompt_b, "", -1);
  ASSERT_EQ(adoption_b.cached_tokens, 8);
  ASSERT_EQ(adoption_b.block_ids.size(), 2u);
  EXPECT_EQ(adoption_b.block_ids[0], blocks_a[0]);
  EXPECT_EQ(adoption_b.block_ids[1], blocks_a[1]);
  ASSERT_TRUE(f.manager->Reserve(2, 9).ok());
  const std::vector<int32_t> blocks_b(f.manager->Blocks(2).begin(), f.manager->Blocks(2).end());
  ASSERT_EQ(blocks_b.size(), 3u);
  ASSERT_EQ(f.pool->FreeBlocks(), 4);  // Six blocks held, two of them shared.

  f.manager->ClearPrefixCache();
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 0);

  // Releasing one sharer returns only its private tail; the shared blocks
  // stay alive for the other sharer.
  ASSERT_TRUE(f.manager->Release(1).ok());
  EXPECT_EQ(f.pool->FreeBlocks(), 5);

  // A fresh request must never receive any block request 2 still holds.
  const auto granted = f.manager->Reserve(3, 12);
  ASSERT_TRUE(granted.ok());
  ASSERT_EQ(granted->size(), 3u);
  for (const int32_t block : *granted) {
    EXPECT_NE(block, blocks_b[0]);
    EXPECT_NE(block, blocks_b[1]);
    EXPECT_NE(block, blocks_b[2]);
  }

  // The surviving sharer's release frees exactly its own blocks, once.
  ASSERT_TRUE(f.manager->Release(2).ok());
  EXPECT_EQ(f.pool->FreeBlocks(), 5);  // Eight blocks minus request 3's three.
}

TEST(CacheManagerPrefixTest, InFlightRequestDoesNotRepublishAfterClear) {
  // A clear models a weight change: KV computed before it must never reenter
  // the index through an in-flight request's later publications.
  Fixture f(true);
  const std::vector<TokenId> prompt = Tokens(8);
  std::vector<TokenId> output;

  EXPECT_EQ(f.manager->AdoptPrefix(1, prompt, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 8).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, output).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);

  f.manager->ClearPrefixCache();
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 0);

  // The request keeps decoding and completes further blocks after the clear;
  // none of its pre-clear KV may become a cache hit.
  for (int step = 0; step < 4; ++step) {
    output.push_back(static_cast<TokenId>(300 + step));
    ASSERT_TRUE(
        f.manager->Reserve(1, static_cast<int64_t>(prompt.size() + output.size())).ok());
    ASSERT_TRUE(f.manager->PublishComputed(1, prompt, output).ok());
    EXPECT_EQ(f.manager->CachedPrefixBlocks(), 0);
  }

  // A fresh request publishes into the cleared index normally.
  const std::vector<TokenId> prompt_b = Tokens(9);
  EXPECT_EQ(f.manager->AdoptPrefix(2, prompt_b, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(2, 9).ok());
  ASSERT_TRUE(f.manager->PublishComputed(2, prompt_b, {}).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);

  // The pre-clear request's blocks return exactly once, without touching the
  // fresh entries.
  ASSERT_TRUE(f.manager->Release(1).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);
}

TEST(CacheManagerPrefixTest, ClearIsRepeatableAndFreshRequestsAdoptAfterwards) {
  Fixture f(true);
  const std::vector<TokenId> prompt = Tokens(8);

  // Seed the index, then clear twice in a row.
  EXPECT_EQ(f.manager->AdoptPrefix(1, prompt, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(1, 8).ok());
  ASSERT_TRUE(f.manager->PublishComputed(1, prompt, {}).ok());
  ASSERT_TRUE(f.manager->Release(1).ok());
  ASSERT_EQ(f.manager->CachedPrefixBlocks(), 2);

  f.manager->ClearPrefixCache();
  f.manager->ClearPrefixCache();
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 0);

  // The old hashes miss, and publication into the cleared index works.
  EXPECT_EQ(f.manager->AdoptPrefix(2, prompt, "", -1).cached_tokens, 0);
  ASSERT_TRUE(f.manager->Reserve(2, 8).ok());
  ASSERT_TRUE(f.manager->PublishComputed(2, prompt, {}).ok());
  EXPECT_EQ(f.manager->CachedPrefixBlocks(), 2);

  // A third request adopts the post-clear entries.
  const std::vector<int32_t> blocks_b(f.manager->Blocks(2).begin(), f.manager->Blocks(2).end());
  const auto adoption = f.manager->AdoptPrefix(3, prompt, "", 8);
  EXPECT_EQ(adoption.cached_tokens, 8);
  ASSERT_EQ(adoption.block_ids.size(), 2u);
  EXPECT_EQ(adoption.block_ids[0], blocks_b[0]);
  EXPECT_EQ(adoption.block_ids[1], blocks_b[1]);
}

}  // namespace
}  // namespace inferx
