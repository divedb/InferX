#include "inferx/cache/cache_manager.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace inferx {
namespace {

constexpr int64_t kBlockSize = 16;

CacheComponent PerToken(int64_t bytes_per_token) {
  return CacheComponent::PerToken(bytes_per_token, DataType::kBFloat16);
}

/// `count` deterministic prompt tokens starting at `base`.
std::vector<TokenId> Prompt(int64_t count, int base = 100) {
  std::vector<TokenId> tokens(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    tokens[static_cast<size_t>(i)] = base + static_cast<TokenId>(i);
  }
  return tokens;
}

StatusOr<CachePlan> MakePlan(std::vector<CacheSpec> layers) {
  CacheRequirements requirements;
  for (CacheSpec& spec : layers) requirements.layers.push_back(std::move(spec));
  return PlanCacheGroups(requirements, kBlockSize);
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(CacheManagerCreateTest, ValidatesPlanAgainstThePools) {
  auto full_plan = MakePlan({CacheSpec::FullAttention({PerToken(64)})});
  ASSERT_TRUE(full_plan.ok()) << full_plan.status();

  // No groups at all.
  EXPECT_EQ(CacheManager::Create(nullptr, nullptr, CachePlan{}, true).status().code(),
            absl::StatusCode::kInvalidArgument);

  // Token-wise plan without a block pool.
  EXPECT_EQ(
      CacheManager::Create(nullptr, nullptr, *full_plan, true).status().code(),
      absl::StatusCode::kInvalidArgument);

  // Plan block size differs from the pool's.
  auto odd_pool = KvBlockPool::Create(1, 4, kBlockSize * 2, PerToken(64), Device::Cpu());
  ASSERT_TRUE(odd_pool.ok()) << odd_pool.status();
  EXPECT_EQ(
      CacheManager::Create(&*odd_pool, nullptr, *full_plan, true).status().code(),
      absl::StatusCode::kInvalidArgument);

  // Recurrent plan without a state pool; a null block pool is fine when the
  // plan has no token-wise groups.
  auto recurrent_plan =
      MakePlan({CacheSpec::Recurrent({CacheComponent::FixedSlot(1024, DataType::kFloat32)},
                                     kBlockSize)});
  ASSERT_TRUE(recurrent_plan.ok()) << recurrent_plan.status();
  EXPECT_EQ(
      CacheManager::Create(nullptr, nullptr, *recurrent_plan, true).status().code(),
      absl::StatusCode::kInvalidArgument);
  auto state_pool = RecurrentStatePool::Create(*recurrent_plan->RecurrentGroup(), 2,
                                               Device::Cpu());
  ASSERT_TRUE(state_pool.ok()) << state_pool.status();
  EXPECT_TRUE(CacheManager::Create(nullptr, &*state_pool, *recurrent_plan, true).ok());
}

// ---------------------------------------------------------------------------
// Full-attention model
// ---------------------------------------------------------------------------

class CacheManagerFullTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto plan = MakePlan({CacheSpec::FullAttention({PerToken(64)})});
    ASSERT_TRUE(plan.ok()) << plan.status();
    auto blocks = KvBlockPool::Create(1, 8, kBlockSize, PerToken(64), Device::Cpu());
    ASSERT_TRUE(blocks.ok()) << blocks.status();
    pool_ = std::make_unique<KvBlockPool>(std::move(*blocks));
    auto manager = CacheManager::Create(Pool(), nullptr, *plan, /*prefix=*/true);
    ASSERT_TRUE(manager.ok()) << manager.status();
    manager_ = std::move(*manager);
  }

  KvBlockPool* Pool() { return pool_.get(); }

  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<CacheManager> manager_;
};

TEST_F(CacheManagerFullTest, ReserveGrowsViewsAndCounters) {
  EXPECT_EQ(manager_->NumTokenWiseGroups(), 1);
  EXPECT_FALSE(manager_->HasRecurrentGroup());
  EXPECT_EQ(manager_->BlockSize(), kBlockSize);
  EXPECT_EQ(manager_->NumBlocks(), 8);
  EXPECT_EQ(manager_->NumFreeBlocks(), 8);

  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 32}).ok());
  const absl::Span<const BlockId> blocks = manager_->GroupBlocks(1, 0);
  ASSERT_EQ(blocks.size(), 2u);
  EXPECT_NE(blocks[0], kNullBlockId);  // 0 is the reserved null block
  EXPECT_EQ(manager_->NumFreeBlocks(), 6);
  EXPECT_DOUBLE_EQ(manager_->Usage(), 0.25);

  // Growing asks only for the delta.
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{32, 48}).ok());
  EXPECT_EQ(manager_->GroupBlocks(1, 0).size(), 3u);

  // Full attention never skips.
  EXPECT_EQ(manager_->GroupSkippedTokens(0, 1234), 0);

  // Unknown ids and out-of-range groups have empty views.
  EXPECT_TRUE(manager_->GroupBlocks(99, 0).empty());
  EXPECT_TRUE(manager_->GroupBlocks(1, 1).empty());
}

TEST_F(CacheManagerFullTest, ReleaseIsIdempotentAndDestructorDrains) {
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 16}).ok());
  ASSERT_TRUE(manager_->Release(1).ok());
  EXPECT_EQ(manager_->NumFreeBlocks(), 8);
  EXPECT_TRUE(manager_->GroupBlocks(1, 0).empty());
  // Unknown or already-released ids are no-ops.
  EXPECT_TRUE(manager_->Release(1).ok());
  EXPECT_TRUE(manager_->Release(99).ok());

  // An outstanding request is drained by the destructor, not leaked.
  ASSERT_TRUE(manager_->Reserve(2, GrowthTarget{0, 16}).ok());
  manager_.reset();
  EXPECT_EQ(pool_->NumFreeBlocks(), 8);
}

TEST_F(CacheManagerFullTest, ExhaustionLeavesRequestUnchanged) {
  auto tiny = MakePlan({CacheSpec::FullAttention({PerToken(64)})});
  ASSERT_TRUE(tiny.ok()) << tiny.status();
  auto small_pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(small_pool.ok()) << small_pool.status();
  auto manager = CacheManager::Create(&*small_pool, nullptr, *tiny, true);
  ASSERT_TRUE(manager.ok()) << manager.status();

  // Three blocks cannot fit, and the capacity check knows it.
  EXPECT_FALSE((*manager)->CanReserve(1, GrowthTarget{0, 48}));
  const Status failed = (*manager)->Reserve(1, GrowthTarget{0, 48});
  EXPECT_EQ(failed.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_TRUE((*manager)->GroupBlocks(1, 0).empty());
  EXPECT_EQ(small_pool->NumFreeBlocks(), 2);

  // Once the pool really is full, CanReserve turns false.
  ASSERT_TRUE((*manager)->Reserve(1, GrowthTarget{0, 32}).ok());
  EXPECT_FALSE((*manager)->CanReserve(1, GrowthTarget{0, 48}));
}

TEST_F(CacheManagerFullTest, AdoptPublishRoundtripWithStatsAndSalt) {
  const std::vector<TokenId> prompt = Prompt(40);  // 2 full blocks + 8 remainder

  // Producer: compute the prompt, publish its full blocks, finish.
  auto first = manager_->AdoptPrefix(1, prompt, /*salt=*/"", /*max=*/40);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->cached_tokens, 0);
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 40}).ok());
  ASSERT_TRUE(manager_->PublishComputed(1, prompt, {}, 40).ok());
  EXPECT_EQ(manager_->NumCachedBlocks(), 2);
  ASSERT_TRUE(manager_->Release(1).ok());

  // Consumer with the same prompt and salt adopts both blocks.
  auto second = manager_->AdoptPrefix(2, prompt, "", 40);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ(second->cached_tokens, 2 * kBlockSize);
  ASSERT_EQ(second->group_block_ids.size(), 1u);
  EXPECT_EQ(second->group_block_ids[0].size(), 2u);
  EXPECT_EQ(manager_->GroupBlocks(2, 0), second->group_block_ids[0]);
  // The adopted blocks are pinned, not free.
  EXPECT_EQ(manager_->NumCachedBlocks(), 2);
  EXPECT_EQ(manager_->NumFreeBlocks(), 6);  // tail freed; 2 adopted blocks pinned

  // Growing past the hit only needs the uncovered tail.
  ASSERT_TRUE(manager_->Reserve(2, GrowthTarget{32, 40}).ok());
  EXPECT_EQ(manager_->GroupBlocks(2, 0).size(), 3u);

  // A different salt is a different tenant: same prompt, no hit.
  auto tenant = manager_->AdoptPrefix(3, prompt, "tenant-b", 40);
  ASSERT_TRUE(tenant.ok()) << tenant.status();
  EXPECT_EQ(tenant->cached_tokens, 0);
  EXPECT_TRUE(tenant->group_block_ids.empty());

  // The caller's n-1 cap and whole-block truncation bound the adoption.
  auto capped = manager_->AdoptPrefix(4, prompt, "", kBlockSize);
  ASSERT_TRUE(capped.ok()) << capped.status();
  EXPECT_EQ(capped->cached_tokens, kBlockSize);
  EXPECT_EQ(capped->group_block_ids[0].size(), 1u);

  // Duplicate registration is rejected.
  EXPECT_EQ(manager_->AdoptPrefix(2, prompt, "", 40).status().code(),
            absl::StatusCode::kInvalidArgument);

  EXPECT_EQ(manager_->stats().prefix_queries, 4);
  EXPECT_EQ(manager_->stats().prefix_hits, 2);  // `second` and `capped`
  EXPECT_EQ(manager_->stats().prefix_queried_tokens, 4 * 40);
  EXPECT_EQ(manager_->stats().prefix_hit_tokens, 32 + 16);
}

TEST_F(CacheManagerFullTest, PublishComputedValidatesTheRequest) {
  const std::vector<TokenId> prompt = Prompt(32);
  EXPECT_EQ(manager_->PublishComputed(7, prompt, {}, 32).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(CacheManagerFullTest, ForkSharesBlocksAndQueuesCopyOnWriteWork) {
  const std::vector<TokenId> prompt = Prompt(20);  // 2 blocks, tail partial
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 20}).ok());
  const absl::Span<const BlockId> parent_blocks = manager_->GroupBlocks(1, 0);
  ASSERT_EQ(parent_blocks.size(), 2u);

  CacheManager::RunnerWork work = manager_->TakeRunnerWork();
  EXPECT_TRUE(work.block_copies.empty());

  ASSERT_TRUE(manager_->Fork(1, {10, 11}).ok());
  for (const uint64_t child : {10u, 11u}) {
    EXPECT_EQ(manager_->GroupBlocks(child, 0), parent_blocks);
  }

  // Writing into the shared partial block duplicates it for the writer.
  ASSERT_TRUE(manager_->Reserve(10, GrowthTarget{20, 24}).ok());
  work = manager_->TakeRunnerWork();
  ASSERT_EQ(work.block_copies.size(), 1u);
  EXPECT_EQ(work.block_copies[0].src, parent_blocks[1]);
  EXPECT_EQ(manager_->GroupBlocks(10, 0)[0], parent_blocks[0]);
  EXPECT_NE(manager_->GroupBlocks(10, 0)[1], parent_blocks[1]);
  // The parent and the other child still share the original.
  EXPECT_EQ(manager_->GroupBlocks(1, 0)[1], parent_blocks[1]);
  EXPECT_EQ(manager_->GroupBlocks(11, 0)[1], parent_blocks[1]);

  // Child ids must be fresh and non-empty; the parent must exist.
  EXPECT_EQ(manager_->Fork(1, {10}).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->Fork(1, {}).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->Fork(99, {12}).code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CacheManagerFullTest, ResetPrefixCacheRequiresDrainedRequests) {
  const std::vector<TokenId> prompt = Prompt(32);
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 32}).ok());
  ASSERT_TRUE(manager_->PublishComputed(1, prompt, {}, 32).ok());
  ASSERT_TRUE(manager_->Release(1).ok());
  EXPECT_EQ(manager_->NumCachedBlocks(), 2);

  // The entries are live again once another request adopts them.
  auto adopted = manager_->AdoptPrefix(2, prompt, "", 32);
  ASSERT_TRUE(adopted.ok()) << adopted.status();
  EXPECT_EQ(adopted->cached_tokens, 32);
  EXPECT_EQ(manager_->ResetPrefixCache().code(), absl::StatusCode::kFailedPrecondition);

  ASSERT_TRUE(manager_->Release(2).ok());
  ASSERT_TRUE(manager_->ResetPrefixCache().ok());
  EXPECT_EQ(manager_->NumCachedBlocks(), 0);
}

TEST(CacheManagerNoPrefixTest, CachingDisabledStillRegistersAndGrows) {
  auto plan = MakePlan({CacheSpec::FullAttention({PerToken(64)})});
  ASSERT_TRUE(plan.ok()) << plan.status();
  auto blocks = KvBlockPool::Create(1, 4, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(blocks.ok()) << blocks.status();
  auto manager = CacheManager::Create(&*blocks, nullptr, *plan, /*prefix=*/false);
  ASSERT_TRUE(manager.ok()) << manager.status();

  EXPECT_FALSE((*manager)->PrefixCachingEnabled());

  const std::vector<TokenId> prompt = Prompt(32);
  auto adoption = (*manager)->AdoptPrefix(1, prompt, "", 32);
  ASSERT_TRUE(adoption.ok()) << adoption.status();
  EXPECT_EQ(adoption->cached_tokens, 0);
  EXPECT_TRUE(adoption->group_block_ids.empty());
  EXPECT_EQ((*manager)->stats().prefix_queries, 0);  // not even counted

  // The request is registered and grows normally.
  ASSERT_TRUE((*manager)->Reserve(1, GrowthTarget{0, 32}).ok());
  ASSERT_TRUE((*manager)->PublishComputed(1, prompt, {}, 32).ok());
  EXPECT_EQ((*manager)->NumCachedBlocks(), 0);
  EXPECT_EQ((*manager)->GroupBlocks(1, 0).size(), 2u);
}

// ---------------------------------------------------------------------------
// Sliding-window model
// ---------------------------------------------------------------------------

TEST(CacheManagerWindowTest, ReserveRollsOffBelowTheWindow) {
  auto plan = MakePlan({CacheSpec::SlidingWindow({PerToken(64)}, 32)});
  ASSERT_TRUE(plan.ok()) << plan.status();
  auto blocks = KvBlockPool::Create(1, 16, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(blocks.ok()) << blocks.status();
  auto manager = CacheManager::Create(&*blocks, nullptr, *plan, true);
  ASSERT_TRUE(manager.ok()) << manager.status();

  ASSERT_TRUE((*manager)->Reserve(1, GrowthTarget{0, 128}).ok());
  EXPECT_EQ((*manager)->GroupBlocks(1, 0).size(), 8u);
  EXPECT_EQ((*manager)->NumFreeBlocks(), 8);

  // The next step first rolls the window forward: six blocks fall below the
  // retention floor at 128 and free before the step's block is reserved.
  ASSERT_TRUE((*manager)->Reserve(1, GrowthTarget{128, 130}).ok());
  const absl::Span<const BlockId> table = (*manager)->GroupBlocks(1, 0);
  ASSERT_EQ(table.size(), 9u);
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(table[static_cast<size_t>(i)], kNullBlockId) << "entry " << i;
  }
  EXPECT_NE(table[6], kNullBlockId);
  EXPECT_EQ((*manager)->NumFreeBlocks(), 13);  // 16 usable, 3 held after roll + growth

  // The runner's window_left comes from the skipped-token view.
  EXPECT_EQ((*manager)->GroupSkippedTokens(0, 129), 129 - 32 + 1);
}

// ---------------------------------------------------------------------------
// Hybrid model (full attention + recurrent)
// ---------------------------------------------------------------------------

class CacheManagerHybridTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto plan = MakePlan({CacheSpec::FullAttention({PerToken(64)}),
                          CacheSpec::Recurrent(
                              {CacheComponent::FixedSlot(1024, DataType::kFloat32)},
                              kBlockSize)});
    ASSERT_TRUE(plan.ok()) << plan.status();
    plan_ = std::make_unique<CachePlan>(std::move(*plan));
    auto blocks = KvBlockPool::Create(1, 8, kBlockSize, PerToken(64), Device::Cpu());
    ASSERT_TRUE(blocks.ok()) << blocks.status();
    pool_ = std::make_unique<KvBlockPool>(std::move(*blocks));
    auto states =
        RecurrentStatePool::Create(*plan_->RecurrentGroup(), 2, Device::Cpu());
    ASSERT_TRUE(states.ok()) << states.status();
    state_pool_ = std::make_unique<RecurrentStatePool>(std::move(*states));
    auto manager = CacheManager::Create(Pool(), StatePool(), *plan_, true);
    ASSERT_TRUE(manager.ok()) << manager.status();
    manager_ = std::move(*manager);
  }

  KvBlockPool* Pool() { return pool_.get(); }
  RecurrentStatePool* StatePool() { return state_pool_.get(); }

  std::unique_ptr<CachePlan> plan_;
  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<RecurrentStatePool> state_pool_;
  std::unique_ptr<CacheManager> manager_;
};

TEST_F(CacheManagerHybridTest, ReserveGrantsBlocksAndStateSlotTogether) {
  EXPECT_EQ(manager_->NumTokenWiseGroups(), 1);
  EXPECT_TRUE(manager_->HasRecurrentGroup());
  EXPECT_EQ(manager_->NumFreeSlots(), 2);

  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 16}).ok());
  EXPECT_EQ(manager_->GroupBlocks(1, 0).size(), 1u);
  EXPECT_EQ(manager_->StateSlot(1), 0);

  // The runner must reset the fresh slot before the step reads it.
  CacheManager::RunnerWork work = manager_->TakeRunnerWork();
  ASSERT_EQ(work.slots_to_reset.size(), 1u);
  EXPECT_EQ(work.slots_to_reset[0], 0);

  // Steady state: no new slot, no queued work.
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{16, 32}).ok());
  work = manager_->TakeRunnerWork();
  EXPECT_TRUE(work.slots_to_reset.empty());
  EXPECT_EQ(manager_->StateSlot(1), 0);

  ASSERT_TRUE(manager_->Release(1).ok());
  EXPECT_EQ(manager_->StateSlot(1), -1);
  EXPECT_EQ(manager_->NumFreeSlots(), 2);
  EXPECT_EQ(manager_->NumFreeBlocks(), 8);
}

TEST_F(CacheManagerHybridTest, SlotExhaustionLeavesNoPartialState) {
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 16}).ok());  // slot 0
  ASSERT_TRUE(manager_->Reserve(2, GrowthTarget{0, 16}).ok());  // slot 1
  EXPECT_EQ(manager_->NumFreeSlots(), 0);
  static_cast<void>(manager_->TakeRunnerWork());  // drain the two slot resets

  const Status failed = manager_->Reserve(3, GrowthTarget{0, 16});
  EXPECT_EQ(failed.code(), absl::StatusCode::kResourceExhausted);
  // The auto-registered request left nothing behind.
  EXPECT_EQ(manager_->StateSlot(3), -1);
  EXPECT_TRUE(manager_->GroupBlocks(3, 0).empty());
  EXPECT_EQ(manager_->NumFreeBlocks(), 6);
  EXPECT_TRUE(manager_->TakeRunnerWork().slots_to_reset.empty());
}

TEST_F(CacheManagerHybridTest, HybridNeverAdoptsOrPublishes) {
  const std::vector<TokenId> prompt = Prompt(32);

  // Seed the index through a non-hybrid manager over the same pool.
  auto pure = MakePlan({CacheSpec::FullAttention({PerToken(64)})});
  ASSERT_TRUE(pure.ok()) << pure.status();
  auto seeder = CacheManager::Create(Pool(), nullptr, *pure, true);
  ASSERT_TRUE(seeder.ok()) << seeder.status();
  ASSERT_TRUE((*seeder)->Reserve(1, GrowthTarget{0, 32}).ok());
  ASSERT_TRUE((*seeder)->PublishComputed(1, prompt, {}, 32).ok());
  ASSERT_TRUE((*seeder)->Release(1).ok());
  ASSERT_EQ(Pool()->NumCachedBlocks(), 2);

  // The hybrid facade counts the query as a miss: without matching recurrent
  // state the KV blocks could never be reused.
  auto adoption = manager_->AdoptPrefix(5, prompt, "", 32);
  ASSERT_TRUE(adoption.ok()) << adoption.status();
  EXPECT_EQ(adoption->cached_tokens, 0);
  EXPECT_TRUE(adoption->group_block_ids.empty());
  EXPECT_EQ(manager_->stats().prefix_hits, 0);
  EXPECT_EQ(manager_->stats().prefix_queries, 1);

  ASSERT_TRUE(manager_->Reserve(5, GrowthTarget{0, 16}).ok());
  ASSERT_TRUE(manager_->PublishComputed(5, prompt, {}, 16).ok());
  EXPECT_EQ(Pool()->NumCachedBlocks(), 2);  // nothing new from the hybrid
}

TEST_F(CacheManagerHybridTest, ForkCopiesStateToFreshSlots) {
  ASSERT_TRUE(manager_->Reserve(1, GrowthTarget{0, 32}).ok());
  EXPECT_EQ(manager_->StateSlot(1), 0);

  ASSERT_TRUE(manager_->Fork(1, {10}).ok());
  EXPECT_EQ(manager_->StateSlot(10), 1);
  EXPECT_EQ(manager_->StateSlot(1), 0);
  EXPECT_EQ(manager_->NumFreeSlots(), 0);

  // Children share the parent's blocks but carry their own state.
  EXPECT_EQ(manager_->GroupBlocks(10, 0), manager_->GroupBlocks(1, 0));

  CacheManager::RunnerWork work = manager_->TakeRunnerWork();
  ASSERT_EQ(work.slot_copies.size(), 1u);
  EXPECT_EQ(work.slot_copies[0].parent_slot, 0);
  EXPECT_EQ(work.slot_copies[0].child_slot, 1);
}

// ---------------------------------------------------------------------------
// The facade's hash chain primitive
// ---------------------------------------------------------------------------

/// The facade's chain discipline, mirror-imaged from ExtendHashes: the salt
/// rides in the first link only, parent digests chain the rest.
std::vector<BlockHash> ChainOver(const std::vector<std::vector<TokenId>>& blocks,
                                 std::string_view salt = {}) {
  const absl::Span<const uint8_t> salt_bytes(
      reinterpret_cast<const uint8_t*>(salt.data()), salt.size());
  std::vector<BlockHash> hashes;
  BlockHash parent{};
  for (size_t i = 0; i < blocks.size(); ++i) {
    const absl::Span<const uint8_t> extra = i == 0 ? salt_bytes : absl::Span<const uint8_t>{};
    parent = ChainHash(parent, absl::MakeConstSpan(blocks[i]), extra);
    hashes.push_back(parent);
  }
  return hashes;
}

TEST(ChainHashTest, DeterministicSaltedAndChainIsolated) {
  const std::vector<std::vector<TokenId>> blocks = {{1, 2, 3, 4}, {5, 6, 7, 8}};
  const auto first = ChainOver(blocks);
  const auto second = ChainOver(blocks);
  ASSERT_EQ(first.size(), 2u);
  EXPECT_EQ(first, second);  // cross-instance determinism

  // A salt namespaces every digest in the chain.
  const auto salted = ChainOver(blocks, "tenant-a");
  EXPECT_NE(first, salted);
  EXPECT_EQ(ChainOver(blocks, "tenant-a"), salted);

  // Any mutation changes the chain from that block onward, never before it.
  auto mutated = blocks;
  mutated[1][2] = 99;
  const auto mutated_hashes = ChainOver(mutated);
  EXPECT_EQ(mutated_hashes[0], first[0]);
  EXPECT_NE(mutated_hashes[1], first[1]);

  // Block boundaries are length-framed: re-splitting the same tokens
  // differently yields different digests.
  const auto resplit = ChainOver({{1, 2}, {3, 4}, {5, 6, 7, 8}});
  EXPECT_NE(resplit[0], first[0]);
}

}  // namespace
}  // namespace inferx
