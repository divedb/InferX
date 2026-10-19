#include "inferx/cache/cache_group_manager.h"

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

BlockHash Hash(int seed) {
  BlockHash hash;
  hash.bytes[0] = static_cast<uint8_t>(seed);
  hash.bytes[1] = static_cast<uint8_t>(seed >> 8);
  hash.bytes[2] = static_cast<uint8_t>(seed >> 16);
  return hash;
}

std::vector<BlockHash> HashRange(int seed, int64_t count) {
  std::vector<BlockHash> hashes;
  hashes.reserve(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) hashes.push_back(Hash(seed + static_cast<int>(i)));
  return hashes;
}

CacheGroupSpec MakeGroup(int64_t index, CacheSpec spec) {
  CacheGroupSpec group;
  group.group_index = index;
  group.layers = {0};
  group.spec = std::move(spec);
  return group;
}

const TokenWiseRequestState* Tokens(const RequestGroupState& req) {
  return std::get_if<TokenWiseRequestState>(&req);
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

TEST(CreateCacheGroupManagerTest, DispatchesOnRetention) {
  auto pool = KvBlockPool::Create(1, 4, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto state_pool =
      RecurrentStatePool::Create(MakeGroup(0, CacheSpec::Recurrent(
                                                  {CacheComponent::FixedSlot(512,
                                                                             DataType::kFloat32)},
                                                  kBlockSize)),
                                 2, Device::Cpu());
  ASSERT_TRUE(state_pool.ok()) << state_pool.status();

  const std::vector<std::pair<CacheSpec, bool>> cases = {
      {CacheSpec::FullAttention({PerToken(64)}), true},
      {CacheSpec::SlidingWindow({PerToken(64)}, 32), true},
      {CacheSpec::ChunkedLocal({PerToken(64)}, 64), true},
      {CacheSpec::Encoder({PerToken(64)}), true},
      {CacheSpec::Recurrent({CacheComponent::FixedSlot(512, DataType::kFloat32)}, kBlockSize),
       false},
  };
  for (size_t i = 0; i < cases.size(); ++i) {
    auto manager = CreateCacheGroupManager(MakeGroup(static_cast<int64_t>(i), cases[i].first),
                                           &*pool, &*state_pool);
    ASSERT_TRUE(manager.ok()) << manager.status();
    EXPECT_EQ((*manager)->group_index(), static_cast<int64_t>(i));
    EXPECT_EQ((*manager)->IsTokenWise(), cases[i].second);
    EXPECT_EQ((*manager)->group().spec.retention, cases[i].first.retention);
  }
}

TEST(CreateCacheGroupManagerTest, RejectsInvalidArguments) {
  auto pool = KvBlockPool::Create(1, 4, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();

  CacheGroupSpec empty = MakeGroup(0, CacheSpec::FullAttention({PerToken(64)}));
  empty.layers.clear();
  EXPECT_EQ(CreateCacheGroupManager(empty, &*pool, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);

  const CacheGroupSpec recurrent = MakeGroup(
      1, CacheSpec::Recurrent({CacheComponent::FixedSlot(512, DataType::kFloat32)}, kBlockSize));
  EXPECT_EQ(CreateCacheGroupManager(recurrent, nullptr, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);

  const CacheGroupSpec token_wise = MakeGroup(2, CacheSpec::FullAttention({PerToken(64)}));
  EXPECT_EQ(CreateCacheGroupManager(token_wise, nullptr, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// ---------------------------------------------------------------------------
// Full attention
// ---------------------------------------------------------------------------

class FullAttentionManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto created = KvBlockPool::Create(1, 8, kBlockSize, PerToken(64), Device::Cpu());
    ASSERT_TRUE(created.ok()) << created.status();
    pool_ = std::make_unique<KvBlockPool>(std::move(*created));
    auto manager = CreateCacheGroupManager(
        MakeGroup(0, CacheSpec::FullAttention({PerToken(64)})), pool_.get(), nullptr);
    ASSERT_TRUE(manager.ok()) << manager.status();
    manager_ = std::move(*manager);
  }

  std::unique_ptr<KvBlockPool> pool_;
  CacheGroupManagerPtr manager_;
};

TEST_F(FullAttentionManagerTest, ReserveGrowsTableAndReportsGranted) {
  RequestGroupState req;
  EXPECT_EQ(manager_->NumNewBlocksNeeded(req, GrowthTarget{0, 32}), 2);
  EXPECT_EQ(manager_->NumBlocksHeldAt(33), 3);
  EXPECT_EQ(manager_->NumBlocksHeldAt(32), 2);

  auto granted = manager_->Reserve(req, GrowthTarget{0, 32});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_EQ(granted->blocks.size(), 2u);
  EXPECT_TRUE(granted->copies.empty());
  ASSERT_NE(Tokens(req), nullptr);
  EXPECT_EQ(Tokens(req)->blocks, granted->blocks);
  EXPECT_EQ(pool_->FreeBlocks(), 6);

  // Growing further only asks for the delta; holding steady asks for nothing.
  EXPECT_EQ(manager_->NumNewBlocksNeeded(req, GrowthTarget{32, 32}), 0);
  auto next = manager_->Reserve(req, GrowthTarget{32, 48});
  ASSERT_TRUE(next.ok()) << next.status();
  EXPECT_EQ(next->blocks.size(), 1u);
  EXPECT_EQ(Tokens(req)->blocks.size(), 3u);
}

TEST_F(FullAttentionManagerTest, ReserveFailureAndRollbackLeaveRequestUnchanged) {
  auto small_pool = KvBlockPool::Create(1, 2, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(small_pool.ok()) << small_pool.status();
  auto manager = CreateCacheGroupManager(MakeGroup(0, CacheSpec::FullAttention({PerToken(64)})),
                                         &*small_pool, nullptr);
  ASSERT_TRUE(manager.ok()) << manager.status();

  RequestGroupState req;
  auto failed = (*manager)->Reserve(req, GrowthTarget{0, 48});
  EXPECT_EQ(failed.status().code(), absl::StatusCode::kResourceExhausted);
  ASSERT_NE(Tokens(req), nullptr);
  EXPECT_TRUE(Tokens(req)->blocks.empty());
  EXPECT_EQ(small_pool->FreeBlocks(), 2);

  auto granted = (*manager)->Reserve(req, GrowthTarget{0, 16});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_EQ(granted->blocks.size(), 1u);
  ASSERT_TRUE((*manager)->Rollback(req, *granted).ok());
  EXPECT_TRUE(Tokens(req)->blocks.empty());
  EXPECT_EQ(small_pool->FreeBlocks(), 2);
  EXPECT_EQ(small_pool->RefCount(granted->blocks.front()), 0u);
}

TEST_F(FullAttentionManagerTest, PublishReleaseAdoptRoundTrip) {
  RequestGroupState a;
  const std::vector<BlockHash> hashes = HashRange(100, 3);
  auto a_blocks = manager_->Reserve(a, GrowthTarget{0, 32});
  ASSERT_TRUE(a_blocks.ok()) << a_blocks.status();
  ASSERT_TRUE(manager_->PublishCompleted(a, hashes, 32).ok());
  EXPECT_EQ(pool_->NumCachedBlocks(), 2);
  EXPECT_EQ(pool_->EvictableBlocks(), 0);

  // Releasing keeps the published blocks discoverable and evictable.
  ASSERT_TRUE(manager_->Release(a).ok());
  EXPECT_EQ(pool_->FreeBlocks(), 6);
  EXPECT_EQ(pool_->EvictableBlocks(), 2);

  // A second request with the same digests finds and adopts the prefix.
  RequestGroupState b;
  const CacheHit hit = manager_->FindLongestCacheHit(hashes, 64);
  ASSERT_EQ(hit.blocks.size(), 2u);
  EXPECT_EQ(hit.num_tokens, 32);
  // Pinning idle cached blocks consumes capacity the scheduler must see.
  EXPECT_EQ(manager_->NumNewBlocksNeeded(b, GrowthTarget{32, 48}, &hit), 3);

  ASSERT_TRUE(manager_->AdoptCached(b, hit).ok());
  ASSERT_NE(Tokens(b), nullptr);
  EXPECT_EQ(Tokens(b)->blocks, hit.blocks);
  EXPECT_EQ(Tokens(b)->num_cached_tokens, 32);
  EXPECT_EQ(pool_->EvictableBlocks(), 0);
  EXPECT_EQ(pool_->RefCount(hit.blocks.front()), 1u);

  // The publish cursor starts past adopted blocks: no double registration.
  ASSERT_TRUE(manager_->PublishCompleted(b, hashes, 32).ok());
  EXPECT_EQ(pool_->NumCachedBlocks(), 2);

  auto grown = manager_->Reserve(b, GrowthTarget{32, 48});
  ASSERT_TRUE(grown.ok()) << grown.status();
  EXPECT_EQ(grown->blocks.size(), 1u);
  ASSERT_TRUE(manager_->Release(b).ok());
  // The adopted prefix stays cached; only the fresh tail block frees.
  EXPECT_EQ(pool_->NumCachedBlocks(), 2);
  EXPECT_EQ(pool_->FreeBlocks(), 6);
}

TEST_F(FullAttentionManagerTest, FullAttentionHitStopsAtFirstMiss) {
  RequestGroupState a;
  const std::vector<BlockHash> hashes = HashRange(200, 3);
  auto a_blocks = manager_->Reserve(a, GrowthTarget{0, 32});
  ASSERT_TRUE(a_blocks.ok()) << a_blocks.status();
  ASSERT_TRUE(manager_->PublishCompleted(a, hashes, 32).ok());
  ASSERT_TRUE(manager_->Release(a).ok());

  // Same first digest, divergent second: only the first block hits.
  std::vector<BlockHash> diverged = hashes;
  diverged[1] = Hash(999);
  RequestGroupState b;
  const CacheHit hit = manager_->FindLongestCacheHit(diverged, 64);
  ASSERT_EQ(hit.blocks.size(), 1u);
  EXPECT_EQ(hit.num_tokens, kBlockSize);

  // The token cap bounds the walk in whole blocks.
  const CacheHit capped = manager_->FindLongestCacheHit(hashes, 20);
  EXPECT_EQ(capped.blocks.size(), 1u);
  EXPECT_EQ(capped.num_tokens, kBlockSize);
}

TEST_F(FullAttentionManagerTest, AdoptRejectsBadRequestsAndHits) {
  RequestGroupState a;
  const std::vector<BlockHash> hashes = HashRange(300, 2);
  auto a_blocks = manager_->Reserve(a, GrowthTarget{0, 32});
  ASSERT_TRUE(a_blocks.ok()) << a_blocks.status();
  ASSERT_TRUE(manager_->PublishCompleted(a, hashes, 32).ok());
  ASSERT_TRUE(manager_->Release(a).ok());
  const CacheHit hit = manager_->FindLongestCacheHit(hashes, 64);
  ASSERT_EQ(hit.blocks.size(), 2u);

  // A hit can only be adopted into an empty request.
  RequestGroupState b;
  ASSERT_TRUE(manager_->Reserve(b, GrowthTarget{0, 16}).ok());
  EXPECT_EQ(manager_->AdoptCached(b, hit).code(),
            absl::StatusCode::kFailedPrecondition);

  CacheHit malformed;
  malformed.blocks = hit.blocks;
  malformed.num_tokens = hit.num_tokens + kBlockSize;  // tokens/blocks mismatch
  RequestGroupState c;
  EXPECT_EQ(manager_->AdoptCached(c, malformed).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(FullAttentionManagerTest, ForkSharesBlocksAndCopyOnWriteDuplicatesThePartialBlock) {
  RequestGroupState parent;
  auto parent_blocks = manager_->Reserve(parent, GrowthTarget{0, 20});
  ASSERT_TRUE(parent_blocks.ok()) << parent_blocks.status();
  ASSERT_EQ(Tokens(parent)->blocks.size(), 2u);  // second block partial at 20

  std::vector<RequestGroupState> children;
  std::vector<SlotCopy> slot_copies;
  ASSERT_TRUE(manager_->Fork(parent, 2, &children, &slot_copies).ok());
  ASSERT_EQ(children.size(), 2u);
  EXPECT_TRUE(slot_copies.empty());
  for (const RequestGroupState& child : children) {
    EXPECT_EQ(Tokens(child)->blocks, Tokens(parent)->blocks);
  }
  EXPECT_EQ(pool_->RefCount(Tokens(parent)->blocks[0]), 3u);
  EXPECT_EQ(pool_->RefCount(Tokens(parent)->blocks[1]), 3u);

  // The first child that writes into the shared partial block duplicates it.
  auto grown = manager_->Reserve(children[0], GrowthTarget{20, 24});
  ASSERT_TRUE(grown.ok()) << grown.status();
  EXPECT_TRUE(grown->blocks.empty());  // no new tail: the copy IS the growth
  ASSERT_EQ(grown->copies.size(), 1u);
  const BlockCopy& copy = grown->copies.front();
  EXPECT_EQ(copy.src, Tokens(parent)->blocks[1]);
  EXPECT_NE(copy.dst, KvBlockPool::kNullBlockId);
  EXPECT_EQ(Tokens(children[0])->blocks,
            (std::vector<BlockId>({Tokens(children[0])->blocks[0], copy.dst})));
  EXPECT_EQ(pool_->RefCount(copy.src), 2u);  // parent + the other child
  EXPECT_EQ(pool_->RefCount(copy.dst), 1u);

  // Rolling the copy back restores the shared block and frees the duplicate.
  ASSERT_TRUE(manager_->Rollback(children[0], *grown).ok());
  EXPECT_EQ(Tokens(children[0])->blocks, Tokens(parent)->blocks);
  EXPECT_EQ(pool_->RefCount(copy.src), 3u);
  EXPECT_EQ(pool_->RefCount(copy.dst), 0u);

  EXPECT_EQ(manager_->Fork(parent, 0, &children, &slot_copies).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(FullAttentionManagerTest, WrongKindStateIsRejected) {
  RequestGroupState recurrent(RecurrentRequestState{});
  EXPECT_EQ(manager_->Reserve(recurrent, GrowthTarget{0, 16}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->AdoptCached(recurrent, CacheHit{}).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->ReleaseSkipped(recurrent, 32).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->Release(recurrent).code(), absl::StatusCode::kInvalidArgument);
}

// ---------------------------------------------------------------------------
// Sliding window
// ---------------------------------------------------------------------------

class SlidingWindowManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto created = KvBlockPool::Create(1, 16, kBlockSize, PerToken(64), Device::Cpu());
    ASSERT_TRUE(created.ok()) << created.status();
    pool_ = std::make_unique<KvBlockPool>(std::move(*created));
    auto manager = CreateCacheGroupManager(
        MakeGroup(0, CacheSpec::SlidingWindow({PerToken(64)}, 32)), pool_.get(), nullptr);
    ASSERT_TRUE(manager.ok()) << manager.status();
    manager_ = std::move(*manager);
  }

  std::unique_ptr<KvBlockPool> pool_;
  CacheGroupManagerPtr manager_;
};

TEST_F(SlidingWindowManagerTest, SkipAndHeldMathFollowsTheWindow) {
  // The next token at position 100 needs the previous 31 tokens and itself.
  EXPECT_EQ(manager_->NumSkippedTokens(100), 69);
  EXPECT_EQ(manager_->NumSkippedTokens(0), 0);
  // 8 blocks span 128 tokens; the window only needs the last two.
  EXPECT_EQ(manager_->NumBlocksHeldAt(128), 2);
  EXPECT_EQ(manager_->NumBlocksHeldAt(32), 2);
}

TEST_F(SlidingWindowManagerTest, ReleaseSkippedRollsOffBelowTheFloor) {
  RequestGroupState req;
  auto granted = manager_->Reserve(req, GrowthTarget{0, 128});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_EQ(Tokens(req)->blocks.size(), 8u);
  const std::vector<BlockId> original = Tokens(req)->blocks;

  ASSERT_TRUE(manager_->ReleaseSkipped(req, 129).ok());
  const std::vector<BlockId>& table = Tokens(req)->blocks;
  ASSERT_EQ(table.size(), 8u);  // indexing by logical block is preserved
  for (int64_t i = 0; i < 6; ++i) {
    EXPECT_EQ(table[static_cast<size_t>(i)], KvBlockPool::kNullBlockId) << "entry " << i;
  }
  EXPECT_EQ(table[6], original[6]);
  EXPECT_EQ(table[7], original[7]);
  EXPECT_EQ(pool_->FreeBlocks(), 6 + 8);  // 16 usable, two still held

  // Sizing sees the post-roll state: growing by one token needs one block.
  EXPECT_EQ(manager_->NumNewBlocksNeeded(req, GrowthTarget{129, 130}), 1);
  auto grown = manager_->Reserve(req, GrowthTarget{129, 130});
  ASSERT_TRUE(grown.ok()) << grown.status();
  EXPECT_EQ(grown->blocks.size(), 1u);
}

TEST_F(SlidingWindowManagerTest, ReservePadsSkippedPrefixWithoutStorage) {
  // Window of 32: prefill at position 64 only needs blocks from block 2 on,
  // so entries 0..1 are null pads that cost no storage.
  RequestGroupState req;
  EXPECT_EQ(manager_->NumNewBlocksNeeded(req, GrowthTarget{64, 80}), 3);
  auto granted = manager_->Reserve(req, GrowthTarget{64, 80});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_EQ(granted->blocks.size(), 3u);
  const std::vector<BlockId>& table = Tokens(req)->blocks;
  ASSERT_EQ(table.size(), 5u);
  for (int64_t i = 0; i < 2; ++i) {
    EXPECT_EQ(table[static_cast<size_t>(i)], KvBlockPool::kNullBlockId) << "entry " << i;
  }
  EXPECT_EQ(table[2], granted->blocks[0]);
  EXPECT_EQ(table[3], granted->blocks[1]);
  EXPECT_EQ(table[4], granted->blocks[2]);
}

TEST_F(SlidingWindowManagerTest, HitSearchLooksPastMisses) {
  RequestGroupState a;
  const std::vector<BlockHash> hashes = HashRange(400, 2);
  auto a_blocks = manager_->Reserve(a, GrowthTarget{0, 32});
  ASSERT_TRUE(a_blocks.ok()) << a_blocks.status();
  ASSERT_TRUE(manager_->PublishCompleted(a, hashes, 32).ok());
  ASSERT_TRUE(manager_->Release(a).ok());

  // Cached, missing, cached: a window hit must cover the window, not the
  // whole prefix, but a gap inside the covered range still shortens it.
  const std::vector<BlockHash> gap = {hashes[0], Hash(999), hashes[1]};
  const CacheHit hit = manager_->FindLongestCacheHit(gap, 64);
  ASSERT_EQ(hit.blocks.size(), 1u);
  EXPECT_EQ(hit.num_tokens, kBlockSize);
}

// ---------------------------------------------------------------------------
// Chunked local
// ---------------------------------------------------------------------------

TEST(ChunkedLocalManagerTest, SkipsAreChunkAligned) {
  auto pool = KvBlockPool::Create(1, 16, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto manager =
      CreateCacheGroupManager(MakeGroup(0, CacheSpec::ChunkedLocal({PerToken(64)}, 64)),
                              &*pool, nullptr);
  ASSERT_TRUE(manager.ok()) << manager.status();

  EXPECT_EQ((*manager)->NumSkippedTokens(63), 0);
  EXPECT_EQ((*manager)->NumSkippedTokens(64), 64);
  EXPECT_EQ((*manager)->NumSkippedTokens(128), 128);  // new chunk starts

  RequestGroupState req;
  auto granted = (*manager)->Reserve(req, GrowthTarget{0, 128});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_EQ(Tokens(req)->blocks.size(), 8u);
  ASSERT_TRUE((*manager)->ReleaseSkipped(req, 64).ok());
  // The first chunk's blocks roll off; the second chunk stays.
  for (int64_t i = 0; i < 4; ++i) {
    EXPECT_EQ(Tokens(req)->blocks[static_cast<size_t>(i)], KvBlockPool::kNullBlockId);
  }
  for (int64_t i = 4; i < 8; ++i) {
    EXPECT_NE(Tokens(req)->blocks[static_cast<size_t>(i)], KvBlockPool::kNullBlockId);
  }
  EXPECT_EQ(pool->FreeBlocks(), 12);
}

// ---------------------------------------------------------------------------
// Cross attention (encoder retention)
// ---------------------------------------------------------------------------

TEST(CrossAttentionManagerTest, SizesFromEncoderLengthAndNeverPublishes) {
  auto pool = KvBlockPool::Create(1, 8, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto manager =
      CreateCacheGroupManager(MakeGroup(0, CacheSpec::Encoder({PerToken(64)})), &*pool,
                              nullptr);
  ASSERT_TRUE(manager.ok()) << manager.status();

  EXPECT_EQ((*manager)->NumBlocksHeldAt(1, 48), 3);  // encoder tokens, not decode steps

  RequestGroupState req;
  auto granted = (*manager)->Reserve(req, GrowthTarget{0, 1, 48});
  ASSERT_TRUE(granted.ok()) << granted.status();
  EXPECT_EQ(granted->blocks.size(), 3u);

  const std::vector<BlockHash> hashes = HashRange(500, 3);
  ASSERT_TRUE((*manager)->PublishCompleted(req, hashes, 48).ok());
  EXPECT_EQ(pool->NumCachedBlocks(), 0);  // encoder caches are never prefix-cached
  EXPECT_TRUE((*manager)->FindLongestCacheHit(hashes, 64).blocks.empty());
}

TEST(CrossAttentionManagerTest, ForkedBlocksAreImmutableSoNoCopyOnWrite) {
  auto pool = KvBlockPool::Create(1, 8, kBlockSize, PerToken(64), Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();
  auto manager =
      CreateCacheGroupManager(MakeGroup(0, CacheSpec::Encoder({PerToken(64)})), &*pool,
                              nullptr);
  ASSERT_TRUE(manager.ok()) << manager.status();

  RequestGroupState parent;
  auto parent_blocks = (*manager)->Reserve(parent, GrowthTarget{0, 20, 20});
  ASSERT_TRUE(parent_blocks.ok()) << parent_blocks.status();

  std::vector<RequestGroupState> children;
  std::vector<SlotCopy> slot_copies;
  ASSERT_TRUE((*manager)->Fork(parent, 2, &children, &slot_copies).ok());

  // Writing into the shared partial block would normally copy-on-write; the
  // encoder cache is written once, so forks share it outright.
  auto grown = (*manager)->Reserve(children[0], GrowthTarget{20, 36, 36});
  ASSERT_TRUE(grown.ok()) << grown.status();
  EXPECT_TRUE(grown->copies.empty());
  EXPECT_EQ(grown->blocks.size(), 1u);
  EXPECT_EQ(Tokens(children[0])->blocks[0], Tokens(parent)->blocks[0]);
  EXPECT_EQ(Tokens(children[0])->blocks[1], Tokens(parent)->blocks[1]);
}

// ---------------------------------------------------------------------------
// Recurrent
// ---------------------------------------------------------------------------

class RecurrentStateManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    group_ = MakeRecurrentGroup();
    auto created = RecurrentStatePool::Create(group_, 3, Device::Cpu());
    ASSERT_TRUE(created.ok()) << created.status();
    state_pool_ = std::make_unique<RecurrentStatePool>(std::move(*created));
    auto manager = CreateCacheGroupManager(group_, nullptr, state_pool_.get());
    ASSERT_TRUE(manager.ok()) << manager.status();
    manager_ = std::move(*manager);
  }

  static CacheGroupSpec MakeRecurrentGroup() {
    CacheGroupSpec group;
    group.layers = {3, 7};  // non-contiguous model layers
    group.spec = CacheSpec::Recurrent(
        {CacheComponent::FixedSlot(512, DataType::kFloat32),
         CacheComponent::FixedSlot(128, DataType::kFloat32)},
        kBlockSize);
    return group;
  }

  CacheGroupSpec group_;
  std::unique_ptr<RecurrentStatePool> state_pool_;
  CacheGroupManagerPtr manager_;
};

TEST_F(RecurrentStateManagerTest, ReserveAssignsOneSlotIdempotently) {
  RequestGroupState req(RecurrentRequestState{});
  EXPECT_EQ(manager_->NumNewBlocksNeeded(req, GrowthTarget{0, 64}), 1);
  EXPECT_EQ(manager_->NumBlocksHeldAt(4096), 1);
  EXPECT_TRUE(manager_->FindLongestCacheHit(HashRange(600, 4), 64).blocks.empty());

  auto granted = manager_->Reserve(req, GrowthTarget{0, 64});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_TRUE(granted->slot.has_value());
  EXPECT_EQ(*granted->slot, 0);
  EXPECT_EQ(std::get<RecurrentRequestState>(req).slot, 0);
  EXPECT_EQ(state_pool_->NumUsedSlots(), 1);

  // Already holds its slot: nothing new needed, nothing new granted.
  EXPECT_EQ(manager_->NumNewBlocksNeeded(req, GrowthTarget{64, 128}), 0);
  auto again = manager_->Reserve(req, GrowthTarget{64, 128});
  ASSERT_TRUE(again.ok()) << again.status();
  EXPECT_FALSE(again->slot.has_value());
  EXPECT_EQ(state_pool_->NumUsedSlots(), 1);
}

TEST_F(RecurrentStateManagerTest, RollbackAndReleaseReturnTheSlot) {
  RequestGroupState req(RecurrentRequestState{});
  auto granted = manager_->Reserve(req, GrowthTarget{0, 64});
  ASSERT_TRUE(granted.ok()) << granted.status();

  Granted mismatched;
  mismatched.slot = 5;
  EXPECT_EQ(manager_->Rollback(req, mismatched).code(),
            absl::StatusCode::kInvalidArgument);

  ASSERT_TRUE(manager_->Rollback(req, *granted).ok());
  EXPECT_EQ(std::get<RecurrentRequestState>(req).slot, -1);
  EXPECT_EQ(state_pool_->NumFreeSlots(), 3);

  granted = manager_->Reserve(req, GrowthTarget{0, 64});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_TRUE(manager_->Release(req).ok());
  EXPECT_EQ(std::get<RecurrentRequestState>(req).slot, -1);
  EXPECT_EQ(state_pool_->NumFreeSlots(), 3);
  // Releasing a request that holds nothing is a harmless no-op.
  EXPECT_TRUE(manager_->Release(req).ok());
}

TEST_F(RecurrentStateManagerTest, ForkGivesFreshSlotsAndReportsCopies) {
  RequestGroupState parent(RecurrentRequestState{});
  auto granted = manager_->Reserve(parent, GrowthTarget{0, 64});
  ASSERT_TRUE(granted.ok()) << granted.status();
  ASSERT_EQ(*granted->slot, 0);

  std::vector<RequestGroupState> children;
  std::vector<SlotCopy> slot_copies;
  ASSERT_TRUE(manager_->Fork(parent, 2, &children, &slot_copies).ok());
  ASSERT_EQ(children.size(), 2u);
  ASSERT_EQ(slot_copies.size(), 2u);
  for (size_t i = 0; i < children.size(); ++i) {
    const int64_t slot = std::get<RecurrentRequestState>(children[i]).slot;
    EXPECT_NE(slot, 0);
    EXPECT_EQ(slot_copies[i].parent_slot, 0);
    EXPECT_EQ(slot_copies[i].child_slot, slot);
  }
  EXPECT_EQ(state_pool_->NumUsedSlots(), 3);
  EXPECT_EQ(std::get<RecurrentRequestState>(parent).slot, 0);

  EXPECT_EQ(manager_->Fork(parent, 0, &children, &slot_copies).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->Fork(RecurrentRequestState{}, 1, &children, &slot_copies).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(RecurrentStateManagerTest, ForkExhaustionRollsBackTakenSlots) {
  auto small = RecurrentStatePool::Create(MakeRecurrentGroup(), 2, Device::Cpu());
  ASSERT_TRUE(small.ok()) << small.status();
  auto manager = CreateCacheGroupManager(MakeRecurrentGroup(), nullptr, &*small);
  ASSERT_TRUE(manager.ok()) << manager.status();

  RequestGroupState parent(RecurrentRequestState{});
  auto granted = (*manager)->Reserve(parent, GrowthTarget{0, 64});
  ASSERT_TRUE(granted.ok()) << granted.status();

  std::vector<RequestGroupState> children;
  std::vector<SlotCopy> slot_copies;
  // One free slot, two children: the second allocation fails and the first
  // taken slot is returned.
  auto forked = (*manager)->Fork(parent, 2, &children, &slot_copies);
  EXPECT_EQ(forked.code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(small->NumFreeSlots(), 1);
}

TEST_F(RecurrentStateManagerTest, WrongKindStateAndBadHitsRejected) {
  RequestGroupState token_wise(TokenWiseRequestState{});
  EXPECT_EQ(manager_->Reserve(token_wise, GrowthTarget{0, 16}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(manager_->Release(token_wise).code(), absl::StatusCode::kInvalidArgument);

  RequestGroupState req(RecurrentRequestState{});
  CacheHit hit;
  hit.blocks = {1};
  hit.num_tokens = kBlockSize;
  EXPECT_EQ(manager_->AdoptCached(req, hit).code(),
            absl::StatusCode::kInvalidArgument);
}

// ---------------------------------------------------------------------------
// RecurrentStatePool storage layout
// ---------------------------------------------------------------------------

TEST(RecurrentStatePoolTest, CreateValidatesTheGroup) {
  const CacheGroupSpec recurrent = [] {
    CacheGroupSpec group;
    group.layers = {0};
    group.spec = CacheSpec::Recurrent({CacheComponent::FixedSlot(256, DataType::kFloat32)},
                                      kBlockSize);
    return group;
  }();
  const CacheGroupSpec token_wise = MakeGroup(0, CacheSpec::FullAttention({PerToken(64)}));

  EXPECT_EQ(RecurrentStatePool::Create(recurrent, 0, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(RecurrentStatePool::Create(token_wise, 2, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
  CacheGroupSpec per_token;
  per_token.layers = {0};
  per_token.spec = CacheSpec::Recurrent({PerToken(64)}, kBlockSize);
  EXPECT_EQ(RecurrentStatePool::Create(per_token, 2, Device::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(RecurrentStatePoolTest, ComponentViewsAreDisjointPerLayerAndComponent) {
  CacheGroupSpec group;
  group.layers = {3, 7};
  group.spec = CacheSpec::Recurrent(
      {CacheComponent::FixedSlot(512, DataType::kFloat32),
       CacheComponent::FixedSlot(128, DataType::kFloat32)},
      kBlockSize);
  auto pool = RecurrentStatePool::Create(group, 4, Device::Cpu());
  ASSERT_TRUE(pool.ok()) << pool.status();

  // 2 layers x (512 + 128) bytes per slot, 4 slots.
  EXPECT_EQ(pool->SlotBytes(), 2 * (512 + 128));
  EXPECT_EQ(pool->Bytes(), 4 * 2 * (512 + 128));
  EXPECT_EQ(pool->PositionOfLayer(3), 0);
  EXPECT_EQ(pool->PositionOfLayer(7), 1);
  EXPECT_EQ(pool->PositionOfLayer(4), -1);

  auto a = pool->ComponentView(3, 0);
  auto b = pool->ComponentView(3, 1);
  auto c = pool->ComponentView(7, 0);
  ASSERT_TRUE(a.ok() && b.ok() && c.ok());
  EXPECT_EQ(a->Dims(), (std::vector<int64_t>({4, 512})));
  EXPECT_EQ(b->Dims(), (std::vector<int64_t>({4, 128})));
  const auto offset = [](const Tensor& from, const Tensor& to) {
    return static_cast<int64_t>(reinterpret_cast<const uint8_t*>(to.Bytes()) -
                                reinterpret_cast<const uint8_t*>(from.Bytes()));
  };
  // [position][component] row-major over [slots, bytes_per_slot] regions.
  EXPECT_EQ(offset(*a, *b), 4 * 512);
  EXPECT_EQ(offset(*a, *c), 4 * 512 + 4 * 128);
  EXPECT_EQ(pool->ComponentView(4, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(pool->ComponentView(3, 2).status().code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx
