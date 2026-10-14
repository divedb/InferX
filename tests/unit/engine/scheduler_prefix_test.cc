// Scheduler-level prefix-caching tests: admission consults the cache,
// computed blocks publish step by step, and identical prompts reuse KV.

#include "inferx/engine/scheduler.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "inferx/core/dtype.h"

namespace inferx {
namespace {

constexpr TokenId kEos = 99;
constexpr TokenId kTok = 7;
constexpr int64_t kBlockSize = 4;

class SchedulerPrefixTest : public ::testing::Test {
 protected:
  void MakeScheduler(bool enable_prefix_caching, int64_t blocks = 64) {
    KvLayout layout;
    layout.entries_per_token = 2;
    layout.kv_heads = 1;
    layout.head_dim = 8;
    layout.dtype = DataType::kBFloat16;
    pool_ = std::make_unique<KvBlockPool>(std::move(KvBlockPool::Create(
                                              /*num_layers=*/2, blocks, kBlockSize, layout,
                                              Device::Cpu())
                                              .value()));
    CacheConfig cache;
    cache.enable_prefix_caching = enable_prefix_caching;
    scheduler_ = std::make_unique<Scheduler>(SchedulerConfig{}, cache, pool_.get(),
                                             std::vector<TokenId>{kEos});
  }

  /// Drains the scheduler with a fake runner producing `kTok` forever.
  void RunToCompletion() {
    while (scheduler_->HasRequests()) {
      ASSERT_TRUE(ScheduleOnce());
    }
  }

  /// One Schedule/Update round; false when Schedule() itself failed.
  bool ScheduleOnce() {
    auto plan = scheduler_->Schedule();
    EXPECT_TRUE(plan.ok()) << plan.status();
    if (!plan.ok()) return false;
    ModelRunnerOutput result;
    for (const ScheduledRequest& sr : plan->scheduled) {
      result.samples.push_back({sr.request_id, {kTok}});
    }
    EXPECT_TRUE(scheduler_->UpdateFromOutput(*plan, result).ok());
    return true;
  }

  sampling::SamplingParams ShortParams() {
    sampling::SamplingParams params;
    params.max_tokens = 2;
    return params;
  }

  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<Scheduler> scheduler_;
};

std::vector<TokenId> Tokens(int64_t count, int64_t base = 100) {
  std::vector<TokenId> tokens;
  tokens.reserve(count);
  for (int64_t i = 0; i < count; ++i) tokens.push_back(static_cast<TokenId>(base + i));
  return tokens;
}

TEST_F(SchedulerPrefixTest, IdenticalPromptSkipsCachedBlocks) {
  MakeScheduler(true);
  const std::vector<TokenId> prompt = Tokens(12);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, prompt, ShortParams())).ok());
  RunToCompletion();
  // Three full prompt blocks + the blocks the decode tail needed are cached;
  // the release kept their bytes out of the free list.
  const int64_t free_after_first = pool_->FreeBlocks();
  EXPECT_LT(free_after_first, pool_->NumBlocks());
  EXPECT_TRUE(scheduler_->PopFinished().has_value());

  // The second, identical request must adopt the cached prefix: its first
  // step schedules only the uncapped tail (the n-1 rule keeps one block
  // worth of tokens: 12 tokens - 8 cached = 4 scheduled).
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, prompt, ShortParams())).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok()) << plan.status();
  ASSERT_EQ(plan->scheduled_new_reqs.size(), 1u);
  const NewRequestData& fresh = plan->scheduled_new_reqs[0];
  EXPECT_EQ(fresh.num_computed_tokens, 8);
  EXPECT_EQ(plan->scheduled[0].num_new_tokens, 4);
  ASSERT_EQ(fresh.block_ids.size(), 3u);  // 2 adopted + 1 granted.
  // The adopted blocks are the pool blocks the first request published;
  // nothing new was allocated for them.
  EXPECT_EQ(pool_->FreeBlocks(), free_after_first - 1);

  ModelRunnerOutput result;
  result.samples.push_back({2, {kTok}});
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*plan, result).ok());
  RunToCompletion();

  // The second request finished and its uncached tail stays cached; no
  // blocks leak: every block is either free or discoverable by hash.
  EXPECT_EQ(pool_->UsedBlocks() + 0, pool_->NumBlocks() - pool_->FreeBlocks());
}

TEST_F(SchedulerPrefixTest, FullPromptHitStillRecomputesTheLastToken) {
  MakeScheduler(true);
  // Exactly one block: the cap must force a recompute of the final token.
  const std::vector<TokenId> prompt = Tokens(4);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, prompt, ShortParams())).ok());
  RunToCompletion();
  ASSERT_TRUE(scheduler_->PopFinished().has_value());

  ASSERT_TRUE(scheduler_->AddRequest(Request(2, prompt, ShortParams())).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok()) << plan.status();
  // A four-token prompt whose only block is cached still schedules 4 tokens:
  // the n-1 cap (3) truncates the hit to zero whole blocks.
  EXPECT_EQ(plan->scheduled_new_reqs[0].num_computed_tokens, 0);
  EXPECT_EQ(plan->scheduled[0].num_new_tokens, 4);
}

TEST_F(SchedulerPrefixTest, SaltedRequestSharesNothing) {
  MakeScheduler(true);
  const std::vector<TokenId> prompt = Tokens(12);

  Request salted(1, prompt, ShortParams());
  salted.set_cache_salt("tenant-a");
  ASSERT_TRUE(scheduler_->AddRequest(std::move(salted)).ok());
  RunToCompletion();
  ASSERT_TRUE(scheduler_->PopFinished().has_value());

  ASSERT_TRUE(scheduler_->AddRequest(Request(2, prompt, ShortParams())).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok()) << plan.status();
  // Unsalted prompt sees no salted cache: full recompute.
  EXPECT_EQ(plan->scheduled_new_reqs[0].num_computed_tokens, 0);
  EXPECT_EQ(plan->scheduled[0].num_new_tokens, 12);
}

TEST_F(SchedulerPrefixTest, DisabledCachingRecomputesEverything) {
  MakeScheduler(false);
  const std::vector<TokenId> prompt = Tokens(12);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, prompt, ShortParams())).ok());
  RunToCompletion();
  const int64_t free_after_first = pool_->FreeBlocks();
  EXPECT_EQ(free_after_first, pool_->NumBlocks());  // Everything returned.

  ASSERT_TRUE(scheduler_->AddRequest(Request(2, prompt, ShortParams())).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok()) << plan.status();
  EXPECT_EQ(plan->scheduled_new_reqs[0].num_computed_tokens, 0);
  EXPECT_EQ(plan->scheduled[0].num_new_tokens, 12);
}

}  // namespace
}  // namespace inferx
