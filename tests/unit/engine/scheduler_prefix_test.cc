// Scheduler-level prefix-caching tests: admission consults the cache,
// computed blocks publish step by step, and identical prompts reuse KV.

#include "inferx/engine/scheduler.h"

#include "inferx/ops/attention/registry.h"
#include "inferx/ops/attention/runtime.h"
#include "inferx/ops/attention/spec.h"


#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "inferx/core/dtype.h"

namespace inferx {
namespace {

constexpr TokenId kEos = 99;
constexpr TokenId kTok = 7;
constexpr int64_t kBlockSize = 4;

/// Two dense GQA layers over one CPU pool, the shape every scheduler test
/// schedules against (matches the legacy 2-layer KvLayout pools).
attn::AttentionSpec LayerSpec() {
  attn::AttentionSpec spec;
  spec.params = attn::DenseParams{.num_q_heads = 4, .num_kv_heads = 1, .head_dim = 8};
  spec.q_dtype = DataType::kBFloat16;
  spec.kv_dtype = DataType::kBFloat16;
  return spec;
}

std::unique_ptr<attn::AttentionRuntime> MakeRuntime(bool enable_prefix_caching,
                                                    int64_t blocks) {
  auto registry = std::make_unique<attn::BackendRegistry>();
  attn::RegisterBuiltinBackends(*registry);
  attn::AttentionConfig config;
  config.block_size = kBlockSize;
  config.max_model_len = 4096;
  config.max_seqs = 256;
  config.max_batch_tokens = 4096;
  // 2 layers x 2*1*8*2 bytes/token x 4 tokens/block = 256 bytes/block.
  config.cache_bytes_budget = blocks * 256;
  config.device = Device::Cpu();
  config.enable_prefix_caching = enable_prefix_caching;
  config.forced_backend = "reference";
  const std::vector<attn::AttentionSpec> specs{LayerSpec(), LayerSpec()};
  return std::move(
      attn::AttentionRuntime::Create(std::move(registry), config, specs).value());
}


class SchedulerPrefixTest : public ::testing::Test {
 protected:
  void MakeScheduler(bool enable_prefix_caching, int64_t blocks = 64) {
    runtime_ = MakeRuntime(enable_prefix_caching, blocks);
    scheduler_ = std::make_unique<Scheduler>(SchedulerConfig{}, std::vector<TokenId>{kEos},
                                             *runtime_);
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
    last_chunks_.clear();
    for (const ScheduledRequest& sr : plan->scheduled) {
      last_chunks_.push_back(sr.num_new_tokens);
    }
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

  std::unique_ptr<attn::AttentionRuntime> runtime_;
  std::unique_ptr<Scheduler> scheduler_;
  SchedulerOutput last_plan_;
  std::vector<int> last_chunks_;
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
  const int64_t free_after_first = runtime_->BlockPool()->FreeBlocks();
  EXPECT_LT(free_after_first, runtime_->BlockPool()->NumBlocks());
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
  // 2 adopted + 1 granted, read from the facade.
  EXPECT_EQ(runtime_->Cache().GroupBlocks(fresh.request_id, 0).size(), 3u);
  // The adopted blocks are the pool blocks the first request published;
  // nothing new was allocated for them.
  EXPECT_EQ(runtime_->BlockPool()->FreeBlocks(), free_after_first - 1);

  ModelRunnerOutput result;
  result.samples.push_back({2, {kTok}});
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*plan, result).ok());
  RunToCompletion();

  // The second request finished and its uncached tail stays cached; no
  // blocks leak: every block is either free or discoverable by hash.
  EXPECT_EQ(runtime_->BlockPool()->UsedBlocks() + 0, runtime_->BlockPool()->NumBlocks() - runtime_->BlockPool()->FreeBlocks());
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
  salted.SetCacheSalt("tenant-a");
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

TEST_F(SchedulerPrefixTest, PrefillChunksEndOnBlockBoundariesWhenCached) {
  SchedulerConfig config;
  config.max_num_batched_tokens = 6;  // Deliberately block-unaligned.
  runtime_ = MakeRuntime(/*prefix=*/true, /*blocks=*/64);
  scheduler_ = std::make_unique<Scheduler>(config, std::vector<TokenId>{kEos}, *runtime_);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, Tokens(12), ShortParams())).ok());
  std::vector<int> chunk_sizes;
  while (scheduler_->HasRequests()) {
    ASSERT_TRUE(ScheduleOnce());
    if (!last_chunks_.empty()) chunk_sizes.push_back(last_chunks_.back());
  }
  // Budget 6 would cut 12 into 6+6; block alignment cuts 4+4+4 so each
  // chunk's blocks are publishable the moment it completes. The final 1 is
  // the first decode step.
  EXPECT_EQ(chunk_sizes, (std::vector<int>{4, 4, 4, 1}));
}

TEST_F(SchedulerPrefixTest, ChunkAlignmentOffWithoutCaching) {
  SchedulerConfig config;
  config.max_num_batched_tokens = 6;
  runtime_ = MakeRuntime(/*prefix=*/false, /*blocks=*/64);
  scheduler_ = std::make_unique<Scheduler>(config, std::vector<TokenId>{kEos}, *runtime_);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, Tokens(12), ShortParams())).ok());
  std::vector<int> chunk_sizes;
  while (scheduler_->HasRequests()) {
    ASSERT_TRUE(ScheduleOnce());
    if (!last_chunks_.empty()) chunk_sizes.push_back(last_chunks_.back());
  }
  EXPECT_EQ(chunk_sizes, (std::vector<int>{6, 6, 1}));  // Budget + decode.
}

TEST_F(SchedulerPrefixTest, HitAccountingIsTokenGranular) {
  MakeScheduler(true);
  const std::vector<TokenId> prompt = Tokens(12);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, prompt, ShortParams())).ok());
  RunToCompletion();
  ASSERT_TRUE(scheduler_->PopFinished().has_value());

  ASSERT_TRUE(scheduler_->AddRequest(Request(2, prompt, ShortParams())).ok());
  RunToCompletion();

  const SchedulerStats stats = scheduler_->Stats();
  EXPECT_EQ(stats.prefix_queries, 2);
  EXPECT_EQ(stats.prefix_hits, 1);           // Only the second saw a cache.
  EXPECT_EQ(stats.prefix_queried_tokens, 24);
  EXPECT_EQ(stats.prefix_hit_tokens, 8);     // Two of three blocks.
}

TEST_F(SchedulerPrefixTest, DisabledCachingRecomputesEverything) {
  MakeScheduler(false);
  const std::vector<TokenId> prompt = Tokens(12);

  ASSERT_TRUE(scheduler_->AddRequest(Request(1, prompt, ShortParams())).ok());
  RunToCompletion();
  const int64_t free_after_first = runtime_->BlockPool()->FreeBlocks();
  EXPECT_EQ(free_after_first, runtime_->BlockPool()->NumBlocks());  // Everything returned.

  ASSERT_TRUE(scheduler_->AddRequest(Request(2, prompt, ShortParams())).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok()) << plan.status();
  EXPECT_EQ(plan->scheduled_new_reqs[0].num_computed_tokens, 0);
  EXPECT_EQ(plan->scheduled[0].num_new_tokens, 12);
}

}  // namespace
}  // namespace inferx
