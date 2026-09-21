// Unit tests for the concrete Scheduler: admission, the token budget,
// chunked prefill, decode progression, and finish handling, driven against
// a real KvBlockPool on the host device.

#include "inferx/engine/scheduler.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace inferx {
namespace {

constexpr TokenId kEos = 99;
constexpr int64_t kBlockSize = 4;

class SchedulerTest : public ::testing::Test {
 protected:
  void MakePool(int64_t blocks) {
    KvLayout layout;
    layout.entries_per_token = 2;
    layout.kv_heads = 1;
    layout.head_dim = 8;
    layout.dtype = DataType::kBFloat16;
    pool_ = std::make_unique<KvBlockPool>(
        std::move(KvBlockPool::Create(/*num_layers=*/2, blocks, kBlockSize, layout,
                                      DeviceId::Cpu())
                      .value()));
  }

  void MakeScheduler(SchedulerConfig config) {
    if (pool_ == nullptr) MakePool(64);
    scheduler_ = std::make_unique<Scheduler>(config, pool_.get(), kEos);
  }

  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<Scheduler> scheduler_;
};

ModelRunnerOutput ResultFor(const SchedulerOutput& output, TokenId token) {
  ModelRunnerOutput result;
  for (const ScheduledRequest& sr : output.scheduled) {
    SampledTokens sample;
    sample.request_id = sr.request_id;
    sample.token_ids = {token};
    result.samples.push_back(std::move(sample));
  }
  return result;
}

TEST_F(SchedulerTest, PrefillSchedulesWholePromptAndGrantsBlocks) {
  MakeScheduler(SchedulerConfig{});
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3, 4, 5})).ok());

  StatusOr<SchedulerOutput> output = scheduler_->Schedule();
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_EQ(output->scheduled.size(), 1);
  EXPECT_EQ(output->scheduled[0].num_new_tokens, 5);
  EXPECT_EQ(output->total_num_scheduled_tokens, 5);

  ASSERT_EQ(output->scheduled_new_reqs.size(), 1);
  EXPECT_EQ(output->scheduled_new_reqs[0].request_id, 1);
  // ceil(5 tokens / 4 per block) = 2 blocks.
  EXPECT_EQ(output->scheduled_new_reqs[0].block_ids.size(), 2);
}

TEST_F(SchedulerTest, DecodeStepsScheduleOneTokenEach) {
  MakeScheduler(SchedulerConfig{});
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3})).ok());

  StatusOr<SchedulerOutput> prefill = scheduler_->Schedule();
  ASSERT_TRUE(prefill.ok());
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*prefill, ResultFor(*prefill, 50)).ok());

  StatusOr<SchedulerOutput> decode = scheduler_->Schedule();
  ASSERT_TRUE(decode.ok());
  ASSERT_EQ(decode->scheduled.size(), 1);
  EXPECT_EQ(decode->scheduled[0].num_new_tokens, 1);
  ASSERT_EQ(decode->scheduled_cached_reqs.size(), 1);
  EXPECT_EQ(decode->scheduled_cached_reqs[0].num_computed_tokens, 4);
  EXPECT_TRUE(decode->scheduled_cached_reqs[0].new_block_ids.empty());
}

TEST_F(SchedulerTest, FinishesAtTokenCapAndFreesBlocks) {
  MakeScheduler(SchedulerConfig{});
  SamplingParams params;
  params.max_tokens = 2;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());

  auto step = [&](TokenId token) {
    StatusOr<SchedulerOutput> output = scheduler_->Schedule();
    EXPECT_TRUE(output.ok());
    return scheduler_->UpdateFromOutput(*output, ResultFor(*output, token));
  };
  ASSERT_TRUE(step(7).ok());   // First sample.
  ASSERT_TRUE(step(8).ok());   // Second sample hits the cap.
  EXPECT_FALSE(scheduler_->HasRequests());

  std::optional<Request> finished = scheduler_->PopFinished();
  ASSERT_TRUE(finished.has_value());
  EXPECT_EQ(finished->output(), std::vector<TokenId>({7, 8}));
  EXPECT_EQ(finished->finish_reason(), FinishReason::kLengthCapped);

  // The finish is reported to the runner on the next step, and the blocks
  // are back on the pool's free list.
  StatusOr<SchedulerOutput> drain = scheduler_->Schedule();
  ASSERT_TRUE(drain.ok());
  EXPECT_EQ(drain->finished_request_ids, std::vector<RequestId>({1}));
  EXPECT_EQ(pool_->free_blocks(), pool_->num_blocks());
}

TEST_F(SchedulerTest, FinishesOnEosUnlessIgnored) {
  MakeScheduler(SchedulerConfig{});
  SamplingParams params;
  params.max_tokens = 10;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());

  StatusOr<SchedulerOutput> prefill = scheduler_->Schedule();
  ASSERT_TRUE(prefill.ok());
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*prefill, ResultFor(*prefill, kEos)).ok());
  EXPECT_FALSE(scheduler_->HasRequests());
  EXPECT_EQ(scheduler_->PopFinished()->finish_reason(), FinishReason::kStopped);

  // ignore_eos keeps the request running through the EOS token.
  SamplingParams ignoring;
  ignoring.max_tokens = 10;
  ignoring.ignore_eos = true;
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, {1, 2, 3}, ignoring)).ok());
  StatusOr<SchedulerOutput> again = scheduler_->Schedule();
  ASSERT_TRUE(again.ok());
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*again, ResultFor(*again, kEos)).ok());
  EXPECT_TRUE(scheduler_->HasRequests());
}

TEST_F(SchedulerTest, MidPromptChunkSamplesAreDropped) {
  SchedulerConfig config;
  config.max_num_batched_tokens = 3;
  MakeScheduler(config);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3, 4, 5})).ok());

  // Chunk 1 covers part of the prompt; its sample must not extend output.
  StatusOr<SchedulerOutput> chunk = scheduler_->Schedule();
  ASSERT_TRUE(chunk.ok());
  EXPECT_EQ(chunk->scheduled[0].num_new_tokens, 3);
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*chunk, ResultFor(*chunk, 50)).ok());
  EXPECT_TRUE(scheduler_->PopFinished() == std::nullopt);

  // Chunk 2 completes the prompt and its sample counts.
  StatusOr<SchedulerOutput> rest = scheduler_->Schedule();
  ASSERT_TRUE(rest.ok());
  EXPECT_EQ(rest->scheduled[0].num_new_tokens, 2);
  EXPECT_EQ(rest->scheduled_cached_reqs[0].num_computed_tokens, 5);
}

TEST_F(SchedulerTest, BudgetCapsTokensAcrossRequests) {
  SchedulerConfig config;
  config.max_num_batched_tokens = 4;
  MakeScheduler(config);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3})).ok());
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, {4, 5})).ok());

  // Step 1: request 1's whole prompt (3 tokens) plus a 1-token first chunk
  // of request 2's prompt fills the budget exactly.
  StatusOr<SchedulerOutput> output = scheduler_->Schedule();
  ASSERT_TRUE(output.ok());
  EXPECT_EQ(output->total_num_scheduled_tokens, 4);
  EXPECT_EQ(output->scheduled_new_reqs.size(), 2);
  EXPECT_EQ(output->scheduled[0].num_new_tokens, 3);
  EXPECT_EQ(output->scheduled[1].num_new_tokens, 1);
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*output, ResultFor(*output, 10)).ok());

  // Step 2: request 1 decodes one token; request 2 finishes its prompt.
  StatusOr<SchedulerOutput> next = scheduler_->Schedule();
  ASSERT_TRUE(next.ok());
  ASSERT_EQ(next->scheduled.size(), 2);
  EXPECT_EQ(next->scheduled[0].num_new_tokens, 1);  // decode of request 1
  EXPECT_EQ(next->scheduled[1].num_new_tokens, 1);  // prompt tail of request 2
  EXPECT_EQ(next->scheduled_cached_reqs[1].num_computed_tokens, 2);
}

TEST_F(SchedulerTest, WaitingQueueRejectsWhenFull) {
  SchedulerConfig config;
  config.queue_capacity = 1;
  MakeScheduler(config);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1})).ok());
  EXPECT_EQ(scheduler_->AddRequest(Request(2, {2})).code(),
            absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(scheduler_->Stats().num_rejected, 1);
}

TEST_F(SchedulerTest, AbortFreesBlocksAndFinishesRequest) {
  MakeScheduler(SchedulerConfig{});
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3})).ok());
  StatusOr<SchedulerOutput> output = scheduler_->Schedule();
  ASSERT_TRUE(output.ok());
  EXPECT_LT(pool_->free_blocks(), pool_->num_blocks());

  EXPECT_EQ(scheduler_->AbortRequests({1}), std::vector<uint64_t>({1}));
  EXPECT_FALSE(scheduler_->HasRequests());
  EXPECT_EQ(pool_->free_blocks(), pool_->num_blocks());
  std::optional<Request> aborted = scheduler_->PopFinished();
  ASSERT_TRUE(aborted.has_value());
  EXPECT_EQ(aborted->finish_reason(), FinishReason::kAborted);
}

}  // namespace
}  // namespace inferx
