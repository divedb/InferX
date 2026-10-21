// Unit tests for the concrete Scheduler: admission, the token budget,
// chunked prefill, decode progression, and finish handling, driven against
// a real KvBlockPool on the host device.

#include "inferx/engine/scheduler.h"

#include "inferx/ops/attention/registry.h"
#include "inferx/ops/attention/runtime.h"
#include "inferx/ops/attention/spec.h"


#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace inferx {
namespace {

constexpr TokenId kEos = 99;
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


class SchedulerTest : public ::testing::Test {
 protected:
  void MakeScheduler(SchedulerConfig config) {
    if (runtime_ == nullptr) runtime_ = MakeRuntime(/*prefix=*/true, /*blocks=*/64);
    scheduler_ = std::make_unique<Scheduler>(config, std::vector<TokenId>{kEos}, *runtime_);
  }

  std::unique_ptr<attn::AttentionRuntime> runtime_;
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

TEST_F(SchedulerTest, FailedAdmissionAndAbortDoNotLeakBlocks) {
  runtime_ = MakeRuntime(/*prefix=*/true, /*blocks=*/2);
  MakeScheduler(SchedulerConfig{});
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, std::vector<TokenId>(12, 1))).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok());
  EXPECT_TRUE(plan->scheduled.empty());
  EXPECT_EQ(runtime_->BlockPool()->FreeBlocks(), 2);
  EXPECT_EQ(scheduler_->AbortRequests({1}), std::vector<uint64_t>({1}));
  EXPECT_EQ(runtime_->BlockPool()->FreeBlocks(), 2);
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, {1, 2, 3, 4, 5})).ok());
  auto next = scheduler_->Schedule();
  ASSERT_TRUE(next.ok());
  ASSERT_EQ(next->scheduled_new_reqs.size(), 1);
  // ceil(5 tokens / 4 per block) = 2 blocks, read from the facade.
  EXPECT_EQ(runtime_->Cache().GroupBlocks(next->scheduled_new_reqs[0].request_id, 0).size(),
            2u);
}

TEST_F(SchedulerTest, DestructionReleasesActiveAllocations) {
  MakeScheduler(SchedulerConfig{});
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3})).ok());
  ASSERT_TRUE(scheduler_->Schedule().ok());
  EXPECT_LT(runtime_->BlockPool()->FreeBlocks(), runtime_->BlockPool()->NumBlocks());
  scheduler_.reset();
  EXPECT_EQ(runtime_->BlockPool()->NumFreeBlocks(), runtime_->BlockPool()->NumBlocks());
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
  // ceil(5 tokens / 4 per block) = 2 blocks, read from the facade.
  EXPECT_EQ(runtime_->Cache().GroupBlocks(1, 0).size(), 2u);
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
  EXPECT_EQ(runtime_->Cache().GroupBlocks(decode->scheduled[0].request_id, 0).size(), 1u);
}

TEST_F(SchedulerTest, FinishesAtTokenCapAndFreesBlocks) {
  MakeScheduler(SchedulerConfig{});
  sampling::SamplingParams params;
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
  EXPECT_EQ(finished->Output(), std::vector<TokenId>({7, 8}));
  EXPECT_EQ(finished->GetFinishReason(), FinishReason::kLengthCapped);

  // The finish is reported to the runner on the next step, and the blocks
  // are back on the pool's free list.
  StatusOr<SchedulerOutput> drain = scheduler_->Schedule();
  ASSERT_TRUE(drain.ok());
  EXPECT_EQ(drain->finished_request_ids, std::vector<RequestId>({1}));
  EXPECT_EQ(runtime_->BlockPool()->NumFreeBlocks(), runtime_->BlockPool()->NumBlocks());
}

TEST_F(SchedulerTest, FinishesOnEosUnlessIgnored) {
  MakeScheduler(SchedulerConfig{});
  sampling::SamplingParams params;
  params.max_tokens = 10;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());

  StatusOr<SchedulerOutput> prefill = scheduler_->Schedule();
  ASSERT_TRUE(prefill.ok());
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*prefill, ResultFor(*prefill, kEos)).ok());
  EXPECT_FALSE(scheduler_->HasRequests());
  EXPECT_EQ(scheduler_->PopFinished()->GetFinishReason(), FinishReason::kStopped);

  // ignore_eos keeps the request running through the EOS token.
  sampling::SamplingParams ignoring;
  ignoring.max_tokens = 10;
  ignoring.ignore_eos = true;
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, {1, 2, 3}, ignoring)).ok());
  StatusOr<SchedulerOutput> again = scheduler_->Schedule();
  ASSERT_TRUE(again.ok());
  ASSERT_TRUE(scheduler_->UpdateFromOutput(*again, ResultFor(*again, kEos)).ok());
  EXPECT_TRUE(scheduler_->HasRequests());
}

TEST_F(SchedulerTest, ExplicitStopsOverrideLengthAndReleaseCache) {
  MakeScheduler(SchedulerConfig{});
  for (bool text_stop : {false, true}) {
    sampling::SamplingParams params;
    params.max_tokens = 1;
    params.ignore_eos = true;
    if (!text_stop) params.stop_token_ids = {7};
    ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());
    auto plan = scheduler_->Schedule();
    ASSERT_TRUE(plan.ok());
    auto result = ResultFor(*plan, 7);
    if (text_stop) result.samples[0].finish_reason = FinishReason::kStopped;
    ASSERT_TRUE(scheduler_->UpdateFromOutput(*plan, result).ok());
    auto finished = scheduler_->PopFinished();
    ASSERT_TRUE(finished.has_value());
    EXPECT_EQ(finished->GetFinishReason(), FinishReason::kStopped);
    EXPECT_EQ(finished->NumOutputTokens(), 1);  // Includes the stopping token.
    EXPECT_FALSE(scheduler_->HasRequests());
    EXPECT_EQ(runtime_->BlockPool()->NumFreeBlocks(), runtime_->BlockPool()->NumBlocks());
  }
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
  EXPECT_LT(runtime_->BlockPool()->FreeBlocks(), runtime_->BlockPool()->NumBlocks());

  EXPECT_EQ(scheduler_->AbortRequests({1}), std::vector<uint64_t>({1}));
  EXPECT_FALSE(scheduler_->HasRequests());
  EXPECT_EQ(runtime_->BlockPool()->NumFreeBlocks(), runtime_->BlockPool()->NumBlocks());
  std::optional<Request> aborted = scheduler_->PopFinished();
  ASSERT_TRUE(aborted.has_value());
  EXPECT_EQ(aborted->GetFinishReason(), FinishReason::kAborted);
}

}  // namespace
}  // namespace inferx
