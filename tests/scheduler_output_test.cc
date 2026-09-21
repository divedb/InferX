// Unit tests for the SchedulerOutput structs.

#include "inferx/engine/scheduler_output.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/engine/request.h"

namespace inferx {
namespace {

TEST(SchedulerOutputTest, DefaultStepIsEmpty) {
  const SchedulerOutput output;
  EXPECT_TRUE(output.IsEmpty());
  EXPECT_EQ(output.total_num_scheduled_tokens, 0);
  EXPECT_TRUE(output.scheduled.empty());
  EXPECT_TRUE(output.scheduled_new_reqs.empty());
  EXPECT_TRUE(output.scheduled_cached_reqs.empty());
  EXPECT_TRUE(output.finished_request_ids.empty());
}

TEST(SchedulerOutputTest, MixedStepCarriesNewCachedAndFinished) {
  const std::vector<TokenId> prompt{1, 2, 3};

  NewRequestData new_req;
  new_req.request_id = 7;
  new_req.prompt_token_ids = absl::MakeConstSpan(prompt);
  new_req.block_ids = {0, 1};
  new_req.num_computed_tokens = 0;

  CachedRequestUpdate cached;
  cached.request_id = 9;
  cached.new_block_ids = {5};
  cached.num_computed_tokens = 41;

  SchedulerOutput output;
  output.scheduled = {ScheduledRequest{7, 1}, ScheduledRequest{9, 42}};
  output.scheduled_new_reqs = {new_req};
  output.scheduled_cached_reqs = {cached};
  output.total_num_scheduled_tokens = 43;
  output.finished_request_ids = {4};

  EXPECT_FALSE(output.IsEmpty());
  EXPECT_EQ(output.scheduled.size(), 2);
  EXPECT_EQ(output.total_num_scheduled_tokens,
            output.scheduled[0].num_new_tokens + output.scheduled[1].num_new_tokens);

  // The prompt is a borrow: the span aliases the scheduler's storage rather
  // than copying it.
  EXPECT_EQ(new_req.prompt_token_ids.data(), prompt.data());
  EXPECT_EQ(new_req.prompt_token_ids.size(), prompt.size());
  EXPECT_TRUE(output.scheduled_cached_reqs[0].new_block_ids == std::vector<int32_t>{5});
}

TEST(SchedulerOutputTest, SampledTokensFinishReasonIsOptional) {
  SampledTokens sample;
  sample.request_id = 7;
  sample.token_ids = {42};
  EXPECT_EQ(sample.finish_reason, std::nullopt);

  sample.finish_reason = FinishReason::kLengthCapped;
  EXPECT_EQ(sample.finish_reason, FinishReason::kLengthCapped);
}

}  // namespace
}  // namespace inferx
