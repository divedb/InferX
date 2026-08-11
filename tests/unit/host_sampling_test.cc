#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "engine/host_sampling.h"

namespace inferx::engine {
namespace {

TEST(HostSamplingTest, GreedySelectsLargestLogit) {
  std::vector<float> logits = {1.0f, 4.0f, 2.0f};
  model::ForwardBatch batch;

  EXPECT_EQ(HostSampleRow(logits.data(), logits.size(), batch, 0, nullptr), 1);
}

TEST(HostSamplingTest, MaskAndPresencePenaltyMutateBeforeSelection) {
  std::vector<float> logits = {5.0f, 4.0f, 3.0f};
  model::ForwardBatch batch;
  batch.presence_penalty = {3.0f};
  batch.penalty_history_ids.assign(model::ForwardBatch::kPenaltyHistoryCap,
                                   -1);
  batch.penalty_history_counts.assign(model::ForwardBatch::kPenaltyHistoryCap,
                                      0);
  batch.penalty_history_ids[0] = 0;
  batch.penalty_history_counts[0] = 1;
  batch.mask_token_ids.assign(model::ForwardBatch::kMaskCap, -1);
  batch.mask_token_ids[0] = 1;

  EXPECT_EQ(HostSampleRow(logits.data(), logits.size(), batch, 0, nullptr), 2);
}

TEST(HostSamplingTest, SeededTruncatedSamplingIsReproducible) {
  model::ForwardBatch batch;
  batch.temperature = {0.8f};
  batch.top_p = {1.0f};
  batch.top_k = {2};
  batch.seeds = {12345};

  std::vector<float> first = {5.0f, 4.0f, 1.0f, 0.0f};
  std::vector<float> second = first;
  const int32_t a =
      HostSampleRow(first.data(), first.size(), batch, 0, nullptr);
  const int32_t b =
      HostSampleRow(second.data(), second.size(), batch, 0, nullptr);

  EXPECT_EQ(a, b);
  EXPECT_TRUE(a == 0 || a == 1);
}

TEST(HostSamplingTest, ReportsChosenAndTopLogprobs) {
  std::vector<float> logits = {0.0f, 1.0f, 2.0f};
  model::ForwardBatch batch;
  batch.logprobs_k = {2};
  SampledLogprob result;

  EXPECT_EQ(HostSampleRow(logits.data(), logits.size(), batch, 0, &result), 2);
  ASSERT_TRUE(result.present);
  const float log_z = std::log(std::exp(0.0f) + std::exp(1.0f) +
                               std::exp(2.0f));
  EXPECT_NEAR(result.logprob, 2.0f - log_z, 1e-6f);
  ASSERT_EQ(result.top.size(), 2);
  EXPECT_EQ(result.top[0].first, 2);
  EXPECT_NEAR(result.top[0].second, 2.0f - log_z, 1e-6f);
  EXPECT_EQ(result.top[1].first, 1);
  EXPECT_NEAR(result.top[1].second, 1.0f - log_z, 1e-6f);
}

}  // namespace
}  // namespace inferx::engine
