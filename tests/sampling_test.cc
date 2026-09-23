// Sampler module tests that need no GPU: parameter validation, metadata
// resolution, and the CPU greedy path (which must agree with the CUDA op's
// lowest-index tie break).
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/tensor.h"
#include "inferx/sampling/sampler.h"

namespace {

using inferx::DeviceId;
using inferx::Status;
using inferx::Tensor;
using inferx::ops::ExecutionContext;
using inferx::sampling::SamplingMetadata;
using inferx::sampling::SamplingParams;
using inferx::sampling::Sampler;
using inferx::sampling::SamplerOutput;

TEST(SamplingParamsTest, DefaultsValidate) {
  EXPECT_TRUE(SamplingParams().Validate().ok());
}

TEST(SamplingParamsTest, RejectsOutOfRangeFields) {
  SamplingParams p;
  p.temperature = -0.1f;
  EXPECT_FALSE(p.Validate().ok());
  p = SamplingParams();
  p.top_p = 0.0f;
  EXPECT_FALSE(p.Validate().ok());
  p = SamplingParams();
  p.min_p = 1.5f;
  EXPECT_FALSE(p.Validate().ok());
  p = SamplingParams();
  p.presence_penalty = 3.0f;
  EXPECT_FALSE(p.Validate().ok());
  p = SamplingParams();
  p.repetition_penalty = 0.0f;
  EXPECT_FALSE(p.Validate().ok());
  p = SamplingParams();
  p.temperature = std::nanf("");
  EXPECT_FALSE(p.Validate().ok());
}

TEST(SamplingParamsTest, GreedyDetection) {
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  EXPECT_TRUE(greedy.IsGreedy());
  SamplingParams topk1;
  topk1.top_k = 1;
  EXPECT_TRUE(topk1.IsGreedy());
  EXPECT_FALSE(SamplingParams().IsGreedy());  // Default temperature is 1.
}

TEST(SamplingMetadataTest, ResolvesPerRequestKnobs) {
  SamplingParams a;
  a.temperature = 0.0f;
  SamplingParams b;
  b.temperature = 0.8f;
  b.top_p = 0.9f;
  b.seed = 1234;
  const SamplingMetadata::PerRequest batch[] = {{&a, 7}, {&b, 0}};
  const SamplingMetadata metadata = SamplingMetadata::Build(1000, batch);
  EXPECT_EQ(metadata.batch, 2);
  EXPECT_FALSE(metadata.all_greedy);
  EXPECT_TRUE(metadata.requests[0].greedy);
  EXPECT_EQ(metadata.requests[0].rng_offset, 7);
  EXPECT_FALSE(metadata.requests[1].greedy);
  EXPECT_EQ(metadata.requests[1].top_p, 0.9f);
  EXPECT_EQ(metadata.requests[1].seed, 1234u);
  EXPECT_EQ(metadata.requests[1].rng_offset, 0);
}

class SamplerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = inferx::RuntimeFor(DeviceId::Cpu());
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok()) << stream.status();
    stream_ = *stream;
  }

  Tensor MakeLogits(const std::vector<std::vector<float>>& rows) {
    const int batch = static_cast<int>(rows.size());
    const int64_t vocab = static_cast<int64_t>(rows[0].size());
    auto tensor = Tensor::Empty(inferx::DataType::kFloat, inferx::Shape({batch, vocab}),
                                DeviceId::Cpu());
    EXPECT_TRUE(tensor.ok()) << tensor.status();
    float* out = (*tensor).DataAs<float>();
    for (int i = 0; i < batch; ++i)
      for (int64_t j = 0; j < vocab; ++j) out[i * vocab + j] = rows[i][j];
    return *std::move(tensor);
  }

  inferx::DeviceRuntime* runtime_ = nullptr;
  inferx::Stream stream_;
};

TEST_F(SamplerTest, GreedyCpuPathPicksArgmaxWithLowestIndexTieBreak) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  // Row 0: clear max at 5. Row 1: tie between 2 and 6 -> lowest index wins.
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6},
                                    {1, 0, 7, 1, 2, 3, 7, 0}});
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  const SamplingMetadata::PerRequest batch[] = {{&greedy, 0}, {&greedy, 3}};
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  ExecutionContext ctx(*runtime_, stream_);
  SamplerOutput output;
  ASSERT_TRUE((*sampler)->Sample(ctx, logits, metadata, output).ok());
  ASSERT_TRUE(output.IsDefined());
  const int32_t* ids = output.sampled_token_ids.DataAs<int32_t>();
  EXPECT_EQ(ids[0], 5);
  EXPECT_EQ(ids[1], 2);  // Tie break matches the CUDA op.
}

TEST_F(SamplerTest, NonGreedyBatchIsReportedUnimplemented) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6}});
  SamplingParams random;  // Default temperature is 1: not greedy.
  const SamplingMetadata::PerRequest batch[] = {{&random, 0}};
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  ExecutionContext ctx(*runtime_, stream_);
  SamplerOutput output;
  const Status status = (*sampler)->Sample(ctx, logits, metadata, output);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

TEST_F(SamplerTest, RejectsMetadataMismatch) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6},
                                    {7, 0, 7, 1, 2, 3, 7, 0}});
  SamplingParams greedy;
  greedy.temperature = 0.0f;
  const SamplingMetadata::PerRequest batch[] = {{&greedy, 0}};  // One of two rows.
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  ExecutionContext ctx(*runtime_, stream_);
  SamplerOutput output;
  EXPECT_FALSE((*sampler)->Sample(ctx, logits, metadata, output).ok());
}

}  // namespace
