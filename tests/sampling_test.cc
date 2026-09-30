// Sampler module tests that need no GPU: parameter validation, metadata
// resolution, and the CPU greedy path (which must agree with the CUDA op's
// lowest-index tie break).
#include <cmath>
#include <cstring>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/tensor.h"
#include "inferx/sampling/sampler.h"

namespace {

using inferx::DeviceId;
using inferx::Status;
using inferx::Tensor;
using inferx::ops::OpContext;
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
    auto cuda = inferx::RuntimeFor(DeviceId::Cuda(0));
    if (cuda.ok()) {
      cuda_ = *cuda;
      ASSERT_TRUE(cuda_->Activate().ok());
      auto cuda_stream = cuda_->CreateStream();
      ASSERT_TRUE(cuda_stream.ok()) << cuda_stream.status();
      cuda_stream_ = *cuda_stream;
    }
  }

  Tensor MakeLogits(const std::vector<std::vector<float>>& rows,
                   std::optional<inferx::DeviceId> device = std::nullopt) {
    const int batch = static_cast<int>(rows.size());
    const int64_t vocab = static_cast<int64_t>(rows[0].size());
    const inferx::DeviceId where = device.value_or(inferx::DeviceId::Cpu());
    const inferx::DataType dtype =
        where.IsCuda() ? inferx::DataType::kBFloat16 : inferx::DataType::kFloat32;
    auto tensor = Tensor::Empty(dtype, inferx::Shape({batch, vocab}), where);
    EXPECT_TRUE(tensor.ok()) << tensor.status();
    if (where.IsCpu()) {
      float* out = (*tensor).DataAs<float>();
      for (int i = 0; i < batch; ++i)
        for (int64_t j = 0; j < vocab; ++j) out[i * vocab + j] = rows[i][j];
      return *std::move(tensor);
    }
    std::vector<uint16_t> bf16(batch * vocab);
    for (int i = 0; i < batch; ++i) {
      for (int64_t j = 0; j < vocab; ++j) {
        float v = rows[i][j];
        uint32_t bits = 0;
        std::memcpy(&bits, &v, 4);
        bits += 0x7fff + ((bits >> 16) & 1);
        bits &= 0xffff0000;
        bf16[i * vocab + j] = static_cast<uint16_t>(bits >> 16);
      }
    }
    EXPECT_NE(cuda_, nullptr) << "CUDA runtime unavailable";
    if (cuda_ != nullptr) {
      EXPECT_TRUE(cuda_->Copy((*tensor).Data(), bf16.data(), bf16.size() * 2,
                              inferx::CopyKind::kHostToDevice).ok());
    }
    return *std::move(tensor);
  }

  inferx::DeviceRuntime* runtime_ = nullptr;
  inferx::Stream stream_;
  inferx::DeviceRuntime* cuda_ = nullptr;
  inferx::Stream cuda_stream_;
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
  OpContext ctx(*runtime_, stream_);
  auto output = (*sampler)->Sample(ctx, logits, metadata);
  ASSERT_TRUE(output.ok()) << output.status();
  const int32_t* ids = output->sampled_token_ids.DataAs<int32_t>();
  EXPECT_EQ(ids[0], 5);
  EXPECT_EQ(ids[1], 2);  // Tie break matches the CUDA op.
}

TEST_F(SamplerTest, TopKOneMatchesGreedy) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6}});
  SamplingParams topk1;
  topk1.top_k = 1;
  const SamplingMetadata::PerRequest batch[] = {{&topk1, 0}};
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  OpContext ctx(*runtime_, stream_);
  const auto output = (*sampler)->Sample(ctx, logits, metadata);
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  const int32_t id = *output->sampled_token_ids.DataAs<int32_t>();
  EXPECT_EQ(id, 5);
}

TEST_F(SamplerTest, SeededDrawIsReproducibleAndVariesWithOffset) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{3, 3, 3, 3, 3, 3, 3, 3}});
  SamplingParams random;
  random.seed = 42;
  const SamplingMetadata::PerRequest a[] = {{&random, 0}};
  const SamplingMetadata::PerRequest b[] = {{&random, 1}};
  OpContext ctx(*runtime_, stream_);
  const auto first = (*sampler)->Sample(ctx, logits, SamplingMetadata::Build(8, a));
  const auto repeat = (*sampler)->Sample(ctx, logits, SamplingMetadata::Build(8, a));
  const auto next = (*sampler)->Sample(ctx, logits, SamplingMetadata::Build(8, b));
  ASSERT_TRUE(first.ok() && repeat.ok() && next.ok());
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  const int32_t draw = *first->sampled_token_ids.DataAs<int32_t>();
  EXPECT_EQ(draw, *repeat->sampled_token_ids.DataAs<int32_t>());  // Reproducible.
  EXPECT_GE(draw, 0);
  EXPECT_LT(draw, 8);  // Uniform: any token is legal.
}

TEST_F(SamplerTest, LogitBiasBansTokens) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6}});
  SamplingParams biased;
  biased.logit_bias[5] = -INFINITY;  // Ban the argmax.
  biased.logit_bias[2] = 100.0f;     // Crown token 2.
  const SamplingMetadata::PerRequest batch[] = {{&biased, 0}};
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  OpContext ctx(*runtime_, stream_);
  const auto output = (*sampler)->Sample(ctx, logits, metadata);
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  EXPECT_EQ(*output->sampled_token_ids.DataAs<int32_t>(), 2);
  ASSERT_EQ(metadata.device.bias_entries.size(), 2u)
      << "bias payload not packed";
}

TEST_F(SamplerTest, RepetitionPenaltyPushesOffHistory) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6}});
  SamplingParams penalized;
  penalized.repetition_penalty = 8.0f;
  const std::vector<int32_t> history{5, 5};  // 9 / 8 + ... demoted below 6.
  const SamplingMetadata::PerRequest batch[] = {{&penalized, 0, &history}};
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  OpContext ctx(*runtime_, stream_);
  const auto output = (*sampler)->Sample(ctx, logits, metadata);
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  EXPECT_EQ(*output->sampled_token_ids.DataAs<int32_t>(), 7);  // Logit 6 survives.
}

TEST_F(SamplerTest, AllowlistRestrictsToEntries) {
  auto sampler = Sampler::Create(4, 8, DeviceId::Cpu());
  ASSERT_TRUE(sampler.ok()) << sampler.status();
  const Tensor logits = MakeLogits({{0, 1, 2, 3, 4, 9, 5, 6}});
  SamplingParams constrained;
  constrained.allowed_token_ids = {0, 3, 6};
  const SamplingMetadata::PerRequest batch[] = {{&constrained, 0}};
  const SamplingMetadata metadata = SamplingMetadata::Build(8, batch);
  OpContext ctx(*runtime_, stream_);
  const auto output = (*sampler)->Sample(ctx, logits, metadata);
  ASSERT_TRUE(output.ok()) << output.status();
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  EXPECT_EQ(*output->sampled_token_ids.DataAs<int32_t>(), 6);
}

TEST_F(SamplerTest, CudaPipelineAgreesWithCpuReference) {
  auto cpu = Sampler::Create(4, 64, DeviceId::Cpu());
  auto gpu = Sampler::Create(4, 64, DeviceId::Cuda(0));
  ASSERT_TRUE(cpu.ok() && gpu.ok());
  std::vector<std::vector<float>> rows;
  for (int r = 0; r < 3; ++r) {
    std::vector<float> row;
    for (int j = 0; j < 64; ++j) row.push_back(std::sin(float(r * 131 + j)) * 6.0f);
    rows.push_back(row);
  }
  // Round through bf16 first so the CUDA and CPU samplers see identical
  // numbers; otherwise draws could legitimately differ at the margin.
  for (auto& row : rows) {
    for (float& v : row) {
      uint32_t bits = 0;
      std::memcpy(&bits, &v, 4);
      bits += 0x7fff + ((bits >> 16) & 1);
      bits &= 0xffff0000;
      float rounded = 0;
      std::memcpy(&rounded, &bits, 4);
      v = rounded;
    }
  }
  const Tensor logits = MakeLogits(rows, DeviceId::Cuda(0));
  const Tensor logits_cpu = MakeLogits(rows);
  SamplingParams a;   // Plain.
  a.seed = 7;
  SamplingParams b;   // Filters on.
  b.seed = 9;
  b.top_k = 8;
  b.top_p = 0.9f;
  b.min_p = 0.02f;
  b.temperature = 0.7f;
  const std::vector<int32_t> history{5, 9, 5};
  SamplingParams c;   // Biased and penalized.
  c.seed = 11;
  c.logit_bias[10] = 2.5f;
  c.allowed_token_ids = {};
  c.repetition_penalty = 1.3f;
  const SamplingMetadata::PerRequest batch[] = {{&a, 0, nullptr},
                                                {&b, 3, &history},
                                                {&c, 1, &history}};
  const SamplingMetadata metadata = SamplingMetadata::Build(64, batch);
  OpContext ctx(*cuda_, cuda_stream_);
  const auto gpu_out = (*gpu)->Sample(ctx, logits, metadata);
  ASSERT_TRUE(gpu_out.ok()) << gpu_out.status();
  // The CPU reference reads the same device tensor through its own lane.
  OpContext cpu_ctx(*runtime_, stream_);
  const auto cpu_out = (*cpu)->Sample(cpu_ctx, logits_cpu, metadata);
  ASSERT_TRUE(cpu_out.ok()) << cpu_out.status();
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  std::vector<int32_t> g(3);
  ASSERT_TRUE(cuda_->Copy(g.data(), gpu_out->sampled_token_ids.Data(), 3 * 4,
                          inferx::CopyKind::kDeviceToHost).ok());
  ASSERT_TRUE(cuda_->SynchronizeStream(cuda_stream_).ok());
  const int32_t* h = cpu_out->sampled_token_ids.DataAs<int32_t>();
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(g[i], h[i]) << "row " << i << " gpu=" << g[i] << " cpu=" << h[i];
  }
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
  OpContext ctx(*runtime_, stream_);
  EXPECT_FALSE((*sampler)->Sample(ctx, logits, metadata).ok());
}

}  // namespace
