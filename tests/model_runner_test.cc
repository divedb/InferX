#include "inferx/models/model_runner.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model.h"
#include "inferx/ops/execution_context.h"

namespace inferx {
namespace {

struct ObservedBatch {
  std::vector<int32_t> tokens, positions, qo, kv, last_page_len, logit_rows;
};

/// \brief A fake model: records every prepared ModelInput and returns logits
/// whose argmax walks 10, 11, 12, ... across calls.
///
/// It runs on the CPU device, so int32 tensors are read directly through
/// their host pointers. If the runner ever performed architecture work
/// itself, this test would have nothing left to observe.
class TestModel final : public Model {
 public:
  TestModel() {
    config_.num_hidden_layers = 1;
    config_.num_key_value_heads = 1;
    config_.head_dim = 8;
    config_.hidden_size = 8;
    config_.vocab_size = 128;
    config_.max_position_embeddings = 32;
  }
  const ModelConfig& config() const override { return config_; }

  StatusOr<Tensor> Forward(const ModelInput& input, KvBlockPool& /*cache*/,
                           ops::ExecutionContext& /*ctx*/) override {
    auto read = [](const Tensor& t, int size) {
      return std::vector<int32_t>(t.DataAs<int32_t>(), t.DataAs<int32_t>() + size);
    };
    const auto& a = input.attention;
    batches.push_back({read(input.token_ids, a.num_tokens), read(a.positions, a.num_tokens),
                       read(a.qo_indptr, a.num_seqs + 1), read(a.kv_indptr, a.num_seqs + 1),
                       read(a.last_page_len, a.num_seqs), read(input.logit_rows, a.num_seqs)});
    const int num_seqs = a.num_seqs;
    INFERX_ASSIGN_OR_RETURN(
        Tensor logits, Tensor::Empty(DataType::kFloat, Shape({num_seqs, config_.vocab_size}),
                                     input.token_ids.Device()));
    float* rows = logits.DataAs<float>();
    for (int i = 0; i < num_seqs; ++i) {
      float* row = rows + static_cast<int64_t>(i) * config_.vocab_size;
      for (int64_t j = 0; j < config_.vocab_size; ++j) row[j] = -1.0f;
      row[next_token_] = 1.0f;
    }
    ++next_token_;
    return logits;
  }

  ModelConfig config_;
  std::vector<ObservedBatch> batches;

 private:
  int next_token_ = 10;
};

class ModelRunnerTest : public ::testing::Test {
 protected:
  void MakeRunner(int budget) {
    ModelRunnerConfig config;
    config.device = DeviceId::Cpu();
    config.max_num_batched_tokens = budget;
    config.max_num_seqs = 4;
    config.num_kv_blocks = 64;
    config.block_size = 2;
    auto model = std::make_unique<TestModel>();
    model_ = model.get();
    auto runner = ModelRunner::Create(config, std::move(model));
    ASSERT_TRUE(runner.ok()) << runner.status();
    runner_ = std::move(*runner);
    SchedulerConfig scheduler_config;
    scheduler_config.max_num_batched_tokens = budget;
    scheduler_config.max_num_seqs = 4;
    scheduler_ = std::make_unique<Scheduler>(scheduler_config, runner_->kv_pool(), 127);
  }
  Status Step() {
    INFERX_ASSIGN_OR_RETURN(auto output, scheduler_->Schedule());
    INFERX_ASSIGN_OR_RETURN(auto result, runner_->Run(output));
    return scheduler_->UpdateFromOutput(output, result);
  }
  TestModel* model_ = nullptr;
  std::unique_ptr<ModelRunner> runner_;
  std::unique_ptr<Scheduler> scheduler_;
};

TEST_F(ModelRunnerTest, RepeatedDecodeUsesLastSampleAndAdvancesPositions) {
  MakeRunner(8);
  SamplingParams params;
  params.max_tokens = 4;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(Step().ok());
  ASSERT_FALSE(scheduler_->HasRequests());
  ASSERT_EQ(model_->batches.size(), 4);
  EXPECT_EQ(model_->batches[0].tokens, (std::vector<int32_t>{1, 2, 3}));
  for (int i = 1; i < 4; ++i) {
    EXPECT_EQ(model_->batches[i].tokens, (std::vector<int32_t>{9 + i}));
    EXPECT_EQ(model_->batches[i].positions, (std::vector<int32_t>{2 + i}));
  }
  auto finished = scheduler_->PopFinished();
  ASSERT_TRUE(finished.has_value());
  EXPECT_EQ(finished->output(), (std::vector<int>{10, 11, 12, 13}));
}

TEST_F(ModelRunnerTest, ChunkedPrefillDoesNotSkipPromptTokens) {
  MakeRunner(2);
  SamplingParams params;
  params.max_tokens = 2;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3, 4, 5}, params)).ok());
  for (int i = 0; i < 4; ++i) ASSERT_TRUE(Step().ok());
  ASSERT_FALSE(scheduler_->HasRequests());
  ASSERT_EQ(model_->batches.size(), 4);
  EXPECT_EQ(model_->batches[0].tokens, (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(model_->batches[1].tokens, (std::vector<int32_t>{3, 4}));
  EXPECT_EQ(model_->batches[1].positions, (std::vector<int32_t>{2, 3}));
  EXPECT_EQ(model_->batches[2].tokens, (std::vector<int32_t>{5}));
  EXPECT_EQ(model_->batches[3].tokens, (std::vector<int32_t>{12}));
  EXPECT_EQ(model_->batches[3].positions, (std::vector<int32_t>{5}));
  auto finished = scheduler_->PopFinished();
  ASSERT_TRUE(finished.has_value());
  EXPECT_EQ(finished->output(), (std::vector<int>{12, 13}));
}

TEST_F(ModelRunnerTest, MixedPrefillAndDecodePreserveBatchBoundaries) {
  MakeRunner(4);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3})).ok());
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, {4, 5})).ok());
  ASSERT_TRUE(Step().ok());
  ASSERT_TRUE(Step().ok());
  ASSERT_EQ(model_->batches.size(), 2);
  EXPECT_EQ(model_->batches[0].qo, (std::vector<int32_t>{0, 3, 4}));
  EXPECT_EQ(model_->batches[0].logit_rows, (std::vector<int32_t>{2, 3}));
  EXPECT_EQ(model_->batches[1].tokens, (std::vector<int32_t>{10, 5}));
  EXPECT_EQ(model_->batches[1].positions, (std::vector<int32_t>{3, 1}));
  EXPECT_EQ(model_->batches[1].qo, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(model_->batches[1].kv, (std::vector<int32_t>{0, 2, 3}));
  EXPECT_EQ(model_->batches[1].last_page_len, (std::vector<int32_t>{2, 2}));
}

TEST_F(ModelRunnerTest, FinishOnlyStepAllowsRequestIdReuse) {
  MakeRunner(4);
  SamplingParams params;
  params.max_tokens = 1;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1}, params)).ok());
  ASSERT_TRUE(Step().ok());
  ASSERT_TRUE(Step().ok());  // Finish-only notification.
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {2}, params)).ok());
  ASSERT_TRUE(Step().ok());
  ASSERT_EQ(model_->batches.size(), 2);
  EXPECT_EQ(model_->batches[1].tokens, (std::vector<int32_t>{2}));
  EXPECT_EQ(model_->batches[1].positions, (std::vector<int32_t>{0}));
}

TEST_F(ModelRunnerTest, RejectsContextOverflowBeforeForward) {
  MakeRunner(4);
  model_->config_.max_position_embeddings = 2;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3})).ok());
  EXPECT_EQ(Step().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(model_->batches.empty());
  EXPECT_EQ(runner_->Run({}).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST_F(ModelRunnerTest, RejectsInvalidTokenBeforeForward) {
  MakeRunner(4);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {128})).ok());
  EXPECT_EQ(Step().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(model_->batches.empty());
}

}  // namespace
}  // namespace inferx
