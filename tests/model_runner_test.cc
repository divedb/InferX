#include "inferx/models/model_runner.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/gather.h"

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
  const CheckpointConfig& config() const override { return config_; }

  std::vector<LayerStateSpec> StateRequirements() const override { return requirements; }
  std::vector<LayerStateSpec> requirements = {PagedKvStateSpec{KvLayout{2, 1, 8, DataType::kBFloat16}}};

  StatusOr<Tensor> Forward(const ModelInput& input, ModelState& state,
                           ops::ExecutionContext& /*ctx*/) override {
    EXPECT_NE(state.paged_kv, nullptr);
    EXPECT_EQ(state.layers.size(), requirements.size());
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

  CheckpointConfig config_;
  std::vector<ObservedBatch> batches;

 private:
  int next_token_ = 10;
};

class ModelRunnerTest : public ::testing::Test {
 protected:
  void MakeRunner(int budget) {
    ModelConfig mc;
    mc.device = DeviceId::Cpu();
    SchedulerConfig sc;
    sc.max_num_batched_tokens = budget;
    sc.max_num_seqs = 4;
    CacheConfig cc;
    cc.num_kv_blocks = 64;
    cc.block_size = 2;
    auto model = std::make_unique<TestModel>();
    model_ = model.get();
    auto runner = ModelRunner::Create(mc, cc, sc, ExecutionConfig{}, std::move(model));
    ASSERT_TRUE(runner.ok()) << runner.status();
    runner_ = std::move(*runner);
    scheduler_ = std::make_unique<Scheduler>(sc, runner_->kv_pool(), 127);
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
  sampling::SamplingParams params;
  params.temperature = 0;
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
  sampling::SamplingParams params;
  params.temperature = 0;
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
  sampling::SamplingParams greedy;
  greedy.temperature = 0;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, greedy)).ok());
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, {4, 5}, greedy)).ok());
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
  sampling::SamplingParams params;
  params.temperature = 0;
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

TEST(ModelRunnerStateTest, UsesDeclaredLayoutInsteadOfLegacyDimensions) {
  auto model = std::make_unique<TestModel>();
  model->requirements = {PagedKvStateSpec{KvLayout{2, 2, 4, DataType::kFloat}}};
  ModelConfig mc;
  mc.device = DeviceId::Cpu();
  CacheConfig cc;
  cc.num_kv_blocks = 2;
  auto runner = ModelRunner::Create(mc, cc, SchedulerConfig{}, ExecutionConfig{}, std::move(model));
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_EQ((*runner)->kv_pool()->layout().kv_heads, 2);
  EXPECT_EQ((*runner)->kv_pool()->layout().head_dim, 4);
  EXPECT_EQ((*runner)->kv_pool()->layout().dtype, DataType::kFloat);
}

TEST(ModelRunnerStateTest, RejectsRecurrentStateBeforeAllocation) {
  auto model = std::make_unique<TestModel>();
  model->requirements = {RecurrentStateSpec{1, 2, 4, 4, 4}};
  // Default CUDA device deliberately exercises rejection before device setup.
  auto runner = ModelRunner::Create(ModelConfig{}, CacheConfig{}, SchedulerConfig{},
                                   ExecutionConfig{}, std::move(model));
  EXPECT_EQ(runner.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelRunnerStateTest, RejectsMixedPagedLayoutsBeforeAllocation) {
  auto model = std::make_unique<TestModel>();
  model->config_.num_hidden_layers = 2;
  model->requirements.push_back(PagedKvStateSpec{KvLayout{2, 2, 4, DataType::kBFloat16}});
  auto runner = ModelRunner::Create(ModelConfig{}, CacheConfig{}, SchedulerConfig{},
                                   ExecutionConfig{}, std::move(model));
  EXPECT_EQ(runner.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelRunnerStateTest, RejectsMissingLayerState) {
  auto model = std::make_unique<TestModel>();
  model->requirements.clear();
  auto runner = ModelRunner::Create(ModelConfig{}, CacheConfig{}, SchedulerConfig{},
                                   ExecutionConfig{}, std::move(model));
  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument);
}

// A graph-safe model with a known answer: token (position + 10). Its output
// depends on live device positions and logit-row indices, including after a
// smaller batch reuses a previously captured graph.
class PositionModel final : public Model {
 public:
  PositionModel() {
    config_.num_hidden_layers = 1;
    config_.vocab_size = 128;
    config_.max_position_embeddings = 64;
  }
  Status Init() {
    const auto device = DeviceId::Cuda(0);
    INFERX_ASSIGN_OR_RETURN(table_, Tensor::Empty(DataType::kBFloat16, Shape({64, 128}), device));
    INFERX_ASSIGN_OR_RETURN(hidden_, Tensor::Empty(DataType::kBFloat16, Shape({8, 128}), device));
    INFERX_ASSIGN_OR_RETURN(logits_, Tensor::Empty(DataType::kBFloat16, Shape({4, 128}), device));
    std::vector<uint16_t> table(64 * 128, 0);
    for (int p = 0; p < 64; ++p) table[p * 128 + p + 10] = 0x3f80;
    INFERX_ASSIGN_OR_RETURN(auto runtime, RuntimeFor(device));
    return runtime->Copy(table_.Data(), table.data(), table.size() * sizeof(uint16_t),
                         CopyKind::kHostToDevice);
  }
  bool SupportsCudaGraphs() const override { return true; }
  const CheckpointConfig& config() const override { return config_; }
  std::vector<LayerStateSpec> StateRequirements() const override {
    return {PagedKvStateSpec{KvLayout{2, 1, 8, DataType::kBFloat16}}};
  }
  StatusOr<Tensor> Forward(const ModelInput& input, ModelState&,
                           ops::ExecutionContext& ctx) override {
    ++calls;
    INFERX_ASSIGN_OR_RETURN(auto hidden, hidden_.Slice(0, input.attention.num_tokens));
    INFERX_ASSIGN_OR_RETURN(auto logits, logits_.Slice(0, input.attention.num_seqs));
    INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, table_, input.attention.positions, hidden));
    INFERX_RETURN_IF_ERROR(ops::GatherRows(ctx, hidden, input.logit_rows, logits));
    return logits;
  }
  int calls = 0;
 private:
  CheckpointConfig config_;
  Tensor table_, hidden_, logits_;
};

TEST_F(ModelRunnerTest, RejectsUnknownAttentionBackendBeforeLoadingWeights) {
  ModelConfig mc;
  mc.model_dir = "/nonexistent";
  ExecutionConfig ec;
  ec.attention_backend = "cutlass";
  auto runner = ModelRunner::Create(mc, CacheConfig{}, SchedulerConfig{}, ec);
  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(std::string(runner.status().message()).find("attention backend"), std::string::npos);
}

TEST(ModelRunnerCudaTest, GraphSamplingTracksBatchTurnoverAndPageTransitions) {
  ModelConfig mc;
  SchedulerConfig sc;
  sc.max_num_batched_tokens = 8;
  sc.max_num_seqs = 4;
  CacheConfig cc;
  cc.num_kv_blocks = 64;
  cc.block_size = 2;
  ExecutionConfig ec;
  ec.enable_cuda_graphs = true;
  auto model = std::make_unique<PositionModel>();
  ASSERT_TRUE(model->Init().ok());
  auto* observed = model.get();
  auto created = ModelRunner::Create(mc, cc, sc, ec, std::move(model));
  ASSERT_TRUE(created.ok()) << created.status();
  auto runner = std::move(*created);
  Scheduler scheduler(sc, runner->kv_pool(), 127);
  int steps = 0;
  for (int wave = 0; wave < 2; ++wave) {
    for (int i = 0; i < 4; ++i) {
      sampling::SamplingParams params;
      params.temperature = 0;
      params.ignore_eos = true;
      params.max_tokens = 8 + 3 * i;
      ASSERT_TRUE(scheduler.AddRequest(Request(wave * 4 + i,
          std::vector<int>(3 + i, 1), params)).ok());
    }
    while (scheduler.HasRequests()) {
      auto plan = scheduler.Schedule();
      ASSERT_TRUE(plan.ok()) << plan.status();
      auto result = runner->Run(*plan);
      ASSERT_TRUE(result.ok()) << result.status();
      ASSERT_TRUE(scheduler.UpdateFromOutput(*plan, *result).ok());
      ++steps;
      while (auto finished = scheduler.PopFinished()) {
        const int i = finished->id() % 4;
        ASSERT_EQ(finished->output().size(), 8 + 3 * i);
        for (int j = 0; j < 8 + 3 * i; ++j) {
          EXPECT_EQ(finished->output()[j], 12 + i + j);
        }
      }
    }
  }
  EXPECT_LT(observed->calls, steps);
}

}  // namespace
}  // namespace inferx
