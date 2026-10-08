#include "inferx/engine/model_executor.h"
#include "inferx/models/model_runner.h"

#include <memory>
#include <optional>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/dist/loopback_comm.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model.h"
#include "inferx/ops/op_context.h"
#include "inferx/ops/gather.h"

namespace inferx {
namespace {

struct ObservedBatch {
  std::vector<int32_t> tokens, positions, qo, kv, last_page_len, logit_rows;
};

/// \brief A fake model: records every prepared batch and returns logits
/// whose argmax walks 10, 11, 12, ... across calls.
///
/// It runs on the CPU device, so int32 tensors are read directly through
/// their host pointers. If the runner ever performed architecture work
/// itself, this test would have nothing left to observe.
class TestModel final : public models::CausalLanguageModel {
 public:
  TestModel() {
    info_.architecture = "TestModel";
    info_.vocab_size = 128;
    info_.max_position_embeddings = 32;
    declared.layers = {
        models::PagedKvStateSpec{KvLayout{2, 1, 8, DataType::kBFloat16}}};
  }
  const models::ModelInfo& info() const override { return info_; }
  const models::ModelRequirements& requirements() const override { return declared; }
  models::ModelRequirements declared;

  StatusOr<models::HiddenStates> Forward(const models::ModelInputs& input,
                                         models::ForwardContext& fctx) override {
    observed_comm = &fctx.comm;
    EXPECT_EQ(fctx.comm.size(), expected_world);
    EXPECT_EQ(fctx.comm.rank(), expected_rank);
    INFERX_RETURN_IF_ERROR(injected_failure);
    if (use_collective) {
      INFERX_ASSIGN_OR_RETURN(auto empty,
                              Tensor::Empty(DataType::kBFloat16, Shape{0}, fctx.ops.Device()));
      INFERX_RETURN_IF_ERROR(fctx.comm.AllReduceSum(fctx.ops, empty));
    }
    EXPECT_NE(fctx.state.paged_kv, nullptr);
    EXPECT_EQ(fctx.state.layers.size(), declared.layers.size());
    auto read = [](const Tensor& t, int size) {
      return std::vector<int32_t>(t.DataAs<int32_t>(), t.DataAs<int32_t>() + size);
    };
    const auto& a = input.attention;
    batches.push_back({read(input.token_ids, a.num_tokens), read(a.positions, a.num_tokens),
                       read(a.qo_indptr, a.num_seqs + 1), read(a.kv_indptr, a.num_seqs + 1),
                       read(a.last_page_len, a.num_seqs), {}});
    INFERX_ASSIGN_OR_RETURN(
        Tensor hidden, Tensor::Empty(DataType::kBFloat16,
                                     Shape({std::max<int64_t>(a.num_tokens, 1), 8}),
                                     input.token_ids.Device()));
    return models::HiddenStates{std::move(hidden)};
  }

  StatusOr<models::Logits> ComputeLogits(const models::HiddenStates&,
                                         const models::LogitSelection& selection,
                                         models::ForwardContext& fctx) override {
    const int num_seqs = static_cast<int>(selection.rows.Numel());
    if (!batches.empty()) {
      batches.back().logit_rows =
          std::vector<int32_t>(selection.rows.DataAs<int32_t>(),
                               selection.rows.DataAs<int32_t>() + num_seqs);
    }
    INFERX_ASSIGN_OR_RETURN(
        Tensor logits, Tensor::Empty(DataType::kFloat32, Shape({num_seqs, info_.vocab_size}),
                                     fctx.ops.Device()));
    float* rows = logits.DataAs<float>();
    for (int i = 0; i < num_seqs; ++i) {
      float* row = rows + static_cast<int64_t>(i) * info_.vocab_size;
      // Uniform logits make the sampled token a pure function of the
      // request's seed and RNG offset, exposing stream-position drift.
      const float fill = uniform_logits ? 0.0f : -1.0f;
      for (int64_t j = 0; j < info_.vocab_size; ++j) row[j] = fill;
      if (!uniform_logits) row[next_token_ + token_offset] = 1.0f;
    }
    ++next_token_;
    return models::Logits{std::move(logits)};
  }

  bool uniform_logits = false;

  models::ModelInfo info_;
  std::vector<ObservedBatch> batches;
  dist::CommBackend* observed_comm = nullptr;
  int expected_world = 1;
  int expected_rank = 0;
  int token_offset = 0;
  bool use_collective = false;
  Status injected_failure;

 private:
  int next_token_ = 10;
};

class ModelRunnerTest : public ::testing::Test {
 protected:
  void MakeRunner(int budget, std::unique_ptr<dist::CommBackend> comm = nullptr) {
    ModelConfig mc;
    mc.device.device_type = "cpu";
    SchedulerConfig sc;
    sc.max_num_batched_tokens = budget;
    sc.max_num_seqs = 4;
    CacheConfig cc;
    cc.num_kv_blocks = 64;
    cc.block_size = 2;
    auto model = std::make_unique<TestModel>();
    model_ = model.get();
    auto runner =
        ModelRunner::Create(mc, cc, sc, ExecutionConfig{}, std::move(model), std::move(comm));
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

TEST_F(ModelRunnerTest, PassesTheSuppliedCommunicatorToTheModel) {
  auto comm = std::make_unique<dist::SingleRankComm>();
  auto* expected = comm.get();
  MakeRunner(8, std::move(comm));
  sampling::SamplingParams params;
  params.temperature = 0;
  params.max_tokens = 1;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());
  ASSERT_TRUE(Step().ok());
  EXPECT_EQ(model_->observed_comm, expected);
}

TEST(ModelRunnerStateTest, RejectsTensorParallelDeviceCountMismatch) {
  ModelConfig model;
  model.device.device_ids = {0};
  auto runner = ModelRunner::Create(model, CacheConfig{}, SchedulerConfig{}, ExecutionConfig{},
                                    ParallelConfig{2, 0});
  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument);
}

class RankGroupTest : public ::testing::Test {
 protected:
  void MakeGroup(bool fail_peer = false) {
    auto created = dist::LoopbackWorld::Create(2);
    ASSERT_TRUE(created.ok());
    std::shared_ptr<dist::LoopbackWorld> world = *std::move(created);
    ModelConfig model_config;
    model_config.device.device_type = "cpu";
    CacheConfig cache;
    cache.num_kv_blocks = 32;
    cache.block_size = 2;
    SchedulerConfig scheduling;
    scheduling.max_num_seqs = 2;
    scheduling.max_num_batched_tokens = 4;
    std::vector<std::unique_ptr<ModelRunner>> ranks;
    for (int rank = 0; rank < 2; ++rank) {
      auto model = std::make_unique<TestModel>();
      models_[rank] = model.get();
      model->expected_world = 2;
      model->expected_rank = rank;
      model->token_offset = rank * 50;  // Peer logits must never choose the next token.
      model->use_collective = true;
      if (rank == 1 && fail_peer)
        model->injected_failure = InternalError("injected rank failure");
      auto runner = ModelRunner::Create(model_config, cache, scheduling, ExecutionConfig{},
                                        std::move(model), world->TakeRank(rank));
      ASSERT_TRUE(runner.ok()) << runner.status();
      ranks.push_back(*std::move(runner));
    }
    auto group = ModelExecutor::CreateGroup(
        std::move(ranks), [world](const Status& status) { world->Abort(status); });
    ASSERT_TRUE(group.ok()) << group.status();
    group_ = *std::move(group);
    scheduler_ = std::make_unique<Scheduler>(scheduling, group_->kv_pool(), 127);
  }
  std::unique_ptr<ModelExecutor> group_;
  std::unique_ptr<Scheduler> scheduler_;
  TestModel* models_[2]{};
};

TEST_F(RankGroupTest, SharesRankZeroSamplesAndRequestLifecycle) {
  MakeGroup();
  sampling::SamplingParams params;
  params.temperature = 0;
  params.ignore_eos = true;
  params.max_tokens = 4;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3}, params)).ok());
  int steps = 0;
  while (scheduler_->HasRequests() && ++steps < 10) {
    auto plan = scheduler_->Schedule();
    ASSERT_TRUE(plan.ok());
    auto result = group_->Execute(*plan);
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_TRUE(scheduler_->UpdateFromOutput(*plan, *result).ok());
  }
  ASSERT_FALSE(scheduler_->HasRequests());
  auto finished = scheduler_->PopFinished();
  ASSERT_TRUE(finished.has_value());
  EXPECT_EQ(finished->output(), (std::vector<int>{10, 11, 12, 13}));
  ASSERT_EQ(models_[0]->batches.size(), models_[1]->batches.size());
  for (size_t step = 0; step < models_[0]->batches.size(); ++step) {
    EXPECT_EQ(models_[0]->batches[step].tokens, models_[1]->batches[step].tokens);
    EXPECT_EQ(models_[0]->batches[step].positions, models_[1]->batches[step].positions);
  }
  auto removals = scheduler_->Schedule();
  ASSERT_TRUE(removals.ok());
  ASSERT_TRUE(group_->Execute(*removals).ok());
  // Reuse the request id after every rank has processed the finish notification.
  params.temperature = 0.8f;
  params.seed = 123;
  params.max_tokens = 2;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {7}, params)).ok());
  for (int step = 0; step < 2; ++step) {
    auto plan = scheduler_->Schedule();
    ASSERT_TRUE(plan.ok());
    auto result = group_->Execute(*plan);
    ASSERT_TRUE(result.ok()) << result.status();
    ASSERT_TRUE(scheduler_->UpdateFromOutput(*plan, *result).ok());
    EXPECT_EQ(models_[0]->batches.back().tokens, models_[1]->batches.back().tokens);
  }
  EXPECT_FALSE(scheduler_->HasRequests());
}

TEST_F(RankGroupTest, RankFailureReleasesPeerBlockedInCollective) {
  MakeGroup(true);
  sampling::SamplingParams params;
  params.temperature = 0;
  params.max_tokens = 2;
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, {1}, params)).ok());
  auto plan = scheduler_->Schedule();
  ASSERT_TRUE(plan.ok());
  const auto result = group_->Execute(*plan);
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInternal);
  EXPECT_EQ(result.status().message(), "injected rank failure");
  EXPECT_EQ(group_->Execute(*plan).status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST_F(ModelRunnerTest, ChunkedPrefillSamplesOnlyPromptCompletion) {
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
  EXPECT_EQ(model_->batches[3].tokens, (std::vector<int32_t>{10}));
  EXPECT_EQ(model_->batches[3].positions, (std::vector<int32_t>{5}));
  // Mid-prompt chunks request no logits; the first sample lands on the
  // chunk that completes the prompt, so chunking cannot shift the stream.
  EXPECT_TRUE(model_->batches[0].logit_rows.empty());
  EXPECT_TRUE(model_->batches[1].logit_rows.empty());
  EXPECT_EQ(model_->batches[2].logit_rows, (std::vector<int32_t>{0}));
  auto finished = scheduler_->PopFinished();
  ASSERT_TRUE(finished.has_value());
  EXPECT_EQ(finished->output(), (std::vector<int>{10, 11}));
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
  // Request 1 completes its prompt this step and selects a row; request 2
  // received a mid-prompt chunk (budget) and selects none.
  EXPECT_EQ(model_->batches[0].logit_rows, (std::vector<int32_t>{2}));
  EXPECT_EQ(model_->batches[1].tokens, (std::vector<int32_t>{10, 5}));
  EXPECT_EQ(model_->batches[1].positions, (std::vector<int32_t>{3, 1}));
  EXPECT_EQ(model_->batches[1].qo, (std::vector<int32_t>{0, 1, 2}));
  EXPECT_EQ(model_->batches[1].kv, (std::vector<int32_t>{0, 2, 3}));
  EXPECT_EQ(model_->batches[1].last_page_len, (std::vector<int32_t>{2, 2}));
}

/// RNG advancement must depend only on sampled outputs, never on how the
/// scheduler chunked the prompt: the same seeded request produces the
/// identical token stream under both budgets.
TEST_F(ModelRunnerTest, ChunkedPromptPreservesSeededSamplingStream) {
  const auto run = [&](int budget) -> std::vector<int> {
    MakeRunner(budget);
    model_->uniform_logits = true;
    sampling::SamplingParams params;
    params.temperature = 1.0f;
    params.seed = 7;
    params.max_tokens = 4;
    EXPECT_TRUE(scheduler_->AddRequest(Request(1, {1, 2, 3, 4, 5}, params)).ok());
    while (scheduler_->HasRequests()) {
      const Status step = Step();
      if (!step.ok()) ADD_FAILURE() << "step failed: " << step;
    }
    auto finished = scheduler_->PopFinished();
    if (!finished.has_value()) {
      ADD_FAILURE() << "request did not finish";
      return {};
    }
    return finished->output();
  };
  const auto unchunked = run(8);
  const auto chunked = run(2);
  ASSERT_EQ(unchunked.size(), 4u);
  EXPECT_EQ(unchunked, chunked);
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
  model_->info_.max_position_embeddings = 2;
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
  model->declared.layers = {
      models::PagedKvStateSpec{KvLayout{2, 2, 4, DataType::kFloat32}}};
  ModelConfig mc;
  mc.device.device_type = "cpu";
  CacheConfig cc;
  cc.num_kv_blocks = 2;
  auto runner =
      ModelRunner::Create(mc, cc, SchedulerConfig{}, ExecutionConfig{}, std::move(model));
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_EQ((*runner)->kv_pool()->Layout().kv_heads, 2);
  EXPECT_EQ((*runner)->kv_pool()->Layout().head_dim, 4);
  EXPECT_EQ((*runner)->kv_pool()->Layout().dtype, DataType::kFloat32);
}

TEST(ModelRunnerStateTest, AllocatesMixedLayoutsAndRecurrentState) {
  auto model = std::make_unique<TestModel>();
  // A hybrid stack: per-layer paged layouts plus a recurrent layer.
  model->declared.layers = {models::PagedKvStateSpec{KvLayout{2, 2, 4, DataType::kBFloat16}},
                                models::RecurrentStateSpec{1, 2, 4, 4, 4}};
  ModelConfig mc;
  mc.device.device_type = "cpu";
  CacheConfig cc;
  cc.num_kv_blocks = 2;
  auto runner =
      ModelRunner::Create(mc, cc, SchedulerConfig{}, ExecutionConfig{}, std::move(model));
  ASSERT_TRUE(runner.ok()) << runner.status();
  EXPECT_EQ((*runner)->kv_pool()->LayoutFor(0).kv_heads, 2);
}

TEST(ModelRunnerStateTest, RejectsMissingLayerState) {
  auto model = std::make_unique<TestModel>();
  model->declared.layers.clear();
  auto runner = ModelRunner::Create(ModelConfig{}, CacheConfig{}, SchedulerConfig{},
                                    ExecutionConfig{}, std::move(model));
  EXPECT_EQ(runner.status().code(), absl::StatusCode::kInvalidArgument);
}

// A graph-safe model with a known answer: token (position + 10). Its output
// depends on live device positions and logit-row indices, including after a
// smaller batch reuses a previously captured graph.
class PositionModel final : public models::CausalLanguageModel {
 public:
  PositionModel() {
    info_.architecture = "PositionModel";
    info_.vocab_size = 128;
    info_.max_position_embeddings = 64;
  }
  Status Init() {
    const auto device = DeviceId::Cuda(0);
    INFERX_ASSIGN_OR_RETURN(table_,
                            Tensor::Empty(DataType::kBFloat16, Shape({64, 128}), device));
    INFERX_ASSIGN_OR_RETURN(hidden_,
                            Tensor::Empty(DataType::kBFloat16, Shape({8, 128}), device));
    INFERX_ASSIGN_OR_RETURN(logits_,
                            Tensor::Empty(DataType::kBFloat16, Shape({4, 128}), device));
    std::vector<uint16_t> table(64 * 128, 0);
    for (int p = 0; p < 64; ++p) table[p * 128 + p + 10] = 0x3f80;
    INFERX_ASSIGN_OR_RETURN(auto runtime, RuntimeFor(device));
    return runtime->Copy(table_->Data(), table.data(), table.size() * sizeof(uint16_t),
                         CopyKind::kHostToDevice);
  }

  const models::ModelInfo& info() const override { return info_; }
  const models::ModelRequirements& requirements() const override { return requirements_; }
  // This model is graph-safe by construction; eligibility is its own claim.
  StatusOr<models::StepExecutionInfo> PrepareStep(const models::ModelInputs&,
                                                  const models::LogitSelection&,
                                                  models::ForwardContext&) override {
    models::StepExecutionInfo info;
    info.graph.eligible = true;
    return info;
  }
  StatusOr<models::HiddenStates> Forward(const models::ModelInputs& input,
                                         models::ForwardContext& fctx) override {
    ++calls;
    INFERX_ASSIGN_OR_RETURN(auto hidden, hidden_->Slice(0, input.attention.num_tokens));
    INFERX_RETURN_IF_ERROR(ops::GatherRows(fctx.ops, *table_, input.attention.positions, hidden));
    return models::HiddenStates{std::move(hidden)};
  }
  StatusOr<models::Logits> ComputeLogits(const models::HiddenStates& states,
                                         const models::LogitSelection& selection,
                                         models::ForwardContext& fctx) override {
    INFERX_ASSIGN_OR_RETURN(auto logits, logits_->Slice(0, selection.rows.Numel()));
    INFERX_RETURN_IF_ERROR(ops::GatherRows(fctx.ops, states.hidden, selection.rows, logits));
    return models::Logits{std::move(logits)};
  }
  int calls = 0;

 private:
  models::ModelInfo info_;
  models::ModelRequirements requirements_{
      {models::PagedKvStateSpec{KvLayout{2, 1, 8, DataType::kBFloat16}}}};
  std::optional<Tensor> table_, hidden_, logits_;
};

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
      ASSERT_TRUE(
          scheduler.AddRequest(Request(wave * 4 + i, std::vector<int>(3 + i, 1), params)).ok());
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
