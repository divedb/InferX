#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/dist/loopback_comm.h"
#include "inferx/dist/nccl_comm.h"
#include "inferx/engine/scheduler.h"
#include "inferx/engine/model_executor.h"

namespace inferx {
namespace {

class ObservedModel final : public models::CausalLanguageModel {
 public:
  explicit ObservedModel(std::unique_ptr<models::CausalLanguageModel> model)
      : model_(std::move(model)) {}
  const models::ModelInfo& info() const override { return model_->info(); }
  const models::ModelRequirements& requirements() const override {
    return model_->requirements();
  }
  StatusOr<models::HiddenStates> Forward(const models::ModelInputs& input,
                                         models::ForwardContext& fctx) override {
    return model_->Forward(input, fctx);
  }
  StatusOr<models::Logits> ComputeLogits(const models::HiddenStates& states,
                                         const models::LogitSelection& selection,
                                         models::ForwardContext& fctx) override {
    INFERX_ASSIGN_OR_RETURN(auto logits, model_->ComputeLogits(states, selection, fctx));
    std::vector<uint16_t> bits(logits.values.Numel());
    INFERX_RETURN_IF_ERROR(fctx.ops.Runtime().CopyAsync(
        bits.data(), logits.values.Data(), logits.values.NBytes(), CopyKind::kDeviceToHost,
        fctx.ops.GetStream()));
    INFERX_RETURN_IF_ERROR(fctx.ops.Runtime().SynchronizeStream(fctx.ops.GetStream()));
    std::vector<float> values;
    for (uint16_t value : bits) values.push_back(std::bit_cast<float>(uint32_t(value) << 16));
    steps.push_back(std::move(values));
    return logits;
  }
  std::vector<std::vector<float>> steps;

 private:
  std::unique_ptr<models::CausalLanguageModel> model_;
};

class TensorParallelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_.model_dir = "models/Qwen3-0.6B";
    if (!std::filesystem::exists(model_.model_dir + "/model.safetensors")) {
      GTEST_SKIP() << "local Qwen3-0.6B checkpoint is absent";
    }
    auto runtime = RuntimeFor(DeviceId::Cuda(0));
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    cache_.num_kv_blocks = 32;
    cache_.block_size = 2;  // Exercise page transitions during both prefill and decode.
    scheduler_.max_num_seqs = 2;
    scheduler_.max_num_batched_tokens = 4;
  }

  StatusOr<std::unique_ptr<ModelRunner>> MakeRank(int rank, int size,
                                                  std::unique_ptr<dist::CommBackend> comm,
                                                  ObservedModel** observed) {
    models::ModelLoadOptions options;
    options.directory = model_.model_dir;
    options.device = DeviceId::Cuda(0);
    options.max_tokens = 4;
    options.max_seqs = 2;
    options.parallel = ParallelConfig{size, rank};
    INFERX_ASSIGN_OR_RETURN(auto model, models::LoadModel(options));
    for (const auto& state : model->requirements().layers) {
      EXPECT_EQ(std::get<models::PagedKvStateSpec>(state).layout.kv_heads, 8 / size);
    }
    auto wrapper = std::make_unique<ObservedModel>(std::move(model));
    *observed = wrapper.get();
    return ModelRunner::Create(model_, cache_, scheduler_, ExecutionConfig{},
                               std::move(wrapper), std::move(comm));
  }

  template <class ExecutorT>
  static StatusOr<ModelRunnerOutput> Step(ExecutorT& ex, const SchedulerOutput& plan) {
    return ex.Execute(plan);
  }
  static StatusOr<ModelRunnerOutput> Step(ModelRunner& rank, const SchedulerOutput& plan) {
    return rank.Run(plan);
  }

  template <class ExecutorT>
  StatusOr<std::map<RequestId, std::vector<TokenId>>> Generate(ExecutorT& runner) {
    Scheduler scheduler(scheduler_, runner.kv_pool(), 151645);
    sampling::SamplingParams params;
    params.temperature = 0;
    params.ignore_eos = true;
    params.max_tokens = 4;
    // Tokens span both halves of the vocabulary. Unequal prompt lengths test
    // chunked prefill, mixed prefill/decode, and gathering multiple logit rows.
    INFERX_RETURN_IF_ERROR(
        scheduler.AddRequest(Request(1, {151644, 9707, 11, 80000, 13}, params)));
    INFERX_RETURN_IF_ERROR(scheduler.AddRequest(Request(2, {90000, 42, 151643}, params)));
    std::map<RequestId, std::vector<TokenId>> generated;
    int steps = 0;
    while (scheduler.HasRequests()) {
      if (++steps > 20) return InternalError("generation did not finish");
      INFERX_ASSIGN_OR_RETURN(auto plan, scheduler.Schedule());
      INFERX_ASSIGN_OR_RETURN(auto result, Step(runner, plan));
      INFERX_RETURN_IF_ERROR(scheduler.UpdateFromOutput(plan, result));
      while (auto done = scheduler.PopFinished()) generated[done->id()] = done->output();
    }
    INFERX_ASSIGN_OR_RETURN(auto removals, scheduler.Schedule());
    INFERX_RETURN_IF_ERROR(Step(runner, removals).status());
    return generated;
  }

  DeviceRuntime* runtime_ = nullptr;
  ModelConfig model_;
  CacheConfig cache_;
  SchedulerConfig scheduler_;
};

TEST_F(TensorParallelTest, Qwen3TwoRanksMatchSingleRankPrefillAndDecode) {
  ObservedModel* reference = nullptr;
  auto baseline = MakeRank(0, 1, std::make_unique<dist::SingleRankComm>(), &reference);
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  auto expected = Generate(**baseline);
  ASSERT_TRUE(expected.ok()) << expected.status();
  const auto expected_logits = reference->steps;
  baseline->reset();

  auto created = dist::LoopbackWorld::Create(2);
  ASSERT_TRUE(created.ok());
  std::shared_ptr<dist::LoopbackWorld> world = *std::move(created);
  std::vector<std::unique_ptr<ModelRunner>> ranks;
  std::vector<ObservedModel*> observed(2);
  for (int rank = 0; rank < 2; ++rank) {
    auto runner = MakeRank(rank, 2, world->TakeRank(rank), &observed[rank]);
    ASSERT_TRUE(runner.ok()) << runner.status();
    ranks.push_back(*std::move(runner));
  }
  auto parallel = ModelExecutor::CreateGroup(
      std::move(ranks), [world](const Status& status) { world->Abort(status); });
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  auto actual = Generate(**parallel);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(*actual, *expected);
  ASSERT_EQ(observed[0]->steps.size(), expected_logits.size());
  EXPECT_EQ(observed[0]->steps, observed[1]->steps);
  for (size_t step = 0; step < expected_logits.size(); ++step) {
    SCOPED_TRACE(step);
    const auto& want = expected_logits[step];
    const auto& got = observed[0]->steps[step];
    ASSERT_EQ(got.size(), want.size());
    double error = 0, scale = 0;
    for (size_t i = 0; i < want.size(); ++i) {
      ASSERT_TRUE(std::isfinite(got[i]));
      error += std::pow(double(got[i]) - want[i], 2);
      scale += double(want[i]) * want[i];
    }
    // Unlike TP=1's full FP32 GEMM accumulation, each sharded projection
    // rounds its partial to BF16 before reduction. Across 28 layers this is
    // approximate parity; also require exact rank agreement and generated ids.
    const double relative_l2 = std::sqrt(error / std::max(scale, 1e-12));
    std::printf("Qwen3 TP=2 step %zu relative logit L2: %.5f\n", step, relative_l2);
    EXPECT_LT(relative_l2, 0.05);
  }
}

TEST_F(TensorParallelTest, Qwen3MultiGpuServingRunnerMatchesSingleRank) {
  if (runtime_->DeviceCount() < 2) GTEST_SKIP() << "two CUDA GPUs are required";
  auto id = dist::NcclComm::NewUniqueId();
  if (absl::IsNotFound(id.status())) GTEST_SKIP() << "NCCL is not installed";
  ASSERT_TRUE(id.ok()) << id.status();
  auto baseline = ModelExecutor::Create(model_, cache_, scheduler_, ExecutionConfig{});
  ASSERT_TRUE(baseline.ok()) << baseline.status();
  auto expected = Generate(**baseline);
  ASSERT_TRUE(expected.ok()) << expected.status();
  baseline->reset();
  model_.device.device_ids = {0, 1};
  auto parallel =
      ModelExecutor::Create(model_, cache_, scheduler_, ExecutionConfig{}, ParallelConfig{2, 0});
  ASSERT_TRUE(parallel.ok()) << parallel.status();
  auto actual = Generate(**parallel);
  ASSERT_TRUE(actual.ok()) << actual.status();
  EXPECT_EQ(*actual, *expected);
}

}  // namespace
}  // namespace inferx
