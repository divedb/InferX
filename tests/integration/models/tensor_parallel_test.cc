// Tensor-parallel sharding, end to end on one GPU: two ModelRunners over a
// LoopbackWorld produce the runner's samples, and those samples must
// match the single-rank generation exactly. Exercising ModelRunner's rank
// group path this way covers weight sharding, collectives, and the
// rank-zero-samples protocol.
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/dist/loopback_comm.h"
#include "inferx/engine/model_runner.h"
#include "inferx/engine/worker.h"
#include "inferx/engine/scheduler.h"

namespace inferx {
namespace {

std::vector<TokenId> ReadCsv(const std::string& path) {
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  std::vector<TokenId> tokens;
  std::stringstream cells(line);
  std::string cell;
  while (std::getline(cells, cell, ',')) {
    if (!cell.empty()) tokens.push_back(std::stoi(cell));
  }
  return tokens;
}

StatusOr<std::unique_ptr<ModelRunner>> MakeRunner(int world) {
  ModelConfig model;
  model.model_dir = "models/Qwen3-0.6B";
  model.device.device = "cuda";
  CacheConfig cache;
  cache.block_size = 16;
  cache.num_gpu_blocks_override = 512;
  SchedulerConfig scheduler_config;
  ExecutionConfig execution;

  if (world == 1) {
    return ModelRunner::Create(model, cache, scheduler_config, execution);
  }

  auto comm_world = LoopbackWorld::Create(world);
  if (!comm_world.ok()) return comm_world.status();
  std::vector<std::unique_ptr<Worker>> ranks;
  for (int rank = 0; rank < world; ++rank) {
    ParallelConfig parallel;
    parallel.tensor_parallel_size = world;
    parallel.tensor_parallel_rank = rank;
    INFERX_ASSIGN_OR_RETURN(
        auto runner, Worker::Create(model, parallel, (*comm_world)->TakeRank(rank)));
    ranks.push_back(std::move(runner));
  }
  // The rank-owned comm backends keep the loopback state alive; the world
  // itself only needs to outlive CreateGroup, so the abort hook holds it.
  std::shared_ptr<LoopbackWorld> world_ptr(std::move(*comm_world));
  return ModelRunner::CreateGroup(
      std::move(ranks), cache, scheduler_config, execution,
      [world_ptr](const Status& status) { world_ptr->Abort(status); });
}

StatusOr<std::vector<TokenId>> Generate(ModelRunner& runner,
                                        const SchedulerConfig& config,
                                        const std::vector<TokenId>& prompt, int max_tokens,
                                        TokenId eos) {
  Scheduler scheduler(config, {eos}, *runner.Attention());
  sampling::SamplingParams params;
  params.temperature = 0;
  params.max_tokens = max_tokens;
  INFERX_RETURN_IF_ERROR(scheduler.AddRequest(Request(7, prompt, params)));

  std::vector<TokenId> output;
  while (scheduler.HasRequests()) {
    INFERX_ASSIGN_OR_RETURN(SchedulerOutput plan, scheduler.Schedule());
    if (plan.IsEmpty()) continue;
    INFERX_ASSIGN_OR_RETURN(ModelRunnerOutput result, runner.ExecuteModel(plan));
    INFERX_RETURN_IF_ERROR(scheduler.UpdateFromOutput(plan, result));
    for (const SampledTokens& sample : result.samples) {
      for (TokenId token : sample.token_ids) output.push_back(token);
    }
  }
  return output;
}

TEST(TensorParallelTest, TwoRanksMatchSingleRank) {
  const std::vector<TokenId> prompt = ReadCsv("tests/testdata/qwen3_0_6b/prompt_tokens.txt");
  const std::vector<TokenId> expected =
      ReadCsv("tests/testdata/qwen3_0_6b/expected_tokens.txt");
  ASSERT_EQ(expected.size(), 24u);

  SchedulerConfig config;
  auto single = MakeRunner(1);
  ASSERT_TRUE(single.ok()) << single.status();
  const TokenId eos = (*single)->GetModelSpec().eos_token_ids.front();

  auto reference =
      Generate(**single, config, prompt, static_cast<int>(expected.size()), eos);
  ASSERT_TRUE(reference.ok()) << reference.status();
  ASSERT_EQ(reference->size(), expected.size());
  // The golden trace itself is checked (tie-tolerantly) by model_test;
  // here the sharded ranks must reproduce THIS build's single-rank
  // decisions exactly, ties included.
  auto sharded = MakeRunner(2);
  ASSERT_TRUE(sharded.ok()) << sharded.status();
  auto sharded_output =
      Generate(**sharded, config, prompt, static_cast<int>(expected.size()), eos);
  ASSERT_TRUE(sharded_output.ok()) << sharded_output.status();
  EXPECT_EQ(*sharded_output, *reference);
}

}  // namespace
}  // namespace inferx
