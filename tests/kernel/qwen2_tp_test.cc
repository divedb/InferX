#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/comm/communicator.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/model/qwen2.h"

namespace inferx::model {
namespace {

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/"
         "snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

TEST(Qwen2TensorParallelTest, TwoRanksMatchSingleRankLogits) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  const std::string dir = CheckpointDir();
  std::vector<float> reference;
  {
    auto model = Qwen2Model::LoadFromDirectory(dir);
    if (!model.ok()) GTEST_SKIP() << "model not loaded: " << model.status();
    ASSERT_EQ(model->tensor_parallel_rank(), 0);
    ASSERT_EQ(model->tensor_parallel_size(), 1);
    ASSERT_TRUE(model->ForwardLastLogits({785, 6722, 315, 9625, 374},
                                         &reference)
                    .ok());
  }

  constexpr int kRanks = 2;  // checkpoint has two KV heads
  auto created = comm::CreateHostSimCommunicators(kRanks);
  ASSERT_TRUE(created.ok()) << created.status();
  auto communicators = std::move(*created);
  std::vector<Qwen2Model> models;
  models.reserve(kRanks);
  for (int rank = 0; rank < kRanks; ++rank) {
    auto model = Qwen2Model::LoadFromDirectory(
        dir, std::move(communicators[static_cast<size_t>(rank)]));
    ASSERT_TRUE(model.ok()) << "rank " << rank << ": " << model.status();
    EXPECT_EQ(model->tensor_parallel_rank(), rank);
    EXPECT_EQ(model->tensor_parallel_size(), kRanks);
    models.push_back(*std::move(model));
  }

  std::vector<std::vector<float>> logits(kRanks);
  std::vector<Status> statuses(kRanks);
  std::vector<std::thread> threads;
  for (int rank = 0; rank < kRanks; ++rank) {
    threads.emplace_back([&, rank] {
      statuses[rank] = models[rank].ForwardLastLogits(
          {785, 6722, 315, 9625, 374}, &logits[rank]);
    });
  }
  for (auto& thread : threads) thread.join();

  const auto reference_top = std::max_element(reference.begin(), reference.end()) -
                             reference.begin();
  for (int rank = 0; rank < kRanks; ++rank) {
    ASSERT_TRUE(statuses[rank].ok()) << "rank " << rank << ": " << statuses[rank];
    ASSERT_EQ(logits[rank].size(), reference.size());
    const auto top = std::max_element(logits[rank].begin(), logits[rank].end()) -
                     logits[rank].begin();
    EXPECT_EQ(top, reference_top);
    float max_abs = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i)
      max_abs = std::max(max_abs, std::abs(logits[rank][i] - reference[i]));
    EXPECT_LT(max_abs, 0.5f) << "rank " << rank;
  }
}

TEST(Qwen2TensorParallelTest, RejectsAWorldLargerThanTheKvHeadCount) {
  auto config = ModelConfig::FromDirectory(CheckpointDir());
  auto checkpoint = Checkpoint::Open(CheckpointDir());
  if (!config.ok() || !checkpoint.ok()) GTEST_SKIP() << "checkpoint unavailable";
  auto created = comm::CreateHostSimCommunicators(4);
  ASSERT_TRUE(created.ok()) << created.status();
  auto communicators = std::move(*created);
  auto model = Qwen2Model::Load(*config, *checkpoint,
                                std::move(communicators.front()));
  ASSERT_FALSE(model.ok());
  EXPECT_EQ(model.status().code(), absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::model
