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

ForwardBatch Batch(const std::vector<int32_t>& ids, int32_t block,
                   int64_t start) {
  ForwardBatch batch;
  batch.token_ids = ids;
  batch.num_seqs = 1;
  batch.max_blocks_per_seq = 2;
  batch.block_table = {block, 0};
  for (size_t i = 0; i < ids.size(); ++i) {
    const int64_t position = start + static_cast<int64_t>(i);
    batch.positions.push_back(static_cast<int32_t>(position));
    batch.seq_of_token.push_back(0);
    batch.slots.push_back(block * 16 + static_cast<int32_t>(position));
  }
  batch.logits_indices = {static_cast<int32_t>(ids.size() - 1)};
  return batch;
}

void ExpectClose(const std::vector<float>& got,
                 const std::vector<float>& reference, int rank) {
  ASSERT_EQ(got.size(), reference.size());
  const auto expected_top =
      std::max_element(reference.begin(), reference.end()) - reference.begin();
  const auto top = std::max_element(got.begin(), got.end()) - got.begin();
  EXPECT_EQ(top, expected_top);
  float max_abs = 0.0f;
  for (size_t i = 0; i < reference.size(); ++i)
    max_abs = std::max(max_abs, std::abs(got[i] - reference[i]));
  EXPECT_LT(max_abs, 0.5f) << "rank " << rank;
}

TEST(Qwen2TensorParallelTest, TwoRanksMatchSingleRankLogits) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  const std::string dir = CheckpointDir();
  const std::vector<int32_t> prompt = {785, 6722, 315, 9625, 374};
  std::vector<float> reference, reference_prefill, reference_decode;
  int32_t next_token = 0;
  {
    auto model = Qwen2Model::LoadFromDirectory(dir);
    if (!model.ok()) GTEST_SKIP() << "model not loaded: " << model.status();
    ASSERT_EQ(model->tensor_parallel_rank(), 0);
    ASSERT_EQ(model->tensor_parallel_size(), 1);
    ASSERT_TRUE(model->ForwardLastLogits(prompt, &reference).ok());
    ASSERT_TRUE(model->AttachKvCache(/*num_blocks=*/4, /*block_size=*/16).ok());
    auto block = model->kv_pool()->AllocateBlock();
    ASSERT_TRUE(block.ok()) << block.status();
    ASSERT_TRUE(model->Step(Batch(prompt, *block, 0), &reference_prefill).ok());
    next_token = static_cast<int32_t>(
        std::max_element(reference_prefill.begin(), reference_prefill.end()) -
        reference_prefill.begin());
    ASSERT_TRUE(
        model->Step(Batch({next_token}, *block, prompt.size()),
                    &reference_decode)
            .ok());
    ASSERT_TRUE(model->kv_pool()->FreeBlock(*block).ok());
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
          prompt, &logits[rank]);
    });
  }
  for (auto& thread : threads) thread.join();

  for (int rank = 0; rank < kRanks; ++rank) {
    ASSERT_TRUE(statuses[rank].ok()) << "rank " << rank << ": " << statuses[rank];
    ExpectClose(logits[rank], reference, rank);
  }

  std::vector<int32_t> blocks(kRanks);
  std::vector<ForwardBatch> prefill_batches, decode_batches;
  for (int rank = 0; rank < kRanks; ++rank) {
    ASSERT_TRUE(models[rank]
                    .AttachKvCache(/*num_blocks=*/4, /*block_size=*/16)
                    .ok());
    EXPECT_EQ(models[rank].kv_pool()->bytes(),
              36ull * 4 * 2 * 16 * 1 * 128 * 2);
    auto block = models[rank].kv_pool()->AllocateBlock();
    ASSERT_TRUE(block.ok()) << block.status();
    blocks[rank] = *block;
    prefill_batches.push_back(Batch(prompt, *block, 0));
    decode_batches.push_back(Batch({next_token}, *block, prompt.size()));
    EXPECT_EQ(models[rank].CaptureDecodeGraph(1, 2).code(),
              absl::StatusCode::kUnimplemented);
  }

  const auto run_step = [&](const std::vector<ForwardBatch>& batches,
                            std::vector<std::vector<float>>* output) {
    std::vector<Status> step_status(kRanks);
    std::vector<std::thread> step_threads;
    for (int rank = 0; rank < kRanks; ++rank) {
      step_threads.emplace_back([&, rank] {
        step_status[rank] = models[rank].Step(batches[rank], &(*output)[rank]);
      });
    }
    for (auto& thread : step_threads) thread.join();
    for (int rank = 0; rank < kRanks; ++rank)
      EXPECT_TRUE(step_status[rank].ok()) << step_status[rank];
  };

  std::vector<std::vector<float>> prefill_logits(kRanks), decode_logits(kRanks);
  run_step(prefill_batches, &prefill_logits);
  run_step(decode_batches, &decode_logits);
  for (int rank = 0; rank < kRanks; ++rank) {
    ExpectClose(prefill_logits[rank], reference_prefill, rank);
    ExpectClose(decode_logits[rank], reference_decode, rank);
    EXPECT_TRUE(models[rank].kv_pool()->FreeBlock(blocks[rank]).ok());
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
