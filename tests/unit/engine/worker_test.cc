#include "inferx/engine/worker.h"

#include "gtest/gtest.h"
#include "inferx/dist/loopback_comm.h"

namespace inferx {
namespace {
TEST(WorkerTest, RejectsMismatchedCommunicatorBeforeLoading) {
  ParallelConfig parallel;
  parallel.tensor_parallel_size = 2;
  EXPECT_FALSE(Worker::Create(ModelConfig{}, parallel).ok());
  auto world = LoopbackWorld::Create(2);
  ASSERT_TRUE(world.ok());
  EXPECT_FALSE(Worker::Create(ModelConfig{}, parallel, (*world)->TakeRank(1)).ok());
}

TEST(WorkerTest, LifecycleRejectsOutOfOrderCalls) {
  auto worker = Worker::Create(ModelConfig{});
  ASSERT_TRUE(worker.ok());
  EXPECT_FALSE((*worker)->LoadModel().ok());
  EXPECT_FALSE((*worker)->DetermineAvailableMemory(CacheConfig{}, SchedulerConfig{}).ok());
  EXPECT_FALSE((*worker)->InitializeFromConfig(-1, CacheConfig{}, SchedulerConfig{}).ok());
  EXPECT_FALSE((*worker)->WarmUp(ExecutionConfig{}).ok());
  EXPECT_FALSE((*worker)->ExecuteModel(SchedulerOutput{}).ok());
}

TEST(WorkerTest, RejectsEmptyAndInvalidRankGroups) {
  auto abort = [](const Status&) {};
  EXPECT_FALSE(
      ModelRunner::CreateGroup({}, CacheConfig{}, SchedulerConfig{}, ExecutionConfig{}, abort)
          .ok());
  std::vector<std::unique_ptr<Worker>> ranks;
  ranks.push_back(nullptr);
  EXPECT_FALSE(ModelRunner::CreateGroup(std::move(ranks), CacheConfig{}, SchedulerConfig{},
                                        ExecutionConfig{}, abort)
                   .ok());
}
}  // namespace
}  // namespace inferx
