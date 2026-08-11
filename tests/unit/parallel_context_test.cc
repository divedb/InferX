#include <memory>

#include <gtest/gtest.h>

#include "inferx/comm/communicator.h"
#include "inferx/engine/parallel_context.h"

namespace inferx::engine {
namespace {

TEST(ParallelContextTest, ExposesPlacementAndTensorParallelAxis) {
  const DeviceId device = DeviceId::Cpu();
  ParallelContext context(
      device, 1, 2, std::make_unique<comm::SingleRankComm>(device));

  EXPECT_EQ(context.device(), device);
  EXPECT_EQ(context.tp_rank(), 1);
  EXPECT_EQ(context.tp_size(), 2);
  EXPECT_EQ(context.tp_comm().device(), device);
}

TEST(ParallelContextTest, TransfersCommunicatorToRankModel) {
  const DeviceId device = DeviceId::Cpu();
  ParallelContext context(
      device, 0, 1, std::make_unique<comm::SingleRankComm>(device));

  std::unique_ptr<comm::Communicator> communicator =
      context.TakeTpCommunicator();
  ASSERT_NE(communicator, nullptr);
  EXPECT_EQ(communicator->rank(), 0);
  EXPECT_EQ(communicator->size(), 1);
  EXPECT_EQ(communicator->device(), device);
}

}  // namespace
}  // namespace inferx::engine
