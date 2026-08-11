#include <array>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/comm/communicator.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/model/parallel/linear.h"

namespace inferx::model::parallel {
namespace {

TEST(RowParallelLinearTest, SingleRankCompletionPreservesHostOutput) {
  std::array<float, 4> values = {1.0f, 2.0f, 3.0f, 4.0f};
  const auto output = TensorView::Create(values.data(), DataType::kFloat,
                                         Shape({2, 2}), DeviceId::Cpu());
  ASSERT_TRUE(output.ok()) << output.status();
  comm::SingleRankComm communicator(DeviceId::Cpu());

  EXPECT_TRUE(RowParallelLinear::ReduceOutput(communicator, *output).ok());
  EXPECT_EQ(values, (std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(RowParallelLinearTest, CompletionSumsPartialOutputsAcrossRanks) {
  auto communicators = comm::CreateHostSimCommunicators(2, DeviceId::Cpu());
  ASSERT_TRUE(communicators.ok()) << communicators.status();
  std::array<std::array<float, 2>, 2> values = {
      std::array<float, 2>{1.0f, 2.0f}, std::array<float, 2>{3.0f, 4.0f}};
  std::array<Status, 2> statuses;
  std::vector<std::thread> ranks;
  for (int rank = 0; rank < 2; ++rank) {
    ranks.emplace_back([&, rank] {
      const auto output = TensorView::Create(
          values[rank].data(), DataType::kFloat, Shape({2}), DeviceId::Cpu());
      if (!output.ok()) {
        statuses[rank] = output.status();
        return;
      }
      statuses[rank] =
          RowParallelLinear::ReduceOutput(*(*communicators)[rank], *output);
    });
  }
  for (std::thread& rank : ranks) rank.join();

  ASSERT_TRUE(statuses[0].ok()) << statuses[0];
  ASSERT_TRUE(statuses[1].ok()) << statuses[1];
  EXPECT_EQ(values[0], (std::array<float, 2>{4.0f, 6.0f}));
  EXPECT_EQ(values[1], (std::array<float, 2>{4.0f, 6.0f}));
}

}  // namespace
}  // namespace inferx::model::parallel
