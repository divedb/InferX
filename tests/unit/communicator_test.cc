#include "inferx/comm/communicator.h"
#include "inferx/comm/nccl_communicator.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <thread>
#include <vector>

#if defined(INFERX_WITH_CUDA)
#include <cuda_runtime_api.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#endif

namespace inferx::comm {
namespace {

TEST(NcclCommTest, MissingBackendFailsClearly) {
  if (NcclBackendAvailable()) GTEST_SKIP() << "NCCL backend is installed";
  EXPECT_EQ(CreateNcclUniqueId().status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(CreateNcclCommunicator({}).status().code(),
            absl::StatusCode::kUnimplemented);
}

TensorView CpuView(void* data, DataType dtype, int64_t count) {
  auto view = TensorView::Create(data, dtype, Shape({count}), DeviceId::Cpu());
  EXPECT_TRUE(view.ok()) << view.status();
  return *view;
}

uint16_t Bf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}

float Float(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

TEST(SingleRankCommTest, IsAValidatedNoOp) {
  SingleRankComm comm;
  EXPECT_EQ(comm.backend(), CommBackend::kSingleRank);
  EXPECT_EQ(comm.device(), DeviceId::Cuda(0));
  EXPECT_TRUE(comm.capabilities().cuda_graph_capture);
  EXPECT_TRUE(comm.capabilities().device_collectives);
  std::vector<float> values = {1.0f, -2.0f, 3.5f};
  EXPECT_EQ(comm.rank(), 0);
  EXPECT_EQ(comm.size(), 1);
  EXPECT_TRUE(comm.AllReduceSum(
                      CpuView(values.data(), DataType::kFloat, values.size()))
                  .ok());
  EXPECT_EQ(values, (std::vector<float>{1.0f, -2.0f, 3.5f}));
  EXPECT_FALSE(comm.AllReduceSum(TensorView()).ok());
}

TEST(HostSimCommTest, FourRanksAllReceiveTheSum) {
  auto created = CreateHostSimCommunicators(4);
  ASSERT_TRUE(created.ok()) << created.status();
  auto comms = std::move(*created);
  std::vector<std::vector<float>> values(4, std::vector<float>(3));
  std::vector<Status> statuses(4);
  std::vector<std::thread> threads;
  for (int rank = 0; rank < 4; ++rank) {
    values[rank] = {static_cast<float>(rank + 1), 2.0f * rank, -1.0f};
    threads.emplace_back([&, rank] {
      statuses[rank] = comms[rank]->AllReduceSum(
          CpuView(values[rank].data(), DataType::kFloat, 3));
    });
  }
  for (auto& thread : threads) thread.join();
  for (int rank = 0; rank < 4; ++rank) {
    EXPECT_TRUE(statuses[rank].ok()) << statuses[rank];
    EXPECT_EQ(values[rank], (std::vector<float>{10.0f, 12.0f, -4.0f}));
  }
}

TEST(HostSimCommTest, ReusesTheRendezvousAcrossCollectives) {
  auto created = CreateHostSimCommunicators(3);
  ASSERT_TRUE(created.ok()) << created.status();
  auto comms = std::move(*created);
  std::vector<std::vector<int32_t>> values(3, std::vector<int32_t>(2));
  std::vector<std::thread> threads;
  for (int rank = 0; rank < 3; ++rank) {
    threads.emplace_back([&, rank] {
      for (int round = 0; round < 20; ++round) {
        values[rank] = {rank + round, 1};
        ASSERT_TRUE(comms[rank]
                        ->AllReduceSum(CpuView(values[rank].data(),
                                              DataType::kInt32, 2))
                        .ok());
        EXPECT_EQ(values[rank][0], 3 * round + 3);
        EXPECT_EQ(values[rank][1], 3);
      }
    });
  }
  for (auto& thread : threads) thread.join();
}

TEST(HostSimCommTest, Bf16AccumulatesInFloatInRankOrder) {
  auto created = CreateHostSimCommunicators(2);
  ASSERT_TRUE(created.ok()) << created.status();
  auto comms = std::move(*created);
  std::vector<std::vector<uint16_t>> values = {
      {Bf16(1.5f), Bf16(-2.0f)}, {Bf16(0.25f), Bf16(5.0f)}};
  std::thread rank0([&] {
    EXPECT_TRUE(comms[0]
                    ->AllReduceSum(CpuView(values[0].data(),
                                          DataType::kBFloat16, 2))
                    .ok());
  });
  std::thread rank1([&] {
    EXPECT_TRUE(comms[1]
                    ->AllReduceSum(CpuView(values[1].data(),
                                          DataType::kBFloat16, 2))
                    .ok());
  });
  rank0.join();
  rank1.join();
  for (const auto& rank : values) {
    EXPECT_EQ(Float(rank[0]), 1.75f);
    EXPECT_EQ(Float(rank[1]), 3.0f);
  }
}

TEST(HostSimCommTest, ReportsCollectiveShapeMismatchToEveryRank) {
  auto created = CreateHostSimCommunicators(2);
  ASSERT_TRUE(created.ok()) << created.status();
  auto comms = std::move(*created);
  std::vector<float> a(2, 1.0f), b(3, 2.0f);
  Status sa, sb;
  std::thread rank0([&] {
    sa = comms[0]->AllReduceSum(CpuView(a.data(), DataType::kFloat, 2));
  });
  std::thread rank1([&] {
    sb = comms[1]->AllReduceSum(CpuView(b.data(), DataType::kFloat, 3));
  });
  rank0.join();
  rank1.join();
  EXPECT_EQ(sa.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(sb.code(), absl::StatusCode::kInvalidArgument);
}

TEST(HostSimCommTest, RejectsInvalidWorldSize) {
  EXPECT_FALSE(CreateHostSimCommunicators(0).ok());
  EXPECT_FALSE(CreateHostSimCommunicators(-2).ok());
}

#if defined(INFERX_WITH_CUDA)
TEST(HostSimCommTest, StagesCudaTensorsForDifferentialTesting) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
  auto created = CreateHostSimCommunicators(2);
  ASSERT_TRUE(created.ok()) << created.status();
  auto comms = std::move(*created);
  std::vector<std::vector<float>> host = {{1.0f, 2.0f}, {3.0f, -1.0f}};
  std::vector<DeviceBuffer> buffers;
  std::vector<cudaStream_t> streams(2, nullptr);
  for (int rank = 0; rank < 2; ++rank) {
    auto buffer = DeviceBuffer::Allocate(2 * sizeof(float), DeviceId::Cuda(0));
    ASSERT_TRUE(buffer.ok()) << buffer.status();
    ASSERT_EQ(cudaMemcpy(buffer->data(), host[rank].data(), buffer->size(),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    buffers.push_back(std::move(*buffer));
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking),
              cudaSuccess);
  }
  std::vector<Status> statuses(2);
  std::vector<std::thread> threads;
  for (int rank = 0; rank < 2; ++rank) {
    threads.emplace_back([&, rank] {
      auto view = TensorView::Create(buffers[rank].data(), DataType::kFloat,
                                     Shape({2}), DeviceId::Cuda(0));
      ASSERT_TRUE(view.ok()) << view.status();
      statuses[rank] =
          comms[rank]->AllReduceSum(*view, static_cast<void*>(streams[rank]));
    });
  }
  for (auto& thread : threads) thread.join();
  for (int rank = 0; rank < 2; ++rank) {
    EXPECT_TRUE(statuses[rank].ok()) << statuses[rank];
    ASSERT_EQ(cudaMemcpy(host[rank].data(), buffers[rank].data(),
                         buffers[rank].size(), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_EQ(host[rank], (std::vector<float>{4.0f, 1.0f}));
    EXPECT_EQ(cudaStreamDestroy(streams[rank]), cudaSuccess);
  }
}
#endif

}  // namespace
}  // namespace inferx::comm
