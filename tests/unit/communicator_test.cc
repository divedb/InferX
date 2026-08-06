#include "inferx/comm/communicator.h"

#include <gtest/gtest.h>

#include <bit>
#include <cstdint>
#include <thread>
#include <vector>

#include "inferx/comm/nccl_communicator.h"

#if defined(INFERX_WITH_CUDA)
#include <cuda_runtime_api.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#endif

namespace inferx::comm {
namespace {

uint16_t Bf16(float value);
float Float(uint16_t value);
TensorView CpuView(void* data, DataType dtype, int64_t count);

TEST(NcclCommTest, MissingBackendFailsClearly) {
  if (NcclBackendAvailable()) GTEST_SKIP() << "NCCL backend is installed";
  EXPECT_EQ(CreateNcclUniqueId().status().code(),
            absl::StatusCode::kUnimplemented);
  EXPECT_EQ(CreateNcclCommunicator({}).status().code(),
            absl::StatusCode::kUnimplemented);
}

#if defined(INFERX_WITH_CUDA)
TEST(NcclCommTest, TwoGpuBf16AllReduceUsesTheSuppliedStreams) {
  if (!NcclBackendAvailable() || CudaDeviceCount() < 2) {
    GTEST_SKIP() << "needs an NCCL build and two CUDA devices";
  }

  auto id = CreateNcclUniqueId();
  ASSERT_TRUE(id.ok()) << id.status();
  std::vector<Status> statuses(2);
  std::vector<uint16_t> outputs(2, 0);
  std::vector<std::thread> ranks;

  for (int rank = 0; rank < 2; ++rank) {
    ranks.emplace_back([&, rank] {
      NcclCommConfig config;
      config.rank = rank;
      config.world_size = 2;
      config.local_device = rank;
      config.unique_id = *id;
      auto communicator = CreateNcclCommunicator(config);
      if (!communicator.ok()) {
        statuses[static_cast<size_t>(rank)] = communicator.status();
        return;
      }

      cudaStream_t stream = nullptr;
      cudaError_t error =
          cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
      if (error != cudaSuccess) {
        statuses[static_cast<size_t>(rank)] = CudaErrorToStatus(
            error, "cudaStreamCreateWithFlags", __FILE__, __LINE__);
        return;
      }

      const DeviceId device = DeviceId::Cuda(static_cast<int8_t>(rank));
      auto buffer = DeviceBuffer::Allocate(sizeof(uint16_t), device);
      if (!buffer.ok()) {
        statuses[static_cast<size_t>(rank)] = buffer.status();
        cudaStreamDestroy(stream);
        return;
      }
      const uint16_t input = Bf16(static_cast<float>(rank + 1));
      error = cudaMemcpyAsync(buffer->data(), &input, sizeof(input),
                              cudaMemcpyHostToDevice, stream);
      if (error == cudaSuccess) {
        auto view = TensorView::Create(buffer->data(), DataType::kBFloat16,
                                       Shape({1}), device);
        statuses[static_cast<size_t>(rank)] =
            view.ok() ? (*communicator)->AllReduceSum(*view, stream)
                      : view.status();
      } else {
        statuses[static_cast<size_t>(rank)] =
            CudaErrorToStatus(error, "cudaMemcpyAsync", __FILE__, __LINE__);
      }
      if (statuses[static_cast<size_t>(rank)].ok()) {
        error =
            cudaMemcpyAsync(&outputs[static_cast<size_t>(rank)], buffer->data(),
                            sizeof(uint16_t), cudaMemcpyDeviceToHost, stream);
        if (error == cudaSuccess) error = cudaStreamSynchronize(stream);
        if (error != cudaSuccess) {
          statuses[static_cast<size_t>(rank)] = CudaErrorToStatus(
              error, "NCCL test synchronization", __FILE__, __LINE__);
        }
      }
      cudaStreamDestroy(stream);
    });
  }
  for (auto& rank : ranks) rank.join();

  for (int rank = 0; rank < 2; ++rank) {
    ASSERT_TRUE(statuses[static_cast<size_t>(rank)].ok())
        << "rank " << rank << ": " << statuses[static_cast<size_t>(rank)];
    EXPECT_FLOAT_EQ(Float(outputs[static_cast<size_t>(rank)]), 3.0f);
  }
}

TEST(CommunicatorMetricsTest, TimingSkipsCudaGraphCapture) {
  if (CudaDeviceCount() < 1) GTEST_SKIP() << "needs a CUDA device";
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  cudaStream_t stream = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            cudaSuccess);
  auto metrics = std::make_shared<CommMetrics>();
  std::unique_ptr<Communicator> communicator = ObserveCommunicator(
      std::make_unique<SingleRankComm>(DeviceId::Cuda(0)), metrics,
      {.timing_sample_every = 1, .timing_ring_size = 2});
  auto buffer = DeviceBuffer::Allocate(4, DeviceId::Cuda(0));
  ASSERT_TRUE(buffer.ok()) << buffer.status();
  std::vector<float> values = {1.0f};

  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(buffer->data(), 0, buffer->size(), stream),
            cudaSuccess);
  EXPECT_TRUE(
      communicator
          ->AllReduceSum(
              CpuView(values.data(), DataType::kFloat, values.size()), stream)
          .ok());
  cudaGraph_t graph = nullptr;
  ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);

  EXPECT_EQ(metrics->Snapshot().timing_graph_skips, 1);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  communicator.reset();
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
#endif

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
  EXPECT_TRUE(
      comm.AllReduceSum(CpuView(values.data(), DataType::kFloat, values.size()))
          .ok());
  EXPECT_EQ(values, (std::vector<float>{1.0f, -2.0f, 3.5f}));
  EXPECT_FALSE(comm.AllReduceSum(TensorView()).ok());
}

TEST(CommunicatorMetricsTest, DecoratorCountsCallsBytesFailuresAndAborts) {
  auto metrics = std::make_shared<CommMetrics>();
  std::unique_ptr<Communicator> comm = ObserveCommunicator(
      std::make_unique<SingleRankComm>(DeviceId::Cpu()), metrics);
  std::vector<float> values = {1.0f, 2.0f, 3.0f};

  EXPECT_TRUE(comm->AllReduceSum(
                      CpuView(values.data(), DataType::kFloat, values.size()))
                  .ok());
  EXPECT_FALSE(comm->AllReduceSum(TensorView()).ok());
  EXPECT_TRUE(comm->Abort().ok());

  const CommMetricSnapshot snapshot = metrics->Snapshot();
  EXPECT_EQ(snapshot.all_reduce_calls, 2);
  EXPECT_EQ(snapshot.all_reduce_bytes, values.size() * sizeof(float));
  EXPECT_EQ(snapshot.collective_failures, 1);
  EXPECT_EQ(snapshot.aborts, 1);
  EXPECT_EQ(CommBackendName(comm->backend()), std::string_view("single"));
}

TEST(CommunicatorMetricsTest, LatencySnapshotUsesFixedNonCumulativeBuckets) {
  CommMetrics metrics;
  metrics.RecordLatency(0.00002);
  metrics.RecordLatency(0.00007);
  metrics.RecordLatency(0.2);
  metrics.RecordTimingDrop();
  metrics.RecordGraphSkip();

  const CommMetricSnapshot snapshot = metrics.Snapshot();
  EXPECT_EQ(snapshot.latency_count, 3);
  EXPECT_DOUBLE_EQ(snapshot.latency_sum_seconds, 0.20009);
  EXPECT_EQ(snapshot.latency_buckets[1], 1);
  EXPECT_EQ(snapshot.latency_buckets[3], 1);
  EXPECT_EQ(snapshot.timing_samples_dropped, 1);
  EXPECT_EQ(snapshot.timing_graph_skips, 1);
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
                        ->AllReduceSum(
                            CpuView(values[rank].data(), DataType::kInt32, 2))
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
  std::vector<std::vector<uint16_t>> values = {{Bf16(1.5f), Bf16(-2.0f)},
                                               {Bf16(0.25f), Bf16(5.0f)}};
  std::thread rank0([&] {
    EXPECT_TRUE(
        comms[0]
            ->AllReduceSum(CpuView(values[0].data(), DataType::kBFloat16, 2))
            .ok());
  });
  std::thread rank1([&] {
    EXPECT_TRUE(
        comms[1]
            ->AllReduceSum(CpuView(values[1].data(), DataType::kBFloat16, 2))
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
