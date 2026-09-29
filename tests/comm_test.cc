#include "inferx/dist/comm.h"

#include <future>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/dist/loopback_comm.h"
#include "inferx/dist/nccl_comm.h"

namespace inferx::dist {
namespace {

class CommTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(DeviceId::Cpu());
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok()) << stream.status();
    stream_ = *stream;
  }
  void TearDown() override {
    if (runtime_ != nullptr) {
      EXPECT_TRUE(runtime_->SynchronizeStream(stream_).ok());
      EXPECT_TRUE(runtime_->DestroyStream(stream_).ok());
    }
  }
  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

TEST_F(CommTest, SingleRankReductionAcceptsAllTensorRanks) {
  SingleRankComm comm;
  const ops::ExecutionContext ctx(*runtime_, stream_);
  for (const Shape shape : {Shape{}, Shape{3}, Shape{1, 2, 3}, Shape{0}}) {
    auto tensor = Tensor::Empty(DataType::kBFloat16, shape, DeviceId::Cpu());
    ASSERT_TRUE(tensor.ok()) << tensor.status();
    EXPECT_TRUE(comm.AllReduceSum(ctx, *tensor).ok());
  }
}

TEST_F(CommTest, GatherValidatesShapeDtypeDeviceAndOverlap) {
  SingleRankComm comm;
  const ops::ExecutionContext ctx(*runtime_, stream_);
  auto input = Tensor::Empty(DataType::kBFloat16, Shape{2, 3}, DeviceId::Cpu());
  auto wrong_shape = Tensor::Empty(DataType::kBFloat16, Shape{3, 2}, DeviceId::Cpu());
  auto wrong_dtype = Tensor::Empty(DataType::kFloat32, Shape{2, 3}, DeviceId::Cpu());
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(wrong_shape.ok());
  ASSERT_TRUE(wrong_dtype.ok());
  EXPECT_EQ(comm.AllGatherLastDim(ctx, *input, *wrong_shape).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(comm.AllGatherLastDim(ctx, *input, *wrong_dtype).code(),
            absl::StatusCode::kInvalidArgument);

  // These views exercise metadata validation without accessing a GPU.
  auto wrong_device =
      Tensor::FromBlob(input->Data(), DataType::kBFloat16, Shape{2, 3}, DeviceId::Cuda(0));
  ASSERT_TRUE(wrong_device.ok());
  EXPECT_EQ(comm.AllGatherLastDim(ctx, *input, *wrong_device).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(comm.AllReduceSum(ctx, *wrong_device).code(), absl::StatusCode::kInvalidArgument);

  std::vector<uint16_t> storage(8);
  auto a = Tensor::FromBlob(storage.data(), DataType::kBFloat16, Shape{2, 3}, DeviceId::Cpu());
  auto b =
      Tensor::FromBlob(storage.data() + 1, DataType::kBFloat16, Shape{2, 3}, DeviceId::Cpu());
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  EXPECT_EQ(comm.AllGatherLastDim(ctx, *a, *b).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(comm.AllGatherLastDim(ctx, *b, *a).code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CommTest, LoopbackPropagatesValidationFailureToPeers) {
  auto world = LoopbackWorld::Create(2);
  ASSERT_TRUE(world.ok());
  auto valid = Tensor::Empty(DataType::kBFloat16, Shape{2, 3}, DeviceId::Cpu());
  auto invalid = Tensor::Empty(DataType::kFloat32, Shape{2, 3}, DeviceId::Cpu());
  ASSERT_TRUE(valid.ok());
  ASSERT_TRUE(invalid.ok());
  const ops::ExecutionContext ctx(*runtime_, stream_);
  auto peer = std::async(std::launch::async,
                         [&] { return (*world)->rank(0).AllReduceSum(ctx, *valid); });
  const Status failure = (*world)->rank(1).AllReduceSum(ctx, *invalid);
  EXPECT_EQ(failure.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(peer.get(), failure);
  // A failed world cannot be reused, even after fixing the local input.
  EXPECT_EQ((*world)->rank(1).AllReduceSum(ctx, *valid), failure);
}

TEST_F(CommTest, LoopbackRejectsMismatchedPeerShapes) {
  auto world = LoopbackWorld::Create(2);
  ASSERT_TRUE(world.ok());
  auto a = Tensor::Empty(DataType::kBFloat16, Shape{2, 3}, DeviceId::Cpu());
  auto b = Tensor::Empty(DataType::kBFloat16, Shape{2, 4}, DeviceId::Cpu());
  ASSERT_TRUE(a.ok());
  ASSERT_TRUE(b.ok());
  const ops::ExecutionContext ctx(*runtime_, stream_);
  auto peer =
      std::async(std::launch::async, [&] { return (*world)->rank(0).AllReduceSum(ctx, *a); });
  EXPECT_EQ((*world)->rank(1).AllReduceSum(ctx, *b).code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(peer.get().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CommTest, LoopbackEmptyCollectivesPreserveSequence) {
  auto world = LoopbackWorld::Create(2);
  ASSERT_TRUE(world.ok());
  auto run = [&](int rank) -> Status {
    const ops::ExecutionContext ctx(*runtime_, stream_);
    INFERX_ASSIGN_OR_RETURN(auto partial,
                            Tensor::Empty(DataType::kBFloat16, Shape{0, 3}, DeviceId::Cpu()));
    INFERX_ASSIGN_OR_RETURN(auto full,
                            Tensor::Empty(DataType::kBFloat16, Shape{0, 6}, DeviceId::Cpu()));
    for (int i = 0; i < 3; ++i) {
      INFERX_RETURN_IF_ERROR((*world)->rank(rank).AllReduceSum(ctx, partial));
      INFERX_RETURN_IF_ERROR((*world)->rank(rank).AllGatherLastDim(ctx, partial, full));
    }
    return OkStatus();
  };
  auto peer = std::async(std::launch::async, run, 0);
  EXPECT_TRUE(run(1).ok());
  EXPECT_TRUE(peer.get().ok());
}

class CommCudaTest : public CommTest {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(DeviceId::Cuda(0));
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok()) << stream.status();
    stream_ = *stream;
  }
};

TEST_F(CommCudaTest, SingleRankGatherOrdersCopiesOnTheContextStream) {
  const ops::ExecutionContext ctx(*runtime_, stream_);
  SingleRankComm comm;
  auto input = Tensor::Empty(DataType::kBFloat16, Shape{2, 2}, ctx.device());
  auto output = Tensor::Empty(DataType::kBFloat16, Shape{2, 2}, ctx.device());
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(output.ok());
  std::vector<uint16_t> values{0x3f80, 0x4000, 0x4040, 0x4080}, result(4);
  ASSERT_TRUE(runtime_
                  ->CopyAsync(input->Data(), values.data(), input->NBytes(),
                              CopyKind::kHostToDevice, stream_)
                  .ok());
  ASSERT_TRUE(comm.AllGatherLastDim(ctx, *input, *output).ok());
  ASSERT_TRUE(runtime_
                  ->CopyAsync(result.data(), output->Data(), output->NBytes(),
                              CopyKind::kDeviceToHost, stream_)
                  .ok());
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  EXPECT_EQ(result, values);
}

TEST_F(CommCudaTest, LoopbackReducesAllShapesAndGathersInRankOrder) {
  auto world = LoopbackWorld::Create(3);
  ASSERT_TRUE(world.ok());
  auto run = [&](int rank) -> Status {
    INFERX_ASSIGN_OR_RETURN(auto stream, runtime_->CreateStream());
    const ops::ExecutionContext ctx(*runtime_, stream);
    Status status = [&]() -> Status {
      auto& comm = (*world)->rank(rank);
      // Repeated leading dimensions with different tails exercise slot reuse.
      for (const Shape shape : {Shape{}, Shape{5}, Shape{1, 2, 3}, Shape{1, 2, 4}}) {
        INFERX_ASSIGN_OR_RETURN(auto input,
                                Tensor::Empty(DataType::kBFloat16, shape, ctx.device()));
        const uint16_t value = rank == 0 ? 0x3f80 : rank == 1 ? 0x4000 : 0x4040;
        std::vector<uint16_t> values(input.Numel(), value);
        INFERX_RETURN_IF_ERROR(runtime_->CopyAsync(input.Data(), values.data(), input.NBytes(),
                                                   CopyKind::kHostToDevice, stream));
        INFERX_RETURN_IF_ERROR(comm.AllReduceSum(ctx, input));
        INFERX_RETURN_IF_ERROR(runtime_->CopyAsync(values.data(), input.Data(), input.NBytes(),
                                                   CopyKind::kDeviceToHost, stream));
        INFERX_RETURN_IF_ERROR(runtime_->SynchronizeStream(stream));
        EXPECT_EQ(values, std::vector<uint16_t>(input.Numel(), 0x40c0));  // 1 + 2 + 3 = 6.
      }
      INFERX_ASSIGN_OR_RETURN(auto input,
                              Tensor::Empty(DataType::kBFloat16, Shape{2, 2}, ctx.device()));
      INFERX_ASSIGN_OR_RETURN(auto full,
                              Tensor::Empty(DataType::kBFloat16, Shape{2, 6}, ctx.device()));
      const uint16_t value = rank == 0 ? 0x3f80 : rank == 1 ? 0x4000 : 0x4040;
      std::vector<uint16_t> values(4, value), result(12);
      for (int i = 0; i < 3; ++i) {
        INFERX_RETURN_IF_ERROR(runtime_->CopyAsync(input.Data(), values.data(), input.NBytes(),
                                                   CopyKind::kHostToDevice, stream));
        INFERX_RETURN_IF_ERROR(comm.AllGatherLastDim(ctx, input, full));
        INFERX_RETURN_IF_ERROR(runtime_->CopyAsync(result.data(), full.Data(), full.NBytes(),
                                                   CopyKind::kDeviceToHost, stream));
        INFERX_RETURN_IF_ERROR(runtime_->SynchronizeStream(stream));
        EXPECT_EQ(result,
                  (std::vector<uint16_t>{0x3f80, 0x3f80, 0x4000, 0x4000, 0x4040, 0x4040, 0x3f80,
                                         0x3f80, 0x4000, 0x4000, 0x4040, 0x4040}));
      }
      return OkStatus();
    }();
    (void)runtime_->SynchronizeStream(stream);
    (void)runtime_->DestroyStream(stream);
    return status;
  };
  auto rank0 = std::async(std::launch::async, run, 0);
  auto rank1 = std::async(std::launch::async, run, 1);
  auto rank2 = std::async(std::launch::async, run, 2);
  EXPECT_TRUE(rank0.get().ok());
  EXPECT_TRUE(rank1.get().ok());
  EXPECT_TRUE(rank2.get().ok());
}

TEST_F(CommCudaTest, NcclSingleRankUsesBfloat16AndCopiesGatherOutput) {
  auto id = NcclComm::NewUniqueId();
  if (absl::IsNotFound(id.status())) GTEST_SKIP() << "NCCL is not installed";
  ASSERT_TRUE(id.ok()) << id.status();
  auto comm = NcclComm::Join(0, 1, *id, runtime_->device());
  ASSERT_TRUE(comm.ok()) << comm.status();
  const ops::ExecutionContext ctx(*runtime_, stream_);
  auto input = Tensor::Empty(DataType::kBFloat16, Shape{2, 3}, ctx.device());
  auto output = Tensor::Empty(DataType::kBFloat16, Shape{2, 3}, ctx.device());
  ASSERT_TRUE(input.ok());
  ASSERT_TRUE(output.ok());
  std::vector<uint16_t> values{0x3f80, 0x4000, 0x4040, 0xc000, 0x0000, 0x3f80}, result(6);
  ASSERT_TRUE(runtime_
                  ->CopyAsync(input->Data(), values.data(), input->NBytes(),
                              CopyKind::kHostToDevice, stream_)
                  .ok());
  ASSERT_TRUE((*comm)->AllReduceSum(ctx, *input).ok());
  ASSERT_TRUE((*comm)->AllGatherLastDim(ctx, *input, *output).ok());
  ASSERT_TRUE(runtime_
                  ->CopyAsync(result.data(), output->Data(), output->NBytes(),
                              CopyKind::kDeviceToHost, stream_)
                  .ok());
  ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
  EXPECT_EQ(result, values);
}

}  // namespace
}  // namespace inferx::dist
