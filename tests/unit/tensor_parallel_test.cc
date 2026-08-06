#include "inferx/comm/tensor_parallel.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace inferx::comm {
namespace {

Tensor Matrix(int64_t rows, int64_t cols) {
  auto tensor = Tensor::Empty(DataType::kInt32, Shape({rows, cols}),
                              DeviceId::Cpu());
  EXPECT_TRUE(tensor.ok()) << tensor.status();
  for (int64_t i = 0; i < rows * cols; ++i)
    tensor->data_as<int32_t>()[i] = static_cast<int32_t>(i);
  return *tensor;
}

void ExpectSame(const Tensor& lhs, const Tensor& rhs) {
  ASSERT_EQ(lhs.dtype(), rhs.dtype());
  ASSERT_EQ(lhs.shape(), rhs.shape());
  ASSERT_EQ(lhs.nbytes(), rhs.nbytes());
  EXPECT_EQ(std::memcmp(lhs.data(), rhs.data(), lhs.nbytes()), 0);
}

TEST(TensorParallelShardTest, ColumnParallelSplitsOutputRows) {
  const Tensor weight = Matrix(8, 6);
  std::vector<Tensor> shards;
  for (int rank = 0; rank < 4; ++rank) {
    auto shard = ShardHostTensor(weight, /*axis=*/0, rank, 4);
    ASSERT_TRUE(shard.ok()) << shard.status();
    EXPECT_EQ(shard->shape(), Shape({2, 6}));
    EXPECT_EQ(shard->data_as<int32_t>()[0], rank * 12);
    shards.push_back(std::move(*shard));
  }
  auto reconstructed = ReconstructHostTensor(shards, /*axis=*/0);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  ExpectSame(*reconstructed, weight);
}

TEST(TensorParallelShardTest, RowParallelSplitsInputColumns) {
  const Tensor weight = Matrix(3, 8);
  std::vector<Tensor> shards;
  for (int rank = 0; rank < 4; ++rank) {
    auto shard = ShardHostTensor(weight, /*axis=*/1, rank, 4);
    ASSERT_TRUE(shard.ok()) << shard.status();
    EXPECT_EQ(shard->shape(), Shape({3, 2}));
    EXPECT_EQ(shard->data_as<int32_t>()[0], rank * 2);
    EXPECT_EQ(shard->data_as<int32_t>()[2], 8 + rank * 2);
    shards.push_back(std::move(*shard));
  }
  auto reconstructed = ReconstructHostTensor(shards, /*axis=*/-1);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  ExpectSame(*reconstructed, weight);
}

TEST(TensorParallelShardTest, WorksAcrossOuterDimensions) {
  auto tensor = Tensor::Empty(DataType::kUInt8, Shape({2, 4, 3}),
                              DeviceId::Cpu());
  ASSERT_TRUE(tensor.ok()) << tensor.status();
  for (int i = 0; i < 24; ++i)
    tensor->data_as<uint8_t>()[i] = static_cast<uint8_t>(i);
  std::vector<Tensor> shards;
  for (int rank = 0; rank < 2; ++rank) {
    auto shard = ShardHostTensor(*tensor, /*axis=*/1, rank, 2);
    ASSERT_TRUE(shard.ok()) << shard.status();
    EXPECT_EQ(shard->shape(), Shape({2, 2, 3}));
    shards.push_back(std::move(*shard));
  }
  auto reconstructed = ReconstructHostTensor(shards, 1);
  ASSERT_TRUE(reconstructed.ok()) << reconstructed.status();
  ExpectSame(*reconstructed, *tensor);
}

TEST(TensorParallelShardTest, RejectsInvalidTopologyAndShape) {
  const Tensor weight = Matrix(7, 5);
  EXPECT_FALSE(ShardHostTensor(weight, 0, 0, 4).ok());
  EXPECT_FALSE(ShardHostTensor(weight, 1, 0, 2).ok());
  EXPECT_FALSE(ShardHostTensor(weight, 2, 0, 1).ok());
  EXPECT_FALSE(ShardHostTensor(weight, 0, 2, 2).ok());
  EXPECT_FALSE(ShardHostTensor(weight, 0, 0, 0).ok());
}

TEST(TensorParallelShardTest, RejectsMismatchedReconstruction) {
  std::vector<Tensor> shards = {Matrix(2, 4), Matrix(3, 4)};
  EXPECT_FALSE(ReconstructHostTensor(shards, 0).ok());
  EXPECT_FALSE(ReconstructHostTensor({}, 0).ok());
}

TEST(TensorParallelShardTest, RejectsPackedInt4UntilFormatAwareSharding) {
  auto packed = Tensor::Empty(DataType::kInt4, Shape({4, 8}), DeviceId::Cpu());
  ASSERT_TRUE(packed.ok()) << packed.status();
  const Status status = ShardHostTensor(*packed, 0, 0, 2).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace inferx::comm
