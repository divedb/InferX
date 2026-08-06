#include "inferx/comm/tensor_parallel.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "inferx/core/dtype.h"

namespace inferx::comm {
namespace {

StatusOr<int> NormalizeAxis(int axis, int rank) {
  if (axis < 0) axis += rank;
  if (axis < 0 || axis >= rank)
    return OutOfRangeError("tensor-parallel axis ", axis,
                           " is outside rank ", rank);
  return axis;
}

Status ValidateHostTensor(const Tensor& tensor) {
  if (!tensor.defined())
    return InvalidArgumentError("tensor-parallel shard is undefined");
  if (!tensor.is_cpu())
    return InvalidArgumentError("tensor-parallel sharding requires CPU weights");
  if (DataTypeBitWidth(tensor.dtype()) % 8 != 0)
    return UnimplementedError("tensor-parallel sharding does not support packed ",
                              DataTypeName(tensor.dtype()));
  return OkStatus();
}

int64_t Product(const Shape& shape, int begin, int end) {
  int64_t value = 1;
  for (int i = begin; i < end; ++i) value *= shape.Dim(i);
  return value;
}

}  // namespace

StatusOr<Tensor> ShardHostTensor(const Tensor& tensor, int axis, int rank,
                                 int world_size) {
  INFERX_RETURN_IF_ERROR(ValidateHostTensor(tensor));
  if (world_size <= 0)
    return InvalidArgumentError("world_size must be positive, got ", world_size);
  if (rank < 0 || rank >= world_size)
    return InvalidArgumentError("rank ", rank, " is outside world_size ",
                                world_size);
  INFERX_ASSIGN_OR_RETURN(axis, NormalizeAxis(axis, tensor.rank()));

  const Shape input_shape = tensor.shape();
  const int64_t axis_size = input_shape.Dim(axis);
  if (axis_size % world_size != 0)
    return InvalidArgumentError("dimension ", axis_size, " on axis ", axis,
                                " is not divisible by world_size ", world_size);

  Shape shard_shape = input_shape;
  const int64_t shard_axis = axis_size / world_size;
  shard_shape.SetDim(axis, shard_axis);
  INFERX_ASSIGN_OR_RETURN(
      Tensor shard,
      Tensor::Empty(tensor.dtype(), shard_shape, DeviceId::Cpu()));

  const int64_t outer = Product(input_shape, 0, axis);
  const int64_t inner = Product(input_shape, axis + 1, tensor.rank());
  const int64_t element_bytes = DataTypeBitWidth(tensor.dtype()) / 8;
  const int64_t copy_bytes = shard_axis * inner * element_bytes;
  const auto* src = static_cast<const std::byte*>(tensor.data());
  auto* dst = static_cast<std::byte*>(shard.data());
  for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
    const int64_t src_element =
        (outer_index * axis_size + rank * shard_axis) * inner;
    if (copy_bytes != 0)
      std::memcpy(dst + outer_index * copy_bytes,
                  src + src_element * element_bytes, copy_bytes);
  }
  return shard;
}

StatusOr<Tensor> ReconstructHostTensor(const std::vector<Tensor>& shards,
                                       int axis) {
  if (shards.empty()) return InvalidArgumentError("cannot reconstruct no shards");
  INFERX_RETURN_IF_ERROR(ValidateHostTensor(shards.front()));
  INFERX_ASSIGN_OR_RETURN(axis, NormalizeAxis(axis, shards.front().rank()));

  const DataType dtype = shards.front().dtype();
  const Shape shard_shape = shards.front().shape();
  for (size_t rank = 1; rank < shards.size(); ++rank) {
    INFERX_RETURN_IF_ERROR(ValidateHostTensor(shards[rank]));
    if (shards[rank].dtype() != dtype || shards[rank].shape() != shard_shape)
      return InvalidArgumentError("tensor-parallel shard ", rank,
                                  " has a mismatched dtype or shape");
  }

  Shape output_shape = shard_shape;
  output_shape.SetDim(axis,
                      shard_shape.Dim(axis) * static_cast<int64_t>(shards.size()));
  INFERX_ASSIGN_OR_RETURN(
      Tensor output, Tensor::Empty(dtype, output_shape, DeviceId::Cpu()));

  const int64_t outer = Product(shard_shape, 0, axis);
  const int64_t inner = Product(shard_shape, axis + 1, shard_shape.Rank());
  const int64_t element_bytes = DataTypeBitWidth(dtype) / 8;
  const int64_t chunk_bytes = shard_shape.Dim(axis) * inner * element_bytes;
  auto* dst = static_cast<std::byte*>(output.data());
  for (int64_t outer_index = 0; outer_index < outer; ++outer_index) {
    for (size_t rank = 0; rank < shards.size(); ++rank) {
      const auto* src = static_cast<const std::byte*>(shards[rank].data());
      if (chunk_bytes != 0)
        std::memcpy(dst + (outer_index * shards.size() + rank) * chunk_bytes,
                    src + outer_index * chunk_bytes, chunk_bytes);
    }
  }
  return output;
}

}  // namespace inferx::comm
