#include "inferx/comm/communicator.h"

#include <bit>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace inferx::comm {
namespace {

Status ValidateTensor(const TensorView& tensor, bool host_only) {
  if (!tensor.IsDefined())
    return InvalidArgumentError("AllReduceSum: tensor is undefined");
  if (host_only && !tensor.IsCpu())
    return InvalidArgumentError("HostSimComm requires a CPU tensor");
  switch (tensor.GetDataType()) {
    case DataType::kFloat:
    case DataType::kDouble:
    case DataType::kInt32:
    case DataType::kInt64:
    case DataType::kBFloat16:
      return OkStatus();
    default:
      return UnimplementedError("AllReduceSum does not support dtype ",
                                DataTypeName(tensor.GetDataType()));
  }
}

float Bf16ToFloat(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

uint16_t FloatToBf16(float value) {
  uint32_t bits = std::bit_cast<uint32_t>(value);
  const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}

template <typename T>
void SumTyped(const std::vector<std::vector<std::byte>>& inputs, int64_t count,
              void* output) {
  auto* dst = static_cast<T*>(output);
  for (int64_t i = 0; i < count; ++i) {
    T sum{};
    for (const auto& input : inputs)
      sum += reinterpret_cast<const T*>(input.data())[i];
    dst[i] = sum;
  }
}

void SumBf16(const std::vector<std::vector<std::byte>>& inputs, int64_t count,
             void* output) {
  auto* dst = static_cast<uint16_t*>(output);
  for (int64_t i = 0; i < count; ++i) {
    float sum = 0.0f;
    for (const auto& input : inputs)
      sum += Bf16ToFloat(reinterpret_cast<const uint16_t*>(input.data())[i]);
    dst[i] = FloatToBf16(sum);
  }
}

struct HostSimState {
  explicit HostSimState(int world_size)
      : size(world_size), inputs(static_cast<size_t>(world_size)),
        outputs(static_cast<size_t>(world_size), nullptr) {}

  Status AllReduce(int rank, const TensorView& tensor) {
    INFERX_RETURN_IF_ERROR(ValidateTensor(tensor, /*host_only=*/true));

    std::unique_lock lock(mu);
    if (arrived == 0) {
      dtype = tensor.GetDataType();
      count = tensor.Numel();
      bytes = tensor.NBytes();
      collective_status = OkStatus();
    } else if (dtype != tensor.GetDataType() || count != tensor.Numel()) {
      collective_status = InvalidArgumentError(
          "HostSimComm collective mismatch: expected ", DataTypeName(dtype),
          "[", count, "] but rank ", rank, " supplied ",
          DataTypeName(tensor.GetDataType()), "[", tensor.Numel(), "]");
    }

    inputs[static_cast<size_t>(rank)].resize(tensor.NBytes());
    if (tensor.NBytes() != 0) {
      std::memcpy(inputs[static_cast<size_t>(rank)].data(), tensor.Data(),
                  tensor.NBytes());
    }
    outputs[static_cast<size_t>(rank)] = tensor.Data();
    ++arrived;

    if (arrived == size) {
      if (collective_status.ok()) ReduceAndBroadcast();
      ready = true;
      cv.notify_all();
    } else {
      cv.wait(lock, [&] { return ready; });
    }

    const Status result = collective_status;
    ++departed;
    if (departed == size) {
      arrived = 0;
      departed = 0;
      ready = false;
      cv.notify_all();
    } else {
      cv.wait(lock, [&] { return !ready; });
    }
    return result;
  }

  void ReduceAndBroadcast() {
    std::vector<std::byte> result(bytes);
    switch (dtype) {
      case DataType::kFloat:
        SumTyped<float>(inputs, count, result.data());
        break;
      case DataType::kDouble:
        SumTyped<double>(inputs, count, result.data());
        break;
      case DataType::kInt32:
        SumTyped<int32_t>(inputs, count, result.data());
        break;
      case DataType::kInt64:
        SumTyped<int64_t>(inputs, count, result.data());
        break;
      case DataType::kBFloat16:
        SumBf16(inputs, count, result.data());
        break;
      default:
        collective_status = InternalError("validated dtype reached no reducer");
        return;
    }
    if (bytes != 0) {
      for (void* output : outputs) std::memcpy(output, result.data(), bytes);
    }
  }

  const int size;
  std::mutex mu;
  std::condition_variable cv;
  int arrived = 0;
  int departed = 0;
  bool ready = false;
  DataType dtype = DataType::kUndefined;
  int64_t count = 0;
  size_t bytes = 0;
  Status collective_status;
  std::vector<std::vector<std::byte>> inputs;
  std::vector<void*> outputs;
};

class HostSimComm final : public Communicator {
 public:
  HostSimComm(int rank, std::shared_ptr<HostSimState> state)
      : rank_(rank), state_(std::move(state)) {}

  int rank() const override { return rank_; }
  int size() const override { return state_->size; }
  Status AllReduceSum(const TensorView& tensor) override {
    return state_->AllReduce(rank_, tensor);
  }

 private:
  int rank_;
  std::shared_ptr<HostSimState> state_;
};

}  // namespace

Status SingleRankComm::AllReduceSum(const TensorView& tensor) {
  return ValidateTensor(tensor, /*host_only=*/false);
}

StatusOr<std::vector<std::unique_ptr<Communicator>>>
CreateHostSimCommunicators(int size) {
  if (size <= 0)
    return InvalidArgumentError("communicator size must be positive, got ",
                                size);
  auto state = std::make_shared<HostSimState>(size);
  std::vector<std::unique_ptr<Communicator>> communicators;
  communicators.reserve(static_cast<size_t>(size));
  for (int rank = 0; rank < size; ++rank)
    communicators.push_back(std::make_unique<HostSimComm>(rank, state));
  return communicators;
}

}  // namespace inferx::comm
