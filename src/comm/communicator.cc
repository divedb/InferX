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

#if defined(INFERX_WITH_CUDA)
#include <cuda_runtime_api.h>

#include "inferx/core/cuda_utils.h"
#endif

namespace inferx::comm {
namespace {

class ObservedCommunicator final : public Communicator {
 public:
  ObservedCommunicator(std::unique_ptr<Communicator> inner,
                       std::shared_ptr<CommMetrics> metrics,
                       CommObservationConfig config)
      : inner_(std::move(inner)),
        metrics_(std::move(metrics)),
        config_(config) {
#if defined(INFERX_WITH_CUDA)
    if (config_.timing_sample_every == 0 || !inner_->device().IsCuda()) return;
    if (config_.timing_ring_size == 0) config_.timing_ring_size = 1;
    if (cudaSetDevice(static_cast<int>(inner_->device().index)) !=
        cudaSuccess) {
      metrics_->RecordTimingDrop();
      config_.timing_sample_every = 0;
      return;
    }
    slots_.resize(config_.timing_ring_size);
    for (Slot& slot : slots_) {
      if (cudaEventCreate(&slot.start) != cudaSuccess ||
          cudaEventCreate(&slot.end) != cudaSuccess) {
        metrics_->RecordTimingDrop();
        config_.timing_sample_every = 0;
        break;
      }
    }
#endif
  }

  ~ObservedCommunicator() override {
#if defined(INFERX_WITH_CUDA)
    for (Slot& slot : slots_) {
      if (slot.start != nullptr) (void)cudaEventDestroy(slot.start);
      if (slot.end != nullptr) (void)cudaEventDestroy(slot.end);
    }
#endif
  }

  int rank() const override { return inner_->rank(); }
  int size() const override { return inner_->size(); }
  DeviceId device() const override { return inner_->device(); }
  CommBackend backend() const override { return inner_->backend(); }
  CommCapabilities capabilities() const override {
    return inner_->capabilities();
  }

  Status AllReduceSum(const TensorView& tensor, void* stream) override {
    const uint64_t bytes = tensor.IsDefined() ? tensor.NBytes() : 0;
#if defined(INFERX_WITH_CUDA)
    PollSamples();
    Slot* sample = BeginSample(stream);
#endif
    Status result = inner_->AllReduceSum(tensor, stream);
#if defined(INFERX_WITH_CUDA)
    EndSample(sample, stream, result.ok());
#endif
    metrics_->RecordAllReduce(bytes, result.ok());
    return result;
  }

  Status Abort() override {
    metrics_->RecordAbort();
    return inner_->Abort();
  }

 private:
#if defined(INFERX_WITH_CUDA)
  struct Slot {
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    bool active = false;
    bool record = true;
  };

  void PollSamples() noexcept {
    for (Slot& slot : slots_) {
      if (!slot.active) continue;
      const cudaError_t ready = cudaEventQuery(slot.end);
      if (ready == cudaErrorNotReady) continue;
      if (ready == cudaSuccess) {
        if (slot.record) {
          float milliseconds = 0.0f;
          if (cudaEventElapsedTime(&milliseconds, slot.start, slot.end) ==
              cudaSuccess) {
            metrics_->RecordLatency(milliseconds / 1000.0);
          } else {
            metrics_->RecordTimingDrop();
          }
        }
      } else {
        metrics_->RecordTimingDrop();
      }
      slot.active = false;
    }
  }

  Slot* BeginSample(void* stream) noexcept {
    if (config_.timing_sample_every == 0 || slots_.empty()) return nullptr;
    if (++collective_sequence_ % config_.timing_sample_every != 0) {
      return nullptr;
    }
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(cuda_stream, &capture) != cudaSuccess) {
      metrics_->RecordTimingDrop();
      return nullptr;
    }
    if (capture != cudaStreamCaptureStatusNone) {
      metrics_->RecordGraphSkip();
      return nullptr;
    }
    for (Slot& slot : slots_) {
      if (slot.active) continue;
      if (cudaEventRecord(slot.start, cuda_stream) != cudaSuccess) {
        metrics_->RecordTimingDrop();
        return nullptr;
      }
      return &slot;
    }
    metrics_->RecordTimingDrop();
    return nullptr;
  }

  void EndSample(Slot* slot, void* stream, bool success) noexcept {
    if (slot == nullptr) return;
    if (cudaEventRecord(slot->end, static_cast<cudaStream_t>(stream)) !=
        cudaSuccess) {
      metrics_->RecordTimingDrop();
      return;
    }
    slot->active = true;
    slot->record = success;
    if (!success) metrics_->RecordTimingDrop();
  }
#endif

  std::unique_ptr<Communicator> inner_;
  std::shared_ptr<CommMetrics> metrics_;
  CommObservationConfig config_;
#if defined(INFERX_WITH_CUDA)
  std::vector<Slot> slots_;
  uint64_t collective_sequence_ = 0;
#endif
};

Status ValidateTensor(const TensorView& tensor, bool allow_cuda) {
  if (!tensor.IsDefined())
    return InvalidArgumentError("AllReduceSum: tensor is undefined");
  if (!tensor.IsCpu() && !tensor.IsCuda())
    return InvalidArgumentError("AllReduceSum: unsupported device");
  if (tensor.IsCuda() && !allow_cuda)
    return UnimplementedError(
        "HostSimComm CUDA tensors require a CUDA-enabled build");
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
      : size(world_size),
        inputs(static_cast<size_t>(world_size)),
        outputs(static_cast<size_t>(world_size)) {}

  Status AllReduce(int rank, const TensorView& tensor, void* stream) {
#if defined(INFERX_WITH_CUDA)
    constexpr bool kAllowCuda = true;
#else
    constexpr bool kAllowCuda = false;
    (void)stream;
#endif
    INFERX_RETURN_IF_ERROR(ValidateTensor(tensor, kAllowCuda));

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
      if (tensor.IsCpu()) {
        std::memcpy(inputs[static_cast<size_t>(rank)].data(), tensor.Data(),
                    tensor.NBytes());
      } else {
#if defined(INFERX_WITH_CUDA)
        auto cuda_stream = static_cast<cudaStream_t>(stream);
        cudaError_t error = cudaStreamSynchronize(cuda_stream);
        if (error == cudaSuccess) {
          error = cudaMemcpy(inputs[static_cast<size_t>(rank)].data(),
                             tensor.Data(), tensor.NBytes(),
                             cudaMemcpyDeviceToHost);
        }
        if (error != cudaSuccess) {
          collective_status =
              InternalError("HostSimComm CUDA staging failed on rank ", rank,
                            ": ", cudaGetErrorString(error));
        }
#endif
      }
    }
    outputs[static_cast<size_t>(rank)] = tensor;
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
      for (const TensorView& output : outputs) {
        if (output.IsCpu()) {
          std::memcpy(output.Data(), result.data(), bytes);
        } else {
#if defined(INFERX_WITH_CUDA)
          const cudaError_t error = cudaMemcpy(output.Data(), result.data(),
                                               bytes, cudaMemcpyHostToDevice);
          if (error != cudaSuccess) {
            collective_status =
                InternalError("HostSimComm CUDA broadcast failed: ",
                              cudaGetErrorString(error));
            return;
          }
#endif
        }
      }
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
  std::vector<TensorView> outputs;
};

class HostSimComm final : public Communicator {
 public:
  HostSimComm(int rank, std::shared_ptr<HostSimState> state)
      : rank_(rank), state_(std::move(state)) {}

  int rank() const override { return rank_; }
  int size() const override { return state_->size; }
  DeviceId device() const override {
#if defined(INFERX_WITH_CUDA)
    return DeviceId::Cuda(0);
#else
    return DeviceId::Cpu();
#endif
  }
  CommBackend backend() const override { return CommBackend::kHostSim; }
  CommCapabilities capabilities() const override { return {}; }
  Status AllReduceSum(const TensorView& tensor, void* stream) override {
    return state_->AllReduce(rank_, tensor, stream);
  }
  Status Abort() override { return OkStatus(); }

 private:
  int rank_;
  std::shared_ptr<HostSimState> state_;
};

}  // namespace

const char* CommBackendName(CommBackend backend) {
  switch (backend) {
    case CommBackend::kSingleRank:
      return "single";
    case CommBackend::kHostSim:
      return "host_sim";
    case CommBackend::kNccl:
      return "nccl";
    case CommBackend::kMscclpp:
      return "mscclpp";
  }
  return "unknown";
}

CommMetricSnapshot CommMetrics::Snapshot() const noexcept {
  CommMetricSnapshot snapshot = {
      .all_reduce_calls = all_reduce_calls_.load(std::memory_order_relaxed),
      .all_reduce_bytes = all_reduce_bytes_.load(std::memory_order_relaxed),
      .collective_failures =
          collective_failures_.load(std::memory_order_relaxed),
      .aborts = aborts_.load(std::memory_order_relaxed),
  };
  for (size_t i = 0; i < snapshot.latency_buckets.size(); ++i) {
    snapshot.latency_buckets[i] =
        latency_buckets_[i].load(std::memory_order_relaxed);
  }
  snapshot.latency_count = latency_count_.load(std::memory_order_relaxed);
  snapshot.latency_sum_seconds =
      latency_sum_seconds_.load(std::memory_order_relaxed);
  snapshot.timing_samples_dropped =
      timing_samples_dropped_.load(std::memory_order_relaxed);
  snapshot.timing_graph_skips =
      timing_graph_skips_.load(std::memory_order_relaxed);
  return snapshot;
}

void CommMetrics::RecordAllReduce(uint64_t bytes, bool success) noexcept {
  all_reduce_calls_.fetch_add(1, std::memory_order_relaxed);
  all_reduce_bytes_.fetch_add(bytes, std::memory_order_relaxed);
  if (!success) collective_failures_.fetch_add(1, std::memory_order_relaxed);
}

void CommMetrics::RecordAbort() noexcept {
  aborts_.fetch_add(1, std::memory_order_relaxed);
}

void CommMetrics::RecordLatency(double seconds) noexcept {
  const auto bucket =
      std::lower_bound(kCollectiveLatencyBuckets.begin(),
                       kCollectiveLatencyBuckets.end(), seconds);
  if (bucket != kCollectiveLatencyBuckets.end()) {
    latency_buckets_[static_cast<size_t>(bucket -
                                         kCollectiveLatencyBuckets.begin())]
        .fetch_add(1, std::memory_order_relaxed);
  }
  latency_count_.fetch_add(1, std::memory_order_relaxed);
  latency_sum_seconds_.fetch_add(seconds, std::memory_order_relaxed);
}

void CommMetrics::RecordTimingDrop() noexcept {
  timing_samples_dropped_.fetch_add(1, std::memory_order_relaxed);
}

void CommMetrics::RecordGraphSkip() noexcept {
  timing_graph_skips_.fetch_add(1, std::memory_order_relaxed);
}

std::unique_ptr<Communicator> ObserveCommunicator(
    std::unique_ptr<Communicator> inner, std::shared_ptr<CommMetrics> metrics,
    CommObservationConfig config) {
  if (inner == nullptr || metrics == nullptr) return inner;
  return std::make_unique<ObservedCommunicator>(std::move(inner),
                                                std::move(metrics), config);
}

Status SingleRankComm::AllReduceSum(const TensorView& tensor, void*) {
  return ValidateTensor(tensor, /*allow_cuda=*/true);
}

StatusOr<std::vector<std::unique_ptr<Communicator>>> CreateHostSimCommunicators(
    int size) {
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
