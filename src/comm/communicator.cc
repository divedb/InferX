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

#include "inferx/core/device_runtime.h"

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
    if (config_.timing_sample_every == 0 || !inner_->device().IsAccelerator())
      return;
    if (config_.timing_ring_size == 0) config_.timing_ring_size = 1;
    auto selected = RuntimeFor(inner_->device());
    if (!selected.ok()) {
      metrics_->RecordTimingDrop();
      config_.timing_sample_every = 0;
      return;
    }
    runtime_ = *selected;
    if (!runtime_->SetDevice(inner_->device()).ok()) {
      metrics_->RecordTimingDrop();
      config_.timing_sample_every = 0;
      return;
    }
    slots_.resize(config_.timing_ring_size);
    for (Slot& slot : slots_) {
      auto start = runtime_->CreateEvent(true);
      auto end = runtime_->CreateEvent(true);
      if (!start.ok() || !end.ok()) {
        if (start.ok()) (void)runtime_->DestroyEvent(*start);
        if (end.ok()) (void)runtime_->DestroyEvent(*end);
        metrics_->RecordTimingDrop();
        config_.timing_sample_every = 0;
        break;
      }
      slot.start = *start;
      slot.end = *end;
    }
  }

  ~ObservedCommunicator() override {
    if (runtime_ == nullptr) return;
    for (Slot& slot : slots_) {
      if (slot.start.handle != nullptr)
        (void)runtime_->DestroyEvent(slot.start);
      if (slot.end.handle != nullptr) (void)runtime_->DestroyEvent(slot.end);
    }
  }

  int rank() const override { return inner_->rank(); }
  int size() const override { return inner_->size(); }
  DeviceId device() const override { return inner_->device(); }
  CommBackend backend() const override { return inner_->backend(); }
  CommCapabilities capabilities() const override {
    return inner_->capabilities();
  }

  Status AllReduceSum(const TensorView& tensor, Stream stream) override {
    const uint64_t bytes = tensor.IsDefined() ? tensor.NBytes() : 0;
    PollSamples();
    Slot* sample = BeginSample(stream);
    Status result = inner_->AllReduceSum(tensor, stream);
    EndSample(sample, stream, result.ok());
    metrics_->RecordAllReduce(bytes, result.ok());
    return result;
  }

  Status Abort() override {
    metrics_->RecordAbort();
    return inner_->Abort();
  }

 private:
  struct Slot {
    DeviceEvent start;
    DeviceEvent end;
    bool active = false;
    bool record = true;
  };

  void PollSamples() noexcept {
    for (Slot& slot : slots_) {
      if (!slot.active) continue;
      StatusOr<bool> ready = runtime_->QueryEvent(slot.end);
      if (ready.ok() && !*ready) continue;
      if (ready.ok()) {
        if (slot.record) {
          StatusOr<float> milliseconds =
              runtime_->ElapsedMs(slot.start, slot.end);
          if (milliseconds.ok()) {
            metrics_->RecordLatency(*milliseconds / 1000.0);
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

  Slot* BeginSample(Stream stream) noexcept {
    if (config_.timing_sample_every == 0 || slots_.empty()) return nullptr;
    if (++collective_sequence_ % config_.timing_sample_every != 0) {
      return nullptr;
    }
    StatusOr<bool> capture = runtime_->IsCapturing(stream);
    if (!capture.ok()) {
      metrics_->RecordTimingDrop();
      return nullptr;
    }
    if (*capture) {
      metrics_->RecordGraphSkip();
      return nullptr;
    }
    for (Slot& slot : slots_) {
      if (slot.active) continue;
      if (!runtime_->RecordEvent(slot.start, stream).ok()) {
        metrics_->RecordTimingDrop();
        return nullptr;
      }
      return &slot;
    }
    metrics_->RecordTimingDrop();
    return nullptr;
  }

  void EndSample(Slot* slot, Stream stream, bool success) noexcept {
    if (slot == nullptr) return;
    if (!runtime_->RecordEvent(slot->end, stream).ok()) {
      metrics_->RecordTimingDrop();
      return;
    }
    slot->active = true;
    slot->record = success;
    if (!success) metrics_->RecordTimingDrop();
  }
  std::unique_ptr<Communicator> inner_;
  std::shared_ptr<CommMetrics> metrics_;
  CommObservationConfig config_;
  DeviceRuntime* runtime_ = nullptr;
  std::vector<Slot> slots_;
  uint64_t collective_sequence_ = 0;
};

Status ValidateTensor(const TensorView& tensor, bool allow_accelerator) {
  if (!tensor.IsDefined())
    return InvalidArgumentError("AllReduceSum: tensor is undefined");
  if (!tensor.IsCpu() && !tensor.Device().IsAccelerator())
    return InvalidArgumentError("AllReduceSum: unsupported device");
  if (tensor.Device().IsAccelerator() && !allow_accelerator)
    return UnimplementedError(
        "HostSimComm accelerator tensors require a matching device runtime");
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
  HostSimState(int world_size, DeviceId selected_device,
               DeviceRuntime* selected_runtime)
      : size(world_size),
        device(selected_device),
        runtime(selected_runtime),
        inputs(static_cast<size_t>(world_size)),
        outputs(static_cast<size_t>(world_size)) {}

  Status AllReduce(int rank, const TensorView& tensor, Stream stream) {
    INFERX_RETURN_IF_ERROR(ValidateTensor(tensor, runtime != nullptr));
    if (tensor.Device() != device) {
      return InvalidArgumentError("HostSimComm rank ", rank, " uses ",
                                  device.ToString(), " but tensor is on ",
                                  tensor.Device().ToString());
    }

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
        Status staged = runtime->SynchronizeStream(stream);
        if (staged.ok()) {
          staged = runtime->Copy(inputs[static_cast<size_t>(rank)].data(),
                                 tensor.Data(), tensor.NBytes(),
                                 CopyKind::kDeviceToHost);
        }
        if (!staged.ok()) collective_status = staged;
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
          Status copied = runtime->Copy(output.Data(), result.data(), bytes,
                                        CopyKind::kHostToDevice);
          if (!copied.ok()) {
            collective_status = copied;
            return;
          }
        }
      }
    }
  }

  const int size;
  const DeviceId device;
  DeviceRuntime* const runtime;
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
  DeviceId device() const override { return state_->device; }
  CommBackend backend() const override { return CommBackend::kHostSim; }
  CommCapabilities capabilities() const override { return {}; }
  Status AllReduceSum(const TensorView& tensor, Stream stream) override {
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

Status SingleRankComm::AllReduceSum(const TensorView& tensor, Stream) {
  INFERX_RETURN_IF_ERROR(ValidateTensor(tensor, /*allow_accelerator=*/true));
  if (tensor.Device() != device_) {
    return InvalidArgumentError("SingleRankComm uses ", device_.ToString(),
                                " but tensor is on ",
                                tensor.Device().ToString());
  }
  return OkStatus();
}

StatusOr<std::vector<std::unique_ptr<Communicator>>> CreateHostSimCommunicators(
    int size, DeviceId device) {
  if (size <= 0)
    return InvalidArgumentError("communicator size must be positive, got ",
                                size);
  DeviceRuntime* runtime = nullptr;
  if (device.IsAccelerator()) {
    INFERX_ASSIGN_OR_RETURN(runtime, RuntimeFor(device));
    INFERX_RETURN_IF_ERROR(runtime->SetDevice(device));
  }
  auto state = std::make_shared<HostSimState>(size, device, runtime);
  std::vector<std::unique_ptr<Communicator>> communicators;
  communicators.reserve(static_cast<size_t>(size));
  for (int rank = 0; rank < size; ++rank)
    communicators.push_back(std::make_unique<HostSimComm>(rank, state));
  return communicators;
}

}  // namespace inferx::comm
