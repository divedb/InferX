#include "inferx/models/worker.h"

#include <algorithm>
#include <string>
#include <utility>

namespace inferx {
namespace models {

// ASSUMED APIs (adjust to the real ones):
//   DeviceRuntime::ForDevice(Device) -> StatusOr<DeviceRuntime*>
//   DeviceRuntime::CreateStream()    -> StatusOr<DeviceStream>
//   DeviceRuntime::TotalMemoryBytes()
//   Model::Create(config, parallel, comm, runtime, stream)
//       -> StatusOr<std::unique_ptr<Model>>
//   ParallelConfig::tp_rank; CacheConfig::gpu_memory_utilization

StatusOr<std::unique_ptr<Worker>> Worker::Create(const ModelConfig& model,
                                                 const ParallelConfig& parallel,
                                                 std::unique_ptr<CommBackend> comm) {
  return std::unique_ptr<Worker>(new Worker(model, parallel, std::move(comm)));
}

Worker::Worker(const ModelConfig& model, const ParallelConfig& parallel,
               std::unique_ptr<CommBackend> comm)
    : model_config_(model),
      parallel_(parallel),
      is_driver_rank_(parallel.tp_rank == 0),
      owned_comm_(std::move(comm)),
      comm_(owned_comm_ ? owned_comm_.get() : static_cast<CommBackend*>(&single_rank_comm_)) {}

Worker::~Worker() = default;

Status Worker::InitDevice() {
  if (runner_) return Internal("InitDevice called twice");

  auto runtime = DeviceRuntime::ForDevice(model_config_.device);
  if (!runtime.ok()) return runtime.status();
  runtime_ = *runtime;

  auto stream = runtime_->CreateStream();
  if (!stream.ok()) return stream.status();
  stream_ = *stream;

  auto model = Model::Create(model_config_, parallel_, comm_, runtime_, stream_);
  if (!model.ok()) return model.status();

  runner_ = std::make_unique<ModelRunner>(std::move(*model), runtime_, stream_, comm_,
                                          is_driver_rank_);
  return OkStatus();
}

Status Worker::LoadModel(const LoadOptions& options) {
  if (!runner_) return Internal("LoadModel called before InitDevice");
  return runner_->LoadModel(options);
}

StatusOr<int64_t> Worker::DetermineAvailableMemory(const CacheConfig& cache,
                                                   const SchedulerConfig& scheduler) {
  if (!runner_) return Internal("DetermineAvailableMemory called before InitDevice");

  // The profile step runs regardless, so scratch blocks and library
  // workspaces (cuBLAS, ...) exist before the KV pool is carved out.
  auto profile = runner_->ProfileRun(scheduler);
  if (!profile.ok()) return profile.status();
  weights_bytes_ = profile->weights_bytes;

  // Explicit block count: the engine ignores the byte budget.
  if (cache.gpu_memory_utilization <= 0) return int64_t{0};

  const int64_t budget =
      static_cast<int64_t>(cache.gpu_memory_utilization * runtime_->TotalMemoryBytes());
  const int64_t available = budget - profile->weights_bytes - profile->peak_activation_bytes;
  if (available <= 0) {
    return ResourceExhausted("memory budget (" + std::to_string(budget) +
                             " bytes) is smaller than weights plus peak activations");
  }
  return available;
}

Status Worker::InitializeFromConfig(const KVCacheConfig& kv, const SchedulerConfig& scheduler) {
  if (!runner_) return Internal("InitializeFromConfig called before InitDevice");
  return runner_->InitializeKVCache(kv, scheduler);
}

Status Worker::WarmUp(const ExecutionConfig& /*execution*/) {
  if (!runner_) return Internal("WarmUp called before InitDevice");
  return runner_->CaptureGraphs();
}

StatusOr<ModelRunnerOutput> Worker::ExecuteModel(const SchedulerOutput& step) {
  if (!runner_) return Internal("ExecuteModel called before InitDevice");

  if (check_health_) {
    Status health = check_health_();
    if (!health.ok()) return health;
  }

  auto out = runner_->ExecuteModel(step);
  if (!out.ok() && abort_) abort_(out.status());
  return out;
}

void Worker::SetFailureHooks(std::function<void(const Status&)> abort,
                             std::function<Status()> check_health) {
  abort_ = std::move(abort);
  check_health_ = std::move(check_health);
}

}  // namespace models
}  // namespace inferx