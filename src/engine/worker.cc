#include "inferx/engine/worker.h"

#include <exception>
#include <utility>

#include "inferx/models/registry.h"

namespace inferx {
StatusOr<std::unique_ptr<Worker>> Worker::Create(const ModelConfig& model,
                                                 const ParallelConfig& parallel,
                                                 std::unique_ptr<CommBackend> comm) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  if ((comm && (comm->rank() != parallel.tensor_parallel_rank ||
                comm->size() != parallel.tensor_parallel_size)) ||
      (!comm && parallel.tensor_parallel_size != 1)) {
    return InvalidArgumentError("worker communicator does not match parallel config");
  }
  return std::unique_ptr<Worker>(new Worker(model, parallel, std::move(comm)));
}

Worker::Worker(const ModelConfig& model, const ParallelConfig& parallel,
               std::unique_ptr<CommBackend> comm)
    : model_config_(model), parallel_(parallel), owned_comm_(std::move(comm)) {
  comm_ = owned_comm_ ? owned_comm_.get() : &single_rank_comm_;
}

Worker::~Worker() {
  if (runtime_) {
    (void)runtime_->Activate();
    if (stream_) (void)runtime_->Streams().SynchronizeStream(stream_.Get());
  }
  runner_.reset();
  stream_.Reset();
}

Status Worker::InitDevice() {
  if (phase_ != Phase::kCreated)
    return FailedPreconditionError("InitDevice called out of order");
  INFERX_ASSIGN_OR_RETURN(auto devices, model_config_.device.Resolve());
  const Device device = devices.front();
  if (!device.IsCuda()) return InvalidArgumentError("model execution requires CUDA");
  INFERX_ASSIGN_OR_RETURN(runtime_, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime_->Activate());
  INFERX_ASSIGN_OR_RETURN(stream_, runtime_->Streams().CreateStream(StreamPriority::kNormal));
  load_options_.directory = model_config_.model_dir;
  load_options_.device = device;
  load_options_.parallel = parallel_;
  INFERX_ASSIGN_OR_RETURN(load_options_.dtype,
                          models::ModelLoadOptions::ParseDType(model_config_.dtype));
  INFERX_ASSIGN_OR_RETURN(auto model, models::BuildModel(load_options_));
  runner_ = std::make_unique<ModelRunner>(std::move(model), runtime_, stream_.Get(), comm_,
                                          IsDriverRank());
  phase_ = Phase::kDevice;
  return OkStatus();
}

Status Worker::LoadModel() {
  if (phase_ != Phase::kDevice) return FailedPreconditionError("LoadModel called out of order");
  INFERX_RETURN_IF_ERROR(runtime_->Activate());
  INFERX_RETURN_IF_ERROR(runner_->LoadModel(load_options_));
  phase_ = Phase::kLoaded;
  return OkStatus();
}

StatusOr<int64_t> Worker::DetermineAvailableMemory(const CacheConfig& cache,
                                                   const SchedulerConfig& scheduler) {
  if (phase_ != Phase::kLoaded) return FailedPreconditionError("profile called out of order");
  INFERX_RETURN_IF_ERROR(runtime_->Activate());
  INFERX_RETURN_IF_ERROR(runner_->ProfileRun(scheduler));
  phase_ = Phase::kProfiled;
  if (cache.gpu_memory_utilization <= 0) return int64_t{-1};
  INFERX_RETURN_IF_ERROR(runtime_->Streams().SynchronizeStream(stream_.Get()));
  INFERX_ASSIGN_OR_RETURN(auto memory, runtime_->Memory().GetMemoryInfo());
  const int64_t used = static_cast<int64_t>(memory.total_bytes - memory.free_bytes);
  const int64_t budget =
      static_cast<int64_t>(memory.total_bytes * cache.gpu_memory_utilization) - used;
  if (budget <= 0)
    return ResourceExhaustedError("no memory remains for KV cache after profiling");
  return budget;
}

StatusOr<int64_t> Worker::AgreeAvailableMemory(int64_t bytes) {
  if (phase_ != Phase::kProfiled)
    return FailedPreconditionError("budget agreement before profile");
  INFERX_RETURN_IF_ERROR(runtime_->Activate());
  ops::OpContext op(*runtime_, stream_.Get());
  return comm_->AllReduceMin(op, bytes);
}

Status Worker::InitializeFromConfig(int64_t kv_bytes, const CacheConfig& cache,
                                    const SchedulerConfig& scheduler) {
  if (phase_ != Phase::kProfiled)
    return FailedPreconditionError("cache init called out of order");
  INFERX_RETURN_IF_ERROR(runtime_->Activate());
  INFERX_RETURN_IF_ERROR(runner_->InitializeKVCache(kv_bytes, cache, scheduler));
  phase_ = Phase::kCache;
  return OkStatus();
}

Status Worker::WarmUp(const ExecutionConfig& execution) {
  if (phase_ != Phase::kCache) return FailedPreconditionError("WarmUp called out of order");
  INFERX_RETURN_IF_ERROR(runtime_->Activate());
  INFERX_RETURN_IF_ERROR(runner_->CaptureGraphs(execution));
  phase_ = Phase::kReady;
  return OkStatus();
}

StatusOr<ModelRunnerOutput> Worker::ExecuteModel(const SchedulerOutput& step) {
  if (phase_ != Phase::kReady) return FailedPreconditionError("worker is not ready");
  if (!failure_.ok()) return failure_;
  if (check_health_) {
    auto status = check_health_();
    if (!status.ok()) return status;
  }
  auto execute = [&]() -> StatusOr<ModelRunnerOutput> {
    INFERX_RETURN_IF_ERROR(runtime_->Activate());
    return runner_->ExecuteModel(step);
  };
  try {
    auto result = execute();
    if (!result.ok()) {
      failure_ = result.status();
      if (abort_) abort_(failure_);
    }
    return result;
  } catch (const std::exception& error) {
    auto status = InternalError("rank execution threw: ", error.what());
    failure_ = status;
    if (abort_) abort_(failure_);
    return failure_;
  }
}

void Worker::SetFailureHooks(std::function<void(const Status&)> abort,
                             std::function<Status()> check_health) {
  abort_ = std::move(abort);
  check_health_ = std::move(check_health);
}
}  // namespace inferx
