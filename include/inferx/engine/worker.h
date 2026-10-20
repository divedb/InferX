#ifndef INFERX_ENGINE_WORKER_H_
#define INFERX_ENGINE_WORKER_H_

#include "inferx/engine/model_runner.h"

namespace inferx {
/// Rank-local owner of device, stream, communicator, memory budgeting, and
/// failure propagation. ModelRunner groups drive each phase across all ranks
/// before allocating caches from the minimum available budget.
class Worker {
 public:
  static StatusOr<std::unique_ptr<Worker>> Create(const ModelConfig& model,
                                                  const ParallelConfig& parallel = {},
                                                  std::unique_ptr<CommBackend> comm = nullptr);
  ~Worker();
  Status InitDevice();
  Status LoadModel();
  StatusOr<int64_t> DetermineAvailableMemory(const CacheConfig& cache,
                                             const SchedulerConfig& scheduler);
  // Used when a caller supplies one rank of an externally driven world.
  StatusOr<int64_t> AgreeAvailableMemory(int64_t bytes);
  Status InitializeFromConfig(int64_t kv_bytes, const CacheConfig& cache,
                              const SchedulerConfig& scheduler);
  Status WarmUp(const ExecutionConfig& execution);
  StatusOr<ModelRunnerOutput> ExecuteModel(const SchedulerOutput& step);
  void SetFailureHooks(std::function<void(const Status&)> abort,
                       std::function<Status()> check_health);
  bool IsDriverRank() const { return parallel_.tensor_parallel_rank == 0; }
  const ParallelConfig& Parallel() const { return parallel_; }
  ModelRunner* Runner() const { return runner_.get(); }

 private:
  Worker(const ModelConfig& model, const ParallelConfig& parallel,
         std::unique_ptr<CommBackend> comm);
  ModelConfig model_config_;
  ParallelConfig parallel_;
  models::ModelLoadOptions load_options_;
  DeviceRuntime* runtime_ = nullptr;
  OwnedStream stream_;
  std::unique_ptr<CommBackend> owned_comm_;
  SingleRankComm single_rank_comm_;
  CommBackend* comm_ = nullptr;
  std::unique_ptr<ModelRunner> runner_;
  enum class Phase { kCreated, kDevice, kLoaded, kProfiled, kCache, kReady };
  Phase phase_ = Phase::kCreated;
  Status failure_;
  std::function<void(const Status&)> abort_;
  std::function<Status()> check_health_;
};
}  // namespace inferx
#endif  // INFERX_ENGINE_WORKER_H_
