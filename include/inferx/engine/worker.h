#ifndef INFERX_MODELS_WORKER_H_
#define INFERX_MODELS_WORKER_H_

#include <cstdint>
#include <functional>
#include <memory>

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/kv_cache_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/dist/comm.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model_runner.h"

namespace inferx {
namespace models {

/// \brief Rank-local lifecycle owner: device, stream, communicator, memory
///        profiling, and failure signalling. Wraps one ModelRunner.
///
/// The Executor drives every Worker through the same sequence:
///
///   InitDevice -> LoadModel -> DetermineAvailableMemory
///   -> (engine: min across ranks, BuildKVCacheConfig)
///   -> InitializeFromConfig -> WarmUp -> ExecuteModel (repeatedly)
///
/// so each phase is testable alone and no collective hides inside the runner.
class Worker {
 public:
  /// \param model    Checkpoint/device config; the device must be CUDA.
  /// \param parallel This rank's tensor-parallel geometry.
  /// \param comm     This rank's communicator; null uses a world-of-one.
  static StatusOr<std::unique_ptr<Worker>> Create(const ModelConfig& model,
                                                  const ParallelConfig& parallel = {},
                                                  std::unique_ptr<CommBackend> comm = nullptr);

  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  /// \brief Acquires the device runtime and stream and builds the runner.
  Status InitDevice();

  /// \brief Loads weights (runner_->LoadModel).
  Status LoadModel(const LoadOptions& options);

  /// \brief Runs the profile step and returns this rank's KV byte budget:
  ///        utilization * device memory - weights - peak activations.
  StatusOr<int64_t> DetermineAvailableMemory(const CacheConfig& cache,
                                             const SchedulerConfig& scheduler);

  /// \brief Allocates the KV pool from the engine-agreed config
  ///        (runner_->InitializeKVCache).
  Status InitializeFromConfig(const KVCacheConfig& kv, const SchedulerConfig& scheduler);

  /// \brief Final warm-up; captures graphs once they are wired.
  Status WarmUp(const ExecutionConfig& execution);

  StatusOr<ModelRunnerOutput> ExecuteModel(const SchedulerOutput& step);

  /// \brief Hooks the executor uses to poison a rank group on failure.
  ///        ExecuteModel reports errors through `abort` and checks
  ///        `check_health` before each step.
  void SetFailureHooks(std::function<void(const Status&)> abort,
                       std::function<Status()> check_health);

  bool IsDriverRank() const { return is_driver_rank_; }
  const ParallelConfig& Parallel() const { return parallel_; }

  /// \brief Valid after InitDevice(). Exposed for tests and the attention
  ///        runtime accessor; not for lifecycle control.
  ModelRunner* Runner() { return runner_.get(); }

 private:
  Worker(const ModelConfig& model, const ParallelConfig& parallel,
         std::unique_ptr<CommBackend> comm);

  const ModelConfig model_config_;
  const ParallelConfig parallel_;
  const bool is_driver_rank_;

  DeviceRuntime* runtime_ = nullptr;
  DeviceStream stream_{};

  // comm_ always points at either owned_comm_ or single_rank_comm_.
  std::unique_ptr<CommBackend> owned_comm_;
  SingleRankComm single_rank_comm_;
  CommBackend* comm_ = nullptr;

  int64_t weights_bytes_ = 0;
  std::unique_ptr<ModelRunner> runner_;

  std::function<void(const Status&)> abort_;
  std::function<Status()> check_health_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_WORKER_H_