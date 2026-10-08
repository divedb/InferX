#ifndef INFERX_ENGINE_MODEL_EXECUTOR_H_
#define INFERX_ENGINE_MODEL_EXECUTOR_H_

#include <functional>
#include <memory>
#include <vector>

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/core/status.h"
#include "inferx/dist/comm.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/model_info.h"
#include "inferx/models/model_runner.h"

namespace inferx {

/// \brief Executes one already-scheduled step across the participating
///        ranks. The executor is rank coordination, never a second
///        scheduler: admission, token budgets, and cache policy stay in
///        the Scheduler; each rank's ModelRunner owns local execution.
///
/// A local deployment uses the same object with a single rank. With no
/// communicator and tensor parallelism above one, the executor builds an
/// NCCL world across the model's devices and drives one rank-local runner
/// per device (multi-GPU serving is eager-only).
class ModelExecutor {
 public:
  /// \brief Builds the executor for a model configuration: one rank-local
  ///        runner, or a rank group across devices for tensor parallelism.
  ///
  /// An explicit `comm` belongs to a rank-local runner and must match
  /// `parallel`; without one, size one uses a single-rank communicator.
  static StatusOr<std::unique_ptr<ModelExecutor>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, const ParallelConfig& parallel = {},
      std::unique_ptr<dist::CommBackend> comm = nullptr);

  /// \brief Combines already-built rank runners behind one executor.
  ///
  /// Ranks must be ordered by communicator rank and use equal
  /// scheduling/cache capacities. Only rank zero samples; its tokens are
  /// applied to every rank before the next step. `abort` must release
  /// blocked collectives and be safe to call concurrently.
  static StatusOr<std::unique_ptr<ModelExecutor>> CreateGroup(
      std::vector<std::unique_ptr<ModelRunner>> ranks,
      std::function<void(const Status&)> abort,
      std::function<Status()> check_health = {});

  ~ModelExecutor();

  /// \brief The KV pool of the coordinating rank (the scheduler allocates
  ///        from it; group ranks share logical page IDs).
  KvBlockPool* kv_pool();

  /// \brief The coordinating rank's engine-facing model facts.
  const models::ModelInfo& model_info() const;

  /// \brief Executes one scheduler step across all participating ranks and
  ///        returns the output owner's samples.
  StatusOr<ModelRunnerOutput> Execute(const SchedulerOutput& output);

 private:
  struct Impl;
  explicit ModelExecutor(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_MODEL_EXECUTOR_H_
