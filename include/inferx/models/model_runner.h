#ifndef INFERX_MODELS_MODEL_RUNNER_H_
#define INFERX_MODELS_MODEL_RUNNER_H_

#include <memory>
#include <functional>
#include <string>
#include <vector>

#include "inferx/config/cache_config.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/dist/comm.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/model.h"
#include "inferx/config/model_config.h"

namespace inferx {

struct ModelRunnerImpl;

/// \brief Executes one scheduler step on the model.
///
/// The runner is the execution entry point between the scheduler and the
/// model: it converts a SchedulerOutput into a flat ModelInput (tokens,
/// positions, attention metadata, KV block tables, logit rows), hands it to
/// Model::Forward, and greedily samples the returned logits. It owns the
/// loaded model, the paged KV block pool the scheduler allocates from, the
/// execution lane (runtime + stream), and whatever per-request state survives
/// between steps (block tables, computed-token watermarks, the last sampled
/// token). It contains no architecture-specific computation; Forward and the
/// ops below it own that.
class ModelRunner {
 public:
  /// \brief Loads the model and allocates the execution lane and KV pool.
  ///
  /// `scheduler` sizes the input buffers (token budget and sequence
  /// capacity) and must match the scheduler the engine steps with.
  /// An explicit `comm` is owned by a rank-local runner and must match
  /// `parallel`. With no communicator, size 1 selects SingleRankComm; larger
  /// worlds create an NCCL rank group across model.device.device_ids (or
  /// ordinals 0..size-1 when omitted). Multi-GPU serving is eager-only.
  static StatusOr<std::unique_ptr<ModelRunner>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, const ParallelConfig& parallel = {},
      std::unique_ptr<dist::CommBackend> comm = nullptr);

  /// Combines rank-local runners behind one scheduler. Ranks must be ordered
  /// by communicator rank and use equal scheduling/cache capacities. Only rank
  /// zero samples; its tokens are applied to every rank before the next step.
  /// `abort` must release blocked collectives and be safe to call concurrently.
  static StatusOr<std::unique_ptr<ModelRunner>> CreateGroup(
      std::vector<std::unique_ptr<ModelRunner>> ranks,
      std::function<void(const Status&)> abort,
      std::function<Status()> check_health = {});

  /// \brief Same, with a model supplied by the caller (tests).
  static StatusOr<std::unique_ptr<ModelRunner>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, std::unique_ptr<Model> loaded,
      std::unique_ptr<dist::CommBackend> comm = nullptr);

  ~ModelRunner();

  /// \brief The KV pool the scheduler allocates blocks from.
  KvBlockPool* kv_pool();

  /// \brief The loaded model's checkpoint configuration.
  const CheckpointConfig& checkpoint_config() const;

  /// \brief Executes one scheduler step.
  ///
  /// \param output The scheduler's plan for this step.
  /// \return       One sampled token per scheduled request, in batch order,
  ///               or an error status.
  StatusOr<ModelRunnerOutput> Run(const SchedulerOutput& output);

 private:
  explicit ModelRunner(std::unique_ptr<ModelRunnerImpl> impl);
  StatusOr<ModelRunnerOutput> RunLocal(const SchedulerOutput& output);

  std::unique_ptr<ModelRunnerImpl> impl_;
};

}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_RUNNER_H_
