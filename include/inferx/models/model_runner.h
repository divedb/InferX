#ifndef INFERX_MODELS_MODEL_RUNNER_H_
#define INFERX_MODELS_MODEL_RUNNER_H_

#include <functional>
#include <memory>
#include <vector>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/dist/comm.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/model.h"
#include "inferx/models/model_info.h"

namespace inferx {

class ModelRunnerImpl;

/// \brief Executes one scheduler step on ONE rank's model, through the
///        model's own execution contract.
///
/// The runner is the rank-local execution boundary: it keeps the execution
/// mirror of scheduled requests (stable slots, committed watermarks, RNG
/// positions), packs flat ragged batches into fixed-capacity buffers,
/// binds persistent state, invokes the model eagerly or through captured
/// decode graphs, and runs sampling for the rows the step selected. It
/// contains no architecture-specific computation and no rank
/// coordination -- engine::ModelExecutor owns multi-rank dispatch.
///
/// Step semantics: only requests whose allocation completes a prompt or
/// continues decode select a logit row; mid-prompt prefill chunks compute
/// state without logits, sampling, or RNG advancement, so chunking never
/// changes a request's sampling stream.
class ModelRunner {
 public:
  /// \brief Rank-local runner over an already-loaded model (tests and the
  ///        executor). `comm` must match the loaded model's parallel
  ///        topology; absent means single rank.
  static StatusOr<std::unique_ptr<ModelRunner>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, std::unique_ptr<models::CausalLanguageModel> loaded,
      std::unique_ptr<dist::CommBackend> comm = nullptr);

  /// \brief Rank-local runner loading its model shard from the model
  ///        directory (worker processes; the coordinator's world building
  ///        lives in engine::ModelExecutor).
  static StatusOr<std::unique_ptr<ModelRunner>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, const ParallelConfig& parallel = {},
      std::unique_ptr<dist::CommBackend> comm = nullptr);

  ~ModelRunner();

  /// \brief The KV pool the scheduler allocates blocks from.
  KvBlockPool* kv_pool();

  /// \brief The loaded model's engine-facing facts.
  const models::ModelInfo& model_info() const;

  /// \brief Executes one scheduler step on this rank.
  ///
  /// \param output The scheduler's plan for this step.
  /// \return       One entry per scheduled request, in batch order;
  ///               requests without a selected output row report an empty
  ///               token vector.
  StatusOr<ModelRunnerOutput> Run(const SchedulerOutput& output);

  /// \brief Publishes a token sampled by the output owner to this rank's
  ///        execution mirror (non-sampling ranks of a group).
  Status ApplyAcceptedTokens(RequestId id, TokenId token);

  /// \brief Group hooks installed by the executor: abort releases peers
  ///        blocked in collectives; check_health polls the world while a
  ///        step drains. Engine-internal.
  void SetFailureHooks(std::function<void(const Status&)> abort,
                       std::function<Status()> check_health);

  /// \brief Marks this rank as a non-sampling group participant and drops
  ///        its sampler. Engine-internal.
  void DisableSampling();

 private:
  explicit ModelRunner(std::unique_ptr<ModelRunnerImpl> impl);
  StatusOr<ModelRunnerOutput> RunLocal(const SchedulerOutput& output);

  std::unique_ptr<ModelRunnerImpl> impl_;
};

}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_RUNNER_H_
