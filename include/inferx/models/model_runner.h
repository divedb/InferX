#ifndef INFERX_MODELS_MODEL_RUNNER_H_
#define INFERX_MODELS_MODEL_RUNNER_H_

#include <memory>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/model.h"
#include "inferx/models/model_config.h"

namespace inferx {

struct ModelRunnerImpl;

/// \brief Configuration for one model runner instance.
struct ModelRunnerConfig {
  std::string model_dir;                ///< Checkpoint directory.
  DeviceId device = DeviceId::Cuda(0);  ///< Placement device.
  /// \brief Upper bound on tokens per step; sizes input buffers. Must match
  ///        the scheduler's token budget.
  int max_num_batched_tokens = 4096;
  /// \brief Upper bound on concurrently scheduled requests.
  int max_num_seqs = 32;
  /// \brief Total KV blocks to allocate across all layers.
  int64_t num_kv_blocks = 2048;
  /// \brief Tokens per KV block.
  int64_t block_size = 16;
};

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
  static StatusOr<std::unique_ptr<ModelRunner>> Create(const ModelRunnerConfig& config);

  /// \brief Same, with a model supplied by the caller (tests).
  static StatusOr<std::unique_ptr<ModelRunner>> Create(const ModelRunnerConfig& config,
                                                       std::unique_ptr<Model> model);

  ~ModelRunner();

  /// \brief The KV pool the scheduler allocates blocks from.
  KvBlockPool* kv_pool();

  /// \brief The loaded model's configuration.
  const ModelConfig& model_config() const;

  /// \brief Executes one scheduler step.
  ///
  /// \param output The scheduler's plan for this step.
  /// \return       One sampled token per scheduled request, in batch order,
  ///               or an error status.
  StatusOr<ModelRunnerOutput> Run(const SchedulerOutput& output);

 private:
  explicit ModelRunner(std::unique_ptr<ModelRunnerImpl> impl);
  std::unique_ptr<ModelRunnerImpl> impl_;
};

}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_RUNNER_H_
