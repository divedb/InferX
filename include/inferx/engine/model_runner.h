#ifndef INFERX_ENGINE_MODEL_RUNNER_H_
#define INFERX_ENGINE_MODEL_RUNNER_H_

#include <functional>
#include <memory>
#include <vector>

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/engine/input_batch.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/sampling/sampler.h"

namespace inferx {
class Worker;

/// Executes scheduler steps. Create builds a group of rank-local Workers;
/// each Worker drives a local runner's lifecycle and owns its device resources.
/// Local runners own the model, attention, sampling, and persistent InputBatch.
/// Only the driver samples; accepted tokens are applied to every peer.
class ModelRunner {
 public:
  static StatusOr<std::unique_ptr<ModelRunner>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, const ParallelConfig& parallel = {},
      std::unique_ptr<CommBackend> comm = nullptr);
  static StatusOr<std::unique_ptr<ModelRunner>> CreateGroup(
      std::vector<std::unique_ptr<Worker>> ranks, const CacheConfig& cache,
      const SchedulerConfig& scheduler, const ExecutionConfig& execution,
      std::function<void(const Status&)> abort, std::function<Status()> check_health = {});

  ModelRunner(std::unique_ptr<models::Model> model, DeviceRuntime* runtime, DeviceStream stream,
              CommBackend* comm, bool is_driver_rank);
  ~ModelRunner();
  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  // Worker-driven rank-local lifecycle, in this order.
  Status LoadModel(const models::ModelLoadOptions& options);
  Status ProfileRun(const SchedulerConfig& scheduler);
  Status InitializeKVCache(int64_t kv_bytes, const CacheConfig& cache,
                           const SchedulerConfig& scheduler);
  Status CaptureGraphs(const ExecutionConfig& execution) const;
  StatusOr<ModelRunnerOutput> ExecuteModel(const SchedulerOutput& step);

  attn::AttentionRuntime* Attention();
  const models::ModelSpec& GetModelSpec() const;
  void EnableLogitsCapture();
  const std::vector<Tensor>& CapturedLogits() const;
  Status ApplyAcceptedTokens(RequestId id, TokenId token);

 private:
  ModelRunner();  // Group facade; local runners use the constructor above.
  StatusOr<ModelRunnerOutput> ExecuteGroup(const SchedulerOutput& step);
  Status RunProfileStep(Allocator& scratch, int64_t max_tokens, int64_t max_seqs);
  Status RunStep(const SchedulerOutput& step, ModelRunnerOutput* out);

  std::vector<std::unique_ptr<Worker>> workers_;
  std::function<void(const Status&)> abort_;
  std::function<Status()> check_health_;

  DeviceRuntime* runtime_ = nullptr;  // Borrowed from Worker.
  DeviceStream stream_{};
  CommBackend* comm_ = nullptr;
  bool is_driver_rank_ = false;
  std::unique_ptr<models::Model> model_;
  // Scratch must outlive all tensors allocated from it.
  std::unique_ptr<models::ScratchArena> scratch_;
  std::unique_ptr<attn::AttentionRuntime> attention_;
  std::vector<attn::AttentionOp> ops_;
  std::unique_ptr<ops::OpContext> op_;
  std::unique_ptr<sampling::Sampler> sampler_;
  std::unique_ptr<InputBatch> input_batch_;
  int64_t max_num_seqs_ = 0;
  bool capture_logits_ = false;
  std::vector<Tensor> captured_logits_;
};
}  // namespace inferx
#endif  // INFERX_ENGINE_MODEL_RUNNER_H_
