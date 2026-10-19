#ifndef INFERX_MODELS_MODEL_RUNNER_H_
#define INFERX_MODELS_MODEL_RUNNER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/config/scheduler_config.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/engine/input_batch.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/models/model_spec.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/op_context.h"
#include "inferx/sampling/sampler.h"

namespace inferx {
namespace models {

/// \brief Test/triage hooks, fixed at construction so the public API does
///        not grow per debug feature.
struct RunnerDebugOptions {
  /// Keep a copy of every step's [sequences, vocab] logits.
  bool capture_logits = false;
};

/// \brief Result of the profile step, consumed by Worker to size the KV pool.
struct ProfileResult {
  int64_t weights_bytes = 0;
  int64_t peak_activation_bytes = 0;  ///< Scratch + library workspaces.
};

/// \brief Per-step execution on one rank: model, attention, sampler, and the
///        persistent InputBatch.
///
/// Lifecycle is driven by Worker, in this order:
///   LoadModel -> ProfileRun -> InitializeKVCache -> CaptureGraphs
///   -> ExecuteModel (repeatedly).
/// Device, stream, communicator, memory budgeting, and failure poisoning
/// belong to Worker/Executor, not here. Scheduling and cache policy live
/// above; kernel dispatch lives below.
///
/// Single-threaded. Device work runs on one stream; ExecuteModel
/// synchronizes it before returning, so a returned ModelRunnerOutput is
/// complete.
class ModelRunner {
 public:
  /// \param model          Constructed but not yet loaded.
  /// \param runtime        Not owned; must outlive the runner.
  /// \param stream         Stream all device work runs on.
  /// \param comm           Not owned; never null (Worker supplies
  ///                       SingleRankComm for a world of one).
  /// \param is_driver_rank True if this rank samples; other ranks only
  ///                       keep state for their shard.
  /// \param debug          Test/triage hooks.
  ModelRunner(std::unique_ptr<Model> model, DeviceRuntime* runtime, DeviceStream stream,
              CommBackend* comm, bool is_driver_rank, RunnerDebugOptions debug = {});
  ~ModelRunner();

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;

  /// \brief Loads weights onto the device.
  Status LoadModel(const LoadOptions& options);

  /// \brief One max-shape forward + logits through a scratch arena, forcing
  ///        every scratch block and library workspace (cuBLAS, ...) to
  ///        materialize, and measuring peak usage. Runs against a throwaway
  ///        attention runtime.
  StatusOr<ProfileResult> ProfileRun(const SchedulerConfig& scheduler);

  /// \brief Creates the attention runtime from the engine-computed config
  ///        and binds one op per attention layer.
  Status InitializeKVCache(const KVCacheConfig& kv, const SchedulerConfig& scheduler);

  /// \brief Captures decode graphs. Not wired into the new model layer yet;
  ///        currently a no-op.
  Status CaptureGraphs();

  /// \brief Applies the scheduler diff, runs forward, samples, and returns
  ///        samples in scheduler order.
  StatusOr<ModelRunnerOutput> ExecuteModel(const SchedulerOutput& step);

  /// \brief The attention runtime owning the cache plan, pools, facade,
  ///        kernels, and metadata builders (docs/attention.md §8.5).
  attn::AttentionRuntime* Attention() { return attention_.get(); }
  const ModelSpec& GetModelSpec() const { return model_->GetModelSpec(); }

  /// \brief One [sequences, vocab] tensor per step; empty unless
  ///        RunnerDebugOptions::capture_logits was set.
  const std::vector<Tensor>& CapturedLogits() const { return captured_logits_; }

 private:
  /// \brief Flattened batch tensors every layer reads for one step.
  struct StepInputs {
    Tensor token_ids;
    Tensor positions;
    Tensor slot_mapping;
    Tensor block_tables;
    Tensor query_start_loc;
    Tensor logits_indices;           ///< Last scheduled token of each sequence.
    std::vector<RequestId> req_ids;  ///< Scheduler order.
    std::vector<int> rows;           ///< InputBatch row per sequence.
    std::vector<int> num_tokens;     ///< Tokens scheduled per sequence.
  };

  /// \brief Adds new requests, applies cached-request updates (including
  ///        tokens accepted on the driver rank, which non-driver ranks
  ///        receive here), and removes finished ids.
  Status UpdateStates(const SchedulerOutput& step);

  /// \brief Builds token ids, positions, slot mapping, and block tables
  ///        from the InputBatch and the cache facade.
  Status PrepareInputs(const SchedulerOutput& step, StepInputs* inputs);

  StatusOr<Tensor> Forward(const StepInputs& inputs);
  StatusOr<Tensor> ComputeLogits(const Tensor& hidden, const StepInputs& inputs);

  /// \brief Samples one token per sequence and appends it to the
  ///        InputBatch. No-op (empty samples) on non-driver ranks.
  Status Sample(const Tensor& logits, const StepInputs& inputs, ModelRunnerOutput* out);

  DeviceRuntime* runtime_;
  DeviceStream stream_;
  CommBackend* comm_;
  const bool is_driver_rank_;
  const RunnerDebugOptions debug_;

  std::unique_ptr<Model> model_;

  // Declaration order is the lifetime contract: ScratchArena owns both the
  // region and the allocator carved from it, so nothing else to keep in sync.
  std::unique_ptr<ScratchArena> scratch_;
  std::unique_ptr<attn::AttentionRuntime> attention_;
  std::vector<attn::AttentionOp> ops_;
  std::unique_ptr<ops::OpContext> op_;
  std::unique_ptr<sampling::Sampler> sampler_;  ///< Null unless driver rank.

  std::unique_ptr<InputBatch> input_batch_;  ///< Sized in InitializeKVCache.
  std::vector<Tensor> captured_logits_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_RUNNER_H_