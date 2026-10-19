#ifndef INFERX_MODELS_MODEL_RUNNER_H_
#define INFERX_MODELS_MODEL_RUNNER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/engine/scheduler_output.h"
#include "inferx/models/load_options.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/models/model.h"
#include "inferx/models/model_spec.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/ops/op_context.h"
#include "inferx/sampling/sampler.h"

namespace inferx {
namespace models {

/// \brief Rank-local execution: owns the model, the KV pool, the scratch
///        workspace, and the sampler, and drives one scheduler step.
///
/// This is the runtime seam the model layer hangs from. Per step it
/// flattens the SchedulerOutput into the batch tensors every layer reads
/// (token ids, positions, page tables), runs the model's forward, samples
/// one token per sequence, and hands the samples back in scheduler order.
/// Scheduling, admission, and cache policy live above it; kernel dispatch
/// lives below it.
///
/// Single-threaded like its callers (the scheduler and executor). Device
/// work runs on one stream; Run() synchronizes that stream before
/// returning, so a returned ModelRunnerOutput is complete.
class ModelRunner {
 public:
  /// \brief Builds the runner for one rank.
  ///
  /// \param model     Checkpoint/device config; the device must be CUDA.
  /// \param cache     KV pool sizing.
  /// \param scheduler Concurrency bounds (max sequences, token budget).
  /// \param execution Execution options (CUDA graphs are not wired yet).
  /// \param parallel  This rank's tensor-parallel geometry.
  /// \param comm      This rank's communicator; null uses the world-of-one.
  static StatusOr<std::unique_ptr<ModelRunner>> Create(
      const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
      const ExecutionConfig& execution, const ParallelConfig& parallel = {},
      std::unique_ptr<CommBackend> comm = nullptr);

  ~ModelRunner();

  /// \brief The attention runtime owning the cache plan, pools, facade,
  ///        kernels, and metadata builders (docs/attention.md §8.5).
  attn::AttentionRuntime* Attention() { return attention_.get(); }
  const ModelSpec& GetModelSpec() const { return model_->GetModelSpec(); }

  /// \brief Executes one scheduler step and returns its samples.
  StatusOr<ModelRunnerOutput> Run(const SchedulerOutput& step);

  /// \brief Hooks the executor uses to poison a rank group on failure.
  void SetFailureHooks(std::function<void(const Status&)> abort,
                       std::function<Status()> check_health);

  /// \brief Stops sampling; this rank only maintains state for its shard.
  void DisableSampling() { sampler_.reset(); }

  /// \brief Keeps a copy of every step's logits for numerical triage and
  ///        tests (greedy tie analysis, parity harnesses).
  void EnableLogitsCapture() { capture_logits_ = true; }

  /// \brief One [sequences, vocab] tensor per executed step, in order.
  const std::vector<Tensor>& CapturedLogits() const { return captured_logits_; }

  /// \brief Appends a token sampled elsewhere (rank zero of a TP group) to
  ///        this rank's view of the sequence.
  Status ApplyAcceptedTokens(RequestId request_id, TokenId token);

 private:
  ModelRunner(std::unique_ptr<Model> model,
              std::unique_ptr<CommBackend> owned_comm, DeviceRuntime* runtime,
              DeviceStream stream);

  /// \brief Runner-side per-request state: the token history and sampling
  ///        knobs. Block tables live in the cache facade, never mirrored
  ///        here (docs/attention.md §8.3).
  struct RequestState {
    std::vector<TokenId> tokens;      ///< Prompt plus accepted outputs.
    int num_computed = 0;             ///< Tokens whose KV this rank wrote.
    sampling::SamplingParams params;  ///< Sampling knobs for this request.
    int64_t rng_offset = 0;           ///< Generated tokens so far.
  };

  /// \brief Load weights, size and allocate the KV pool, prepare graphs.
  ///
  /// vLLM-shaped initialization: after the weights land, a profile step at
  /// the largest shape the scheduler can produce forces every scratch
  /// block and library workspace (cuBLAS, ...) to materialize; the KV pool
  /// then takes the utilization budget's leftover, minimum taken across
  /// the tensor-parallel group so every rank agrees on the block count.
  /// CUDA-graph capture would run last; it is not wired yet.
  Status Initialize(const CacheConfig& cache, const SchedulerConfig& scheduler,
                    const ExecutionConfig& execution);

  /// \brief One max-shape forward + logits through `scratch`, measuring,
  ///        against a throwaway attention runtime.
  Status RunProfileStep(Allocator& scratch, int64_t max_tokens, int64_t max_seqs);

  /// \brief Creates the attention runtime from the computed byte budget
  ///        (or the explicit config fields when utilization sizing is off)
  ///        and binds one op per attention layer.
  Status CreateAttention(int64_t kv_bytes, const CacheConfig& cache,
                         const SchedulerConfig& scheduler);

  /// \brief Captures decode graphs; not wired into the new model layer.
  Status CaptureGraphs(const ExecutionConfig& execution) const;

  Status RunStep(const SchedulerOutput& step, ModelRunnerOutput* out);

  DeviceRuntime* runtime_ = nullptr;
  DeviceStream stream_{};
  std::unique_ptr<Model> model_;
  std::unique_ptr<attn::AttentionRuntime> attention_;
  std::vector<attn::AttentionOp> ops_;
  std::unique_ptr<CommBackend> owned_comm_;
  SingleRankComm single_rank_comm_;
  CommBackend* comm_ = nullptr;
  std::unique_ptr<ops::OpContext> op_;

  // Scratch_region must outlive the allocator carved from it.
  std::unique_ptr<ScratchArena> scratch_;
  std::unique_ptr<sampling::Sampler> sampler_;
  int64_t max_num_seqs_ = 0;

  std::unordered_map<RequestId, RequestState> requests_;
  std::string checkpoint_dir_;
  bool capture_logits_ = false;
  std::vector<Tensor> captured_logits_;

  std::function<void(const Status&)> abort_;
  std::function<Status()> check_health_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_RUNNER_H_
