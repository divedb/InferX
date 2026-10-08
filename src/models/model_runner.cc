#include "inferx/models/model_runner.h"

#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/logging.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/models/load_options.h"
#include "inferx/models/registry.h"
#include "inferx/ops/op_context.h"
#include "inferx/sampling/sampler.h"
#include "inferx/util/hf_config.h"
#include "inferx/models/execution/batch_buffers.h"
#include "inferx/models/execution/graph_manager.h"
#include "inferx/models/execution/request_state.h"
#include "inferx/models/execution/state_storage.h"
#include "inferx/models/execution/step_commit.h"
#include "inferx/models/execution/step_launcher.h"
#include "inferx/models/execution/step_plan.h"
#include "inferx/models/execution/step_sampler.h"
#include "inferx/models/execution/stream_waiter.h"

namespace inferx {
namespace {

namespace exec = models::execution;

/// Model-facing inputs for one step, assembled from the uploaded device views.
struct StepInputs {
  models::ModelInputs model;
  models::LogitSelection selection;
};

template <class Views>
StepInputs MakeStepInputs(const exec::StepPlan& plan, Views& v) {
  const auto& packed = plan.packed;
  StepInputs in{
      models::ModelInputs{
          std::move(v.token_ids),
          models::AttentionBatch{std::move(v.positions), std::move(v.batch_indices),
                                 std::move(v.qo_indptr), std::move(v.kv_indptr),
                                 std::move(v.kv_indices), std::move(v.last_page_len),
                                 absl::MakeConstSpan(packed.qo_indptr),
                                 absl::MakeConstSpan(packed.kv_indptr),
                                 static_cast<int>(packed.tokens.size()), plan.batch()}},
      models::LogitSelection{std::move(v.logit_rows)}};
  in.model.attention.recurrent_indices = std::move(v.recurrent_indices);
  return in;
}

}  // namespace

struct ModelRunnerImpl {
  // Configuration (immutable after Create).
  ModelConfig config;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  DeviceId device;  ///< Resolved once from config.device.PrimaryDevice().
  exec::PlanLimits limits;

  // Device resources.
  DeviceRuntime* runtime = nullptr;
  Stream stream;
  exec::StreamWaiter waiter;
  std::unique_ptr<dist::CommBackend> comm;

  // Model and per-step machinery.
  std::unique_ptr<models::CausalLanguageModel> model;
  exec::StateStorage storage;
  std::optional<exec::BatchBuffers> buffers;
  std::optional<exec::GraphManager> graphs;
  std::unique_ptr<exec::RequestStateTable> requests;
  std::unique_ptr<sampling::Sampler> sampler;

  // Behaviour and failure state.
  bool sample_output = true;
  Status failure;
  std::function<void(const Status&)> abort;

  ~ModelRunnerImpl() {
    if (runtime == nullptr) return;
    (void)runtime->Activate();

    // Drain queued work even after an operation failed before the runner's sync.
    if (stream.handle != nullptr) (void)runtime->SynchronizeStream(stream);
    // Graphs reference model and buffer addresses; destroy them first.
    if (graphs.has_value()) graphs->DestroyAll();
    waiter.Reset();
    if (stream.handle != nullptr) (void)runtime->DestroyStream(stream);
  }

  // ---- Initialization ------------------------------------------------------

  Status Init(std::unique_ptr<models::CausalLanguageModel> loaded) {
    model = std::move(loaded);
    requests = std::make_unique<exec::RequestStateTable>(scheduler.max_num_seqs);
    INFERX_RETURN_IF_ERROR(InitStorage());
    INFERX_RETURN_IF_ERROR(InitBuffers());
    graphs.emplace(*runtime);
    INFERX_RETURN_IF_ERROR(PrepareModel());
    INFERX_RETURN_IF_ERROR(InitSampler());
    InitLimits();
    LogReady();
    return OkStatus();
  }

  Status InitStorage() {
    const auto& requirements = model->requirements().layers;
    if (requirements.empty()) {
      return InvalidArgumentError("model must declare state for each decoder layer");
    }
    INFERX_RETURN_IF_ERROR(storage.Init(requirements, cache.num_kv_blocks, cache.block_size,
                                        scheduler.max_num_seqs, device));
    if (storage.has_recurrent()) requests->InitRecurrentSlots(scheduler.max_num_seqs);
    return OkStatus();
  }

  Status InitBuffers() {
    buffers.emplace();
    return buffers->Init(*runtime, device, scheduler.max_num_batched_tokens,
                         scheduler.max_num_seqs, cache.num_kv_blocks, storage.has_recurrent());
  }

  /// Explicit model preparation: workspace exists before any request;
  /// prepared eager execution performs no device allocation.
  Status PrepareModel() {
    ops::OpContext ctx(*runtime, stream);
    models::ForwardContext fctx = MakeForwardContext(ctx);
    models::ExecutionLimits model_limits;
    model_limits.max_tokens = scheduler.max_num_batched_tokens;
    model_limits.max_sequences = scheduler.max_num_seqs;
    model_limits.max_logit_rows = scheduler.max_num_seqs;
    return model->Prepare(model_limits, fctx);
  }

  Status InitSampler() {
    if (!sample_output) return OkStatus();
    INFERX_ASSIGN_OR_RETURN(sampler, sampling::Sampler::Create(scheduler.max_num_seqs,
                                                               model->info().vocab_size,
                                                               device));
    return OkStatus();
  }

  void InitLimits() {
    limits.max_tokens = scheduler.max_num_batched_tokens;
    limits.max_seqs = scheduler.max_num_seqs;
    limits.num_kv_blocks = cache.num_kv_blocks;
    limits.block_size = cache.block_size;
    limits.max_context = model->info().max_position_embeddings;
    limits.vocab_size = model->info().vocab_size;
    limits.has_recurrent = storage.has_recurrent();
  }

  void LogReady() const {
    const auto usage = model->memory_usage();
    INFERX_LOG(INFO) << "runner ready: device=" << device.ToString()
                     << " model=" << config.model_dir
                     << " kv_blocks=" << cache.num_kv_blocks
                     << " block_size=" << cache.block_size
                     << " max_tokens=" << scheduler.max_num_batched_tokens
                     << " max_seqs=" << scheduler.max_num_seqs
                     << " weights_mb=" << (usage.weights_bytes >> 20)
                     << " workspace_mb=" << (usage.workspace_bytes >> 20);
  }

  models::ForwardContext MakeForwardContext(ops::OpContext& ctx) {
    return models::ForwardContext{ctx, storage.bindings(), *comm};
  }

  // ---- Per-step execution --------------------------------------------------

  /// Retires finished requests, admits new ones, and grows cached ones.
  /// Returns the scheduler's announced watermarks for cached requests.
  StatusOr<exec::ScheduledEnds> ApplyLifecycle(const SchedulerOutput& output) {
    // Process removals before admission so a finished ID can safely be reused.
    for (RequestId id : output.finished_request_ids) {
      INFERX_RETURN_IF_ERROR(requests->Retire(id));
    }
    for (const auto& nr : output.scheduled_new_reqs) {
      INFERX_ASSIGN_OR_RETURN(auto* record, requests->Admit(nr));
      if (storage.has_recurrent()) {
        INFERX_ASSIGN_OR_RETURN(record->recurrent_slot, requests->AcquireRecurrentSlot());
        // A fresh sequence starts from zero state; stream-ordered so the
        // step's kernels (and captured graphs) read zeros first.
        ops::OpContext lifecycle_ctx(*runtime, stream);
        INFERX_RETURN_IF_ERROR(
            storage.mutable_recurrent()->ResetSlot(lifecycle_ctx, record->recurrent_slot));
      }
    }

    exec::ScheduledEnds ends;
    for (const auto& cu : output.scheduled_cached_reqs) {
      auto* record = requests->Find(cu.request_id);
      if (record == nullptr) return InternalError("unknown cached request ", cu.request_id);
      record->block_ids.insert(record->block_ids.end(), cu.new_block_ids.begin(),
                               cu.new_block_ids.end());
      // The scheduler's watermark includes this allocation; the runner keeps
      // its committed start until execution succeeds.
      ends[cu.request_id] = cu.num_computed_tokens;
    }
    return ends;
  }

  StatusOr<ModelRunnerOutput> Execute(const SchedulerOutput& output) {
    INFERX_ASSIGN_OR_RETURN(exec::ScheduledEnds ends, ApplyLifecycle(output));

    ModelRunnerOutput result;
    if (output.scheduled.empty()) return result;

    // Context length is the model's declaration, not a runner-allocated
    // capacity: consult it live rather than the init-time snapshot.
    limits.max_context = model->info().max_position_embeddings;
    INFERX_ASSIGN_OR_RETURN(exec::StepPlan plan, exec::BuildStepPlan(output, *requests,
                                                                     *storage.kv(), limits,
                                                                     ends));
    INFERX_ASSIGN_OR_RETURN(auto views, buffers->Upload(plan.packed, plan.batch(), stream));
    StepInputs in = MakeStepInputs(plan, views);

    ops::OpContext ctx(*runtime, stream);
    models::ForwardContext fctx = MakeForwardContext(ctx);
    INFERX_VLOG(2) << "step: tokens=" << plan.packed.tokens.size() << " seqs=" << plan.batch()
                   << " rows=" << plan.num_rows() << " pure_decode=" << plan.pure_decode;

    exec::GraphPolicy graph_policy;
    graph_policy.allowed = execution.enable_cuda_graphs && plan.pure_decode &&
                           plan.num_rows() == plan.batch() && device.IsCuda();
    graph_policy.key = plan.batch();
    INFERX_ASSIGN_OR_RETURN(auto logits, exec::LaunchStep(*model, *graphs, stream, in.model,
                                                          in.selection, graph_policy, fctx));

    std::vector<int32_t> sampled;
    if (sample_output && plan.num_rows() > 0) {
      if (!logits.has_value()) return InternalError("missing logits for selected rows");
      auto tokens = exec::SampleStep(*sampler, ctx, *runtime, stream, waiter, *buffers, *logits,
                                     plan, model->info().vocab_size);
      if (!tokens.ok()) return tokens.status();
      sampled = std::move(*tokens);
    } else {
      INFERX_RETURN_IF_ERROR(waiter.Wait());
    }

    // Commit progress only after execution and sampling succeeded.
    exec::CommitStep(plan, sampled, sample_output, result);
    return result;
  }
};

// ---- ModelRunner public API ------------------------------------------------

ModelRunner::ModelRunner(std::unique_ptr<ModelRunnerImpl> impl) : impl_(std::move(impl)) {}
ModelRunner::~ModelRunner() = default;
KvBlockPool* ModelRunner::kv_pool() { return impl_->storage.kv(); }
const models::ModelInfo& ModelRunner::model_info() const { return impl_->model->info(); }

void ModelRunner::SetFailureHooks(std::function<void(const Status&)> abort,
                                  std::function<Status()> check_health) {
  impl_->abort = std::move(abort);
  impl_->waiter.SetHealthCheck(std::move(check_health));
}

void ModelRunner::DisableSampling() {
  impl_->sample_output = false;
  impl_->sampler.reset();
}

Status ModelRunner::ApplyAcceptedTokens(RequestId id, TokenId token) {
  auto* record = impl_->requests->Find(id);
  if (record == nullptr) return InternalError("accepted token for unknown request ", id);
  record->last_sampled = token;
  return OkStatus();
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(
    const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
    const ExecutionConfig& execution, std::unique_ptr<models::CausalLanguageModel> loaded,
    std::unique_ptr<dist::CommBackend> comm) {
  if (!comm) comm = std::make_unique<dist::SingleRankComm>();
  if (comm->size() <= 0 || comm->rank() < 0 || comm->rank() >= comm->size()) {
    return InvalidArgumentError("runner requires a valid communicator topology");
  }
  if (!loaded || scheduler.max_num_batched_tokens <= 0 || scheduler.max_num_seqs <= 0 ||
      cache.num_kv_blocks <= 0 || cache.num_kv_blocks > std::numeric_limits<int32_t>::max() ||
      cache.block_size <= 0 || cache.block_size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("runner requires a model, and positive int32 KV capacities");
  }
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  const DeviceId device = model.device.PrimaryDevice();

  auto impl = std::make_unique<ModelRunnerImpl>();
  impl->config = model;
  impl->cache = cache;
  impl->scheduler = scheduler;
  impl->execution = execution;
  impl->device = device;
  impl->comm = std::move(comm);
  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(impl->runtime->Activate());
  INFERX_ASSIGN_OR_RETURN(impl->stream, impl->runtime->CreateStream());
  impl->waiter.Bind(impl->runtime, &impl->stream);
  INFERX_RETURN_IF_ERROR(impl->Init(std::move(loaded)));
  return std::unique_ptr<ModelRunner>(new ModelRunner(std::move(impl)));
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(
    const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
    const ExecutionConfig& execution, const ParallelConfig& parallel,
    std::unique_ptr<dist::CommBackend> comm) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  if (!comm) comm = std::make_unique<dist::SingleRankComm>();
  if (comm->size() != parallel.tensor_parallel_size ||
      comm->rank() != parallel.tensor_parallel_rank) {
    return InvalidArgumentError("runner communicator must match the model's parallel topology");
  }
  if (scheduler.max_num_batched_tokens <= 0 || scheduler.max_num_seqs <= 0) {
    return InvalidArgumentError("runner requires positive token and sequence capacities");
  }
  models::ModelLoadOptions options;
  options.directory = model.model_dir;
  options.device = model.device.PrimaryDevice();
  options.max_tokens = scheduler.max_num_batched_tokens;
  options.max_seqs = scheduler.max_num_seqs;
  options.parallel = parallel;
  INFERX_ASSIGN_OR_RETURN(auto loaded, models::LoadModel(options));
  return Create(model, cache, scheduler, execution, std::move(loaded), std::move(comm));
}

StatusOr<ModelRunnerOutput> ModelRunner::RunLocal(const SchedulerOutput& output) {
  if (!impl_->failure.ok()) {
    return FailedPreconditionError("runner failed previously: ", impl_->failure.message());
  }
  auto result = [&]() -> StatusOr<ModelRunnerOutput> {
    try {
      return impl_->Execute(output);
    } catch (const std::exception& error) {
      return InternalError("rank execution failed: ", error.what());
    } catch (...) {
      return InternalError("rank execution failed with an unknown exception");
    }
  }();
  if (!result.ok()) {
    if (impl_->abort) impl_->abort(result.status());
    // A scheduler step and cache writes cannot be rolled back safely. Drain
    // queued work and require reconstruction instead of retrying partial state.
    (void)impl_->runtime->SynchronizeStream(impl_->stream);
    impl_->failure = result.status();
  }
  return result;
}

StatusOr<ModelRunnerOutput> ModelRunner::Run(const SchedulerOutput& output) {
  return RunLocal(output);
}

}  // namespace inferx
