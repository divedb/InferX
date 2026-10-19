#include "inferx/models/model_runner.h"

#include <algorithm>

#include "inferx/ops/attention/cache_spec_bridge.h"
#include "inferx/ops/attention/op.h"
#include "inferx/ops/attention/registry.h"

#include <utility>

#include "inferx/models/registry.h"

namespace inferx {
namespace models {

namespace {

/// Uploads one host int32 array as a device tensor.
StatusOr<Tensor> UploadInt32(DeviceRuntime& runtime, DeviceStream stream,
                             const std::vector<int32_t>& values) {
  INFERX_ASSIGN_OR_RETURN(
      Tensor tensor,
      Tensor::Empty(DataType::kInt32,
                    Shape({static_cast<int64_t>(values.size())}), runtime.device()));
  if (values.empty()) return tensor;
  INFERX_RETURN_IF_ERROR(runtime.Memory().CopyAsync(tensor.Data(), values.data(),
                                           values.size() * sizeof(int32_t),
                                           CopyKind::kHostToDevice, stream));
  return tensor;
}

}  // namespace

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(
    const ModelConfig& model_config, const CacheConfig& cache,
    const SchedulerConfig& scheduler, const ExecutionConfig& execution,
    const ParallelConfig& parallel, std::unique_ptr<CommBackend> comm) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(model_config.device.Validate());
  INFERX_ASSIGN_OR_RETURN(const std::vector<Device> devices, model_config.device.Resolve());
  const Device device = devices.front();
  if (!device.IsCuda()) {
    return InvalidArgumentError("the model layer executes on CUDA, requested ",
                                device.ToString());
  }

  ModelLoadOptions options;
  options.directory = model_config.model_dir;
  options.device = device;
  INFERX_ASSIGN_OR_RETURN(options.dtype, ModelLoadOptions::ParseDType(model_config.dtype));
  options.parallel = parallel;
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<Model> model, BuildModel(options));

  INFERX_ASSIGN_OR_RETURN(DeviceRuntime* runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime->Activate());
  auto owned_stream = runtime->Streams().CreateStream(StreamPriority::kNormal);
  INFERX_RETURN_IF_ERROR(owned_stream.status());
  const DeviceStream stream = owned_stream->Release();

  std::unique_ptr<ModelRunner> runner(
      new ModelRunner(std::move(model), std::move(comm), runtime, stream));
  runner->checkpoint_dir_ = model_config.model_dir;
  INFERX_RETURN_IF_ERROR(runner->Initialize(cache, scheduler, execution));
  return runner;
}

ModelRunner::ModelRunner(std::unique_ptr<Model> model,
                         std::unique_ptr<CommBackend> owned_comm,
                         DeviceRuntime* runtime, DeviceStream stream)
    : runtime_(runtime),
      stream_(stream),
      model_(std::move(model)),
      owned_comm_(std::move(owned_comm)) {
  comm_ = owned_comm_ != nullptr ? owned_comm_.get() : &single_rank_comm_;
  op_ = std::make_unique<ops::OpContext>(*runtime_, stream_);
}

ModelRunner::~ModelRunner() {
  op_.reset();
  requests_.clear();
  sampler_.reset();
  ops_.clear();
  attention_.reset();  // Frees pools while no step references them.
  scratch_.reset();    // Tensors carved from the region are gone by now.
  if (runtime_ != nullptr && stream_.handle != nullptr) {
    (void)runtime_->Streams().SynchronizeStream(stream_);
    OwnedStream(stream_, &runtime_->Streams()).Reset();
  }
}

void ModelRunner::SetFailureHooks(std::function<void(const Status&)> abort,
                                  std::function<Status()> check_health) {
  abort_ = std::move(abort);
  check_health_ = std::move(check_health);
}

Status ModelRunner::ApplyAcceptedTokens(RequestId request_id, TokenId token) {
  auto it = requests_.find(request_id);
  if (it == requests_.end()) {
    return NotFoundError("apply-tokens for unknown request ", request_id);
  }
  it->second.tokens.push_back(token);
  return OkStatus();
}

StatusOr<ModelRunnerOutput> ModelRunner::Run(const SchedulerOutput& step) {
  ModelRunnerOutput output;
  INFERX_RETURN_IF_ERROR(RunStep(step, &output));
  return output;
}

Status ModelRunner::Initialize(const CacheConfig& cache, const SchedulerConfig& scheduler,
                               const ExecutionConfig& execution) {
  max_num_seqs_ = scheduler.max_num_seqs;

  // Weights first: they are the floor of every budget below.
  INFERX_ASSIGN_OR_RETURN(CheckpointReader checkpoint,
                          CheckpointReader::Open(checkpoint_dir_));
  INFERX_RETURN_IF_ERROR(model_->LoadWeights(checkpoint));

  // Scratch is measured, not computed: the profile step below runs the
  // largest shape the scheduler can produce through a measuring arena, and
  // the arena freezes at its high-water mark (plus margin). The frozen
  // region stays allocated, so the KV budget math below sees it as used.
  const ModelSpec& spec = model_->GetModelSpec();
  const int64_t max_tokens = scheduler.max_num_batched_tokens;
  constexpr double kScratchMargin = 0.25;
  {
    ScratchArena arena = ScratchArena::Measuring(spec.device);
    INFERX_RETURN_IF_ERROR(RunProfileStep(arena, max_tokens,
                                          std::min<int64_t>(max_num_seqs_, max_tokens)));
    INFERX_RETURN_IF_ERROR(arena.Freeze(spec.device, kScratchMargin));
    scratch_ = std::make_unique<ScratchArena>(std::move(arena));
  }
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<sampling::Sampler> sampler,
                          sampling::Sampler::Create(max_num_seqs_, spec.vocab_size,
                                                    spec.device));
  sampler_ = std::move(sampler);

  // Memory-based KV sizing: profile one max-shape step, measure what stays
  // allocated, and give the utilization budget's leftover to the pool. The
  // minimum across the tensor-parallel group keeps block counts uniform.
  int64_t kv_bytes = -1;
  if (cache.gpu_memory_utilization > 0.0f) {
    INFERX_RETURN_IF_ERROR(runtime_->Streams().SynchronizeStream(stream_));
    INFERX_ASSIGN_OR_RETURN(const DeviceMemoryInfo after, runtime_->Memory().GetMemoryInfo());
    const int64_t used = static_cast<int64_t>(after.total_bytes) -
                         static_cast<int64_t>(after.free_bytes);
    kv_bytes = static_cast<int64_t>(static_cast<double>(after.total_bytes) *
                                    cache.gpu_memory_utilization) - used;
    INFERX_ASSIGN_OR_RETURN(kv_bytes, comm_->AllReduceMin(*op_, kv_bytes));
  }

  INFERX_RETURN_IF_ERROR(CreateAttention(kv_bytes, cache, scheduler));
  return CaptureGraphs(execution);
}

Status ModelRunner::RunProfileStep(Allocator& scratch, int64_t max_tokens,
                                   int64_t max_seqs) {
  // One synthetic batch at the scheduler's ceiling: max_seqs sequences
  // splitting max_tokens tokens. A throwaway attention runtime holds the
  // profile KV so attention runs the real kernels; it is destroyed before
  // the runtime that matters is sized.
  const ModelSpec& spec = model_->GetModelSpec();

  auto registry = std::make_unique<attn::BackendRegistry>();
  attn::RegisterBuiltinBackends(*registry);
  attn::AttentionConfig config;
  config.block_size = 16;
  config.max_model_len = std::max<int64_t>(spec.max_position_embeddings, max_tokens);
  config.max_seqs = max_seqs;
  config.max_batch_tokens = max_tokens;
  // Generous throwaway budget: this runtime exists to run the kernels, not
  // to size anything; it dies before the measured freeze.
  config.cache_bytes_budget = 1LL << 32;
  config.device = spec.device;
  config.enable_prefix_caching = false;
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<attn::AttentionRuntime> runtime,
                          attn::AttentionRuntime::Create(std::move(registry), config,
                                                         model_->GetAttentionSpecs()));

  std::vector<attn::AttentionOp> ops;
  for (size_t l = 0; l < model_->GetAttentionSpecs().size(); ++l) {
    INFERX_ASSIGN_OR_RETURN(attn::AttentionOp op, runtime->MakeOp(static_cast<int64_t>(l)));
    ops.push_back(std::move(op));
  }

  // Synthetic requests: tokens_per_seq each, registered through the facade
  // exactly like a real step, so block tables and metadata are real.
  const int64_t tokens_per_seq = std::max<int64_t>(1, max_tokens / max_seqs);
  std::vector<TokenId> token_ids;
  std::vector<std::int32_t> query_lens, seq_lens, sample_rows;
  int64_t remaining = max_tokens;
  uint64_t request_id = 1;
  while (remaining > 0) {
    const int64_t chunk = std::min<int64_t>(tokens_per_seq, remaining);
    INFERX_RETURN_IF_ERROR(
        runtime->Cache().Reserve(request_id, GrowthTarget{0, chunk}));
    for (int64_t t = 0; t < chunk; ++t) token_ids.push_back(static_cast<TokenId>(0));
    query_lens.push_back(static_cast<std::int32_t>(chunk));
    seq_lens.push_back(static_cast<std::int32_t>(chunk));
    sample_rows.push_back(static_cast<std::int32_t>(token_ids.size() - 1));
    remaining -= chunk;
    ++request_id;
  }

  attn::BatchDesc desc;
  desc.num_reqs = static_cast<std::int32_t>(query_lens.size());
  desc.num_tokens = static_cast<std::int32_t>(token_ids.size());
  desc.query_lens = query_lens;
  desc.seq_lens = seq_lens;
  desc.max_query_len = *std::max_element(query_lens.begin(), query_lens.end());
  desc.max_seq_len = *std::max_element(seq_lens.begin(), seq_lens.end());
  desc.block_size = static_cast<std::int32_t>(runtime->GetPlan().block_size);
  desc.num_prefills = desc.num_reqs;
  desc.num_prefill_tokens = desc.num_tokens;
  // Flatten the per-group page tables; models here are single token-wise
  // group, so group 0 carries the batch's whole table. The vector outlives
  // BeginStep, whose BatchDesc borrows it.
  std::vector<BlockId> flat;
  for (uint64_t id = 1; id < request_id; ++id) {
    const auto blocks = runtime->Cache().GroupBlocks(id, 0);
    flat.insert(flat.end(), blocks.begin(), blocks.end());
  }
  desc.block_ids[0] = flat;
  desc.block_ids_num_pages[0] = static_cast<std::int32_t>(flat.size());

  INFERX_ASSIGN_OR_RETURN(attn::StepContext step, runtime->BeginStep(desc, *op_));

  StepInputs inputs;
  INFERX_ASSIGN_OR_RETURN(inputs.token_ids,
                          UploadInt32(*runtime_, stream_, token_ids));
  std::vector<std::int32_t> positions(static_cast<size_t>(token_ids.size()), 0);
  for (size_t i = 0; i < positions.size(); ++i) {
    const std::int32_t seq = static_cast<std::int32_t>(i / tokens_per_seq);
    positions[i] = static_cast<std::int32_t>(i - seq * tokens_per_seq);
  }
  INFERX_ASSIGN_OR_RETURN(inputs.positions, UploadInt32(*runtime_, stream_, positions));

  // The profile runs COMM-FREE through the world-of-one backend: tensor-
  // parallel ranks initialize sequentially (CreateGroup builds them one by
  // one), so a real collective here would deadlock the loopback barrier.
  // The measurements -- arena bytes, cuBLAS workspaces -- are identical
  // either way; collective backends allocate nothing from this arena.
  SingleRankComm profile_comm;
  ForwardContext ctx(*op_, step, profile_comm, scratch, absl::MakeConstSpan(ops));
  INFERX_ASSIGN_OR_RETURN(Tensor hidden,
                          Tensor::Empty(spec.dtype,
                                        Shape({static_cast<int64_t>(token_ids.size()),
                                               spec.hidden_size}),
                                        &scratch));
  INFERX_RETURN_IF_ERROR(model_->Forward(ctx, inputs, hidden));
  // The head path's buffers, measured by allocation: the gathered sample
  // rows and the [rows, vocab] logits. Running ComputeLogits itself would
  // issue the TP all-gather tail, which the identity comm rejects on
  // shape; the bytes are what the measurement needs, not the GEMM.
  {
    INFERX_ASSIGN_OR_RETURN(
        Tensor sampled,
        Tensor::Empty(spec.dtype,
                      Shape({static_cast<int64_t>(sample_rows.size()),
                             spec.hidden_size}),
                      &scratch));
    INFERX_ASSIGN_OR_RETURN(
        Tensor logits,
        Tensor::Empty(spec.dtype,
                      Shape({static_cast<int64_t>(sample_rows.size()),
                             spec.vocab_size}),
                      &scratch));
    (void)sampled;
    (void)logits;
  }
  return runtime_->Streams().SynchronizeStream(stream_);
  // `runtime` dies here: its bytes return to the device before
  // CreateAttention sizes the real pools.
}

Status ModelRunner::CreateAttention(int64_t kv_bytes, const CacheConfig& cache,
                                    const SchedulerConfig& scheduler) {
  const ModelSpec& spec = model_->GetModelSpec();

  // Byte budget: utilization leftover, explicit bytes, or blocks x block
  // bytes (the facade + runtime derive the actual split).
  int64_t budget = kv_bytes;
  if (budget < 0) budget = cache.kv_cache_memory_bytes;
  if (budget <= 0 && cache.num_kv_blocks > 0) {
    // Convert the block count into a byte budget; the runtime rounds down
    // to whole blocks from it.
    int64_t per_block = 0;
    for (const attn::AttentionSpec& attn_spec : model_->GetAttentionSpecs()) {
      if (!attn::NeedsCache(attn_spec)) continue;
      INFERX_ASSIGN_OR_RETURN(const int64_t per_token, attn::CacheBytesPerToken(attn_spec));
      per_block += cache.block_size * per_token;
    }
    budget = cache.num_kv_blocks * per_block;
  }
  if (budget <= 0) {
    return ResourceExhaustedError("KV budget of ", kv_bytes,
                                  " bytes fits no cache; check the cache config");
  }

  auto registry = std::make_unique<attn::BackendRegistry>();
  attn::RegisterBuiltinBackends(*registry);
  attn::AttentionConfig config;
  config.block_size = cache.block_size;
  config.max_model_len =
      std::max<int64_t>(spec.max_position_embeddings, scheduler.max_num_batched_tokens);
  config.max_seqs = max_num_seqs_;
  config.max_batch_tokens = scheduler.max_num_batched_tokens;
  config.cache_bytes_budget = budget;
  config.device = spec.device;
  config.enable_prefix_caching = cache.enable_prefix_caching;
  INFERX_ASSIGN_OR_RETURN(attention_,
                          attn::AttentionRuntime::Create(std::move(registry), config,
                                                         model_->GetAttentionSpecs()));

  ops_.clear();
  const std::vector<attn::AttentionSpec> specs = model_->GetAttentionSpecs();
  for (size_t l = 0; l < specs.size(); ++l) {
    INFERX_ASSIGN_OR_RETURN(attn::AttentionOp op,
                            attention_->MakeOp(static_cast<int64_t>(l)));
    ops_.push_back(std::move(op));
  }
  return OkStatus();
}

Status ModelRunner::CaptureGraphs(const ExecutionConfig& execution) const {
  if (execution.enable_cuda_graphs) {
    return UnimplementedError(
        "CUDA graph replay is not wired into the new model layer yet");
  }
  return OkStatus();
}

Status ModelRunner::RunStep(const SchedulerOutput& step, ModelRunnerOutput* out) {
  const auto fail = [this](Status status) {
    if (abort_) abort_(status);
    return status;
  };

  for (const RequestId id : step.finished_request_ids) requests_.erase(id);
  for (const RequestId id : step.preempted_request_ids) requests_.erase(id);

  // New requests first: the batch walk below needs their state to exist.
  for (const NewRequestData& fresh : step.scheduled_new_reqs) {
    RequestState state;
    state.tokens.assign(fresh.prompt_token_ids.begin(), fresh.prompt_token_ids.end());
    state.num_computed = fresh.num_computed_tokens;
    state.params = fresh.sampling_params;
    requests_[fresh.request_id] = std::move(state);
  }
  // CachedRequestUpdate carried the runner's mirrored block tables; the
  // facade owns them now and the tables below read straight from it.

  if (step.scheduled.empty()) return OkStatus();
  const int64_t block_size = attention_->GetPlan().block_size;
  const std::uint16_t num_groups = attention_->NumGroups();

  // Flatten the step: token stream and positions. Page tables are NOT
  // mirrored here -- they are read from the cache facade below, the single
  // owner (docs/attention.md §8.3).
  std::vector<int32_t> input_ids, positions, sample_rows;
  std::vector<int32_t> query_lens, seq_lens;
  std::vector<RequestState*> batch;
  std::vector<int64_t> prompt_lens;
  int64_t total_new = 0;

  for (const ScheduledRequest& scheduled : step.scheduled) {
    auto it = requests_.find(scheduled.request_id);
    if (it == requests_.end()) {
      return fail(InternalError("scheduled request ", scheduled.request_id,
                                " has no runner state"));
    }
    RequestState& state = it->second;
    const int64_t end = state.num_computed + scheduled.num_new_tokens;
    if (scheduled.num_new_tokens <= 0 || end > static_cast<int64_t>(state.tokens.size())) {
      return fail(InternalError("request ", scheduled.request_id, " schedules ", end,
                                " tokens but only carries ", state.tokens.size()));
    }
    for (int64_t t = state.num_computed; t < end; ++t) {
      input_ids.push_back(state.tokens[t]);
      positions.push_back(static_cast<int32_t>(t));
    }
    sample_rows.push_back(static_cast<int32_t>(total_new + scheduled.num_new_tokens - 1));
    query_lens.push_back(static_cast<std::int32_t>(scheduled.num_new_tokens));
    seq_lens.push_back(static_cast<std::int32_t>(end));
    total_new += scheduled.num_new_tokens;
    batch.push_back(&state);
    prompt_lens.push_back(static_cast<int64_t>(state.tokens.size()) - state.rng_offset);
  }

  INFERX_ASSIGN_OR_RETURN(Tensor ids_tensor, UploadInt32(*runtime_, stream_, input_ids));
  INFERX_ASSIGN_OR_RETURN(Tensor positions_tensor, UploadInt32(*runtime_, stream_, positions));
  INFERX_ASSIGN_OR_RETURN(Tensor sample_rows_tensor, UploadInt32(*runtime_, stream_, sample_rows));

  StepInputs inputs;
  inputs.token_ids = std::move(ids_tensor);
  inputs.positions = std::move(positions_tensor);

  // The batch description over the facade's own tables; decodes-first
  // ordering is the scheduler's contract.
  attn::BatchDesc desc;
  desc.num_reqs = static_cast<std::int32_t>(batch.size());
  desc.num_tokens = static_cast<std::int32_t>(total_new);
  desc.query_lens = query_lens;
  desc.seq_lens = seq_lens;
  desc.max_query_len = *std::max_element(query_lens.begin(), query_lens.end());
  desc.max_seq_len = *std::max_element(seq_lens.begin(), seq_lens.end());
  desc.block_size = static_cast<std::int32_t>(block_size);
  desc.num_decodes = static_cast<std::int32_t>(
      std::count_if(query_lens.begin(), query_lens.end(), [](int32_t l) { return l == 1; }));
  desc.num_decode_tokens = desc.num_decodes;
  desc.num_prefills = desc.num_reqs - desc.num_decodes;
  desc.num_prefill_tokens = desc.num_tokens - desc.num_decode_tokens;
  // The batch description over the scheduler's published snapshot: the
  // runner never reads the live facade tables for a step in flight
  // (docs/attention.md §9).
  for (std::uint16_t g = 0; g < num_groups; ++g) {
    const std::vector<int32_t>& table =
        step.group_block_ids[static_cast<size_t>(g < kMaxSchedulerGroups ? g : 0)];
    desc.block_ids[g] = table;
    desc.block_ids_num_pages[g] = static_cast<std::int32_t>(table.size());
  }

  INFERX_ASSIGN_OR_RETURN(attn::StepContext step_ctx, attention_->BeginStep(desc, *op_));

  const ModelSpec& spec = model_->GetModelSpec();
  ForwardContext ctx(*op_, step_ctx, *comm_, *scratch_, absl::MakeConstSpan(ops_));
  INFERX_ASSIGN_OR_RETURN(
      Tensor hidden,
      Tensor::Empty(spec.dtype,
                    Shape({static_cast<int64_t>(total_new), spec.hidden_size}),
                    scratch_.get()));
  INFERX_RETURN_IF_ERROR(model_->Forward(ctx, inputs, hidden));
  INFERX_ASSIGN_OR_RETURN(
      Tensor logits,
      Tensor::Empty(spec.dtype, Shape({static_cast<int64_t>(batch.size()), spec.vocab_size}),
                    scratch_.get()));
  INFERX_RETURN_IF_ERROR(model_->ComputeLogits(ctx, hidden, sample_rows_tensor, logits));

  if (capture_logits_) {
    // Snapshot into ordinary device memory: the scratch block returns to
    // the step pool when the step's tensors go out of scope.
    INFERX_ASSIGN_OR_RETURN(Tensor snapshot,
                            Tensor::Empty(spec.dtype, logits.GetShape(), op_->GetDevice()));
    INFERX_RETURN_IF_ERROR(logits.CopyTo(snapshot));
    captured_logits_.push_back(std::move(snapshot));
  }

  // Sample one token per sequence. A sequence still inside its (chunked)
  // prompt has no valid next-token distribution; its row samples garbage
  // that the watermark check below discards.
  std::vector<int32_t> sampled(batch.size(), -1);
  if (sampler_ != nullptr) {
    std::vector<sampling::SamplingMetadata::PerRequest> rows;
    rows.reserve(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      const RequestState& state = *batch[i];
      sampling::SamplingMetadata::PerRequest row;
      row.params = &state.params;
      row.rng_offset = state.rng_offset;
      row.history = &state.tokens;
      rows.push_back(row);
    }
    sampling::SamplingMetadata metadata =
        sampling::SamplingMetadata::Build(spec.vocab_size, absl::MakeConstSpan(rows));
    INFERX_ASSIGN_OR_RETURN(sampling::SamplerOutput result,
                            sampler_->Sample(*op_, logits, metadata));
    INFERX_RETURN_IF_ERROR(runtime_->Memory().CopyAsync(
      sampled.data(), result.sampled_token_ids.Data(), sampled.size() * sizeof(int32_t),
      CopyKind::kDeviceToHost, stream_));
  }

  // Everything downstream reads host values; one sync retires the step.
  INFERX_RETURN_IF_ERROR(runtime_->Streams().SynchronizeStream(stream_));

  out->samples.reserve(batch.size());
  for (size_t i = 0; i < batch.size(); ++i) {
    RequestState& state = *batch[i];
    state.num_computed += step.scheduled[i].num_new_tokens;
    SampledTokens sample;
    sample.request_id = step.scheduled[i].request_id;
    if (sampled[i] >= 0 && state.num_computed >= prompt_lens[i]) {
      sample.token_ids.push_back(sampled[i]);
      state.tokens.push_back(sampled[i]);
      state.rng_offset += 1;
    }
    out->samples.push_back(std::move(sample));
  }
  // Prefix publication is the scheduler's job (UpdateFromOutput): the
  // facade is single-threaded with it (docs/attention.md §9).
  return OkStatus();
}

}  // namespace models
}  // namespace inferx
