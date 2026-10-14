#include "inferx/models/model_runner.h"

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

/// Bytes one block occupies across every cached layer.
int64_t BlockBytesAllLayers(const std::vector<KvLayout>& layouts, int64_t block_size) {
  int64_t bytes = 0;
  for (const KvLayout& layout : layouts) {
    bytes += block_size * layout.BytesPerToken();
  }
  return bytes;
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
  attention_plan_ = ops::AttentionPlanWorkspace{};
  scratch_.reset();  // Tensors carved from the region are gone by now.
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
  INFERX_ASSIGN_OR_RETURN(
      attention_plan_.plan,
      Tensor::Empty(DataType::kInt32, Shape({3 * max_tokens + 1}), scratch_.get()));

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

  INFERX_RETURN_IF_ERROR(AllocateKvPool(kv_bytes, cache));
  return CaptureGraphs(execution);
}

Status ModelRunner::RunProfileStep(Allocator& scratch, int64_t max_tokens,
                                   int64_t max_seqs) {
  // One synthetic batch at the scheduler's ceiling: max_seqs sequences
  // splitting max_tokens tokens. A throwaway pool holds the profile KV so
  // attention runs the real kernels; it is destroyed before the pool that
  // matters is sized.
  const ModelSpec& spec = model_->GetModelSpec();
  const int64_t block_size = 16;
  const std::vector<int32_t> blocks((max_tokens + block_size - 1) / block_size);
  std::vector<int32_t> block_ids(blocks.size());
  for (size_t i = 0; i < block_ids.size(); ++i) block_ids[i] = static_cast<int32_t>(i);
  const int64_t tokens_per_seq = std::max<int64_t>(1, max_tokens / max_seqs);
  std::vector<int32_t> token_ids, positions, batch_indices, qo_indptr{0}, kv_indptr{0},
      kv_indices, last_page_len, sample_rows;
  for (int64_t seq = 0; seq < max_seqs; ++seq) {
    const int64_t begin = std::min(seq * tokens_per_seq, max_tokens);
    const int64_t end = seq == max_seqs - 1 ? max_tokens
                                            : std::min(begin + tokens_per_seq, max_tokens);
    for (int64_t t = begin; t < end; ++t) {
      token_ids.push_back(0);
      positions.push_back(static_cast<int32_t>(t - begin));
      batch_indices.push_back(static_cast<int32_t>(seq));
    }
    if (end <= begin) continue;
    sample_rows.push_back(static_cast<int32_t>(token_ids.size() - 1));
    const int64_t kv_len = end;
    const int64_t pages = (kv_len + block_size - 1) / block_size;
    kv_indices.insert(kv_indices.end(), block_ids.begin(), block_ids.begin() + pages);
    qo_indptr.push_back(static_cast<int32_t>(token_ids.size()));
    kv_indptr.push_back(static_cast<int32_t>(kv_indices.size()));
    last_page_len.push_back(static_cast<int32_t>((kv_len - 1) % block_size + 1));
  }

  INFERX_ASSIGN_OR_RETURN(
      KvBlockPool pool,
      KvBlockPool::Create(spec.state.paged_kv.size(), static_cast<int64_t>(blocks.size()),
                          block_size, spec.state.paged_kv.front(), spec.device));

  StepInputs inputs;
  INFERX_ASSIGN_OR_RETURN(inputs.token_ids, UploadInt32(*runtime_, stream_, token_ids));
  INFERX_ASSIGN_OR_RETURN(inputs.positions, UploadInt32(*runtime_, stream_, positions));
  AttentionMetadata attn;
  INFERX_ASSIGN_OR_RETURN(attn.batch_indices, UploadInt32(*runtime_, stream_, batch_indices));
  INFERX_ASSIGN_OR_RETURN(attn.qo_indptr, UploadInt32(*runtime_, stream_, qo_indptr));
  INFERX_ASSIGN_OR_RETURN(attn.kv_indptr, UploadInt32(*runtime_, stream_, kv_indptr));
  INFERX_ASSIGN_OR_RETURN(attn.kv_indices, UploadInt32(*runtime_, stream_, kv_indices));
  INFERX_ASSIGN_OR_RETURN(attn.last_page_len, UploadInt32(*runtime_, stream_, last_page_len));
  attn.host_qo_indptr = absl::MakeConstSpan(qo_indptr);
  attn.num_tokens = static_cast<int>(token_ids.size());
  attn.num_seqs = static_cast<int>(max_seqs);

  // The persistent plan lands in the frozen arena after measurement; the
  // profile step carries its own, freed before Freeze demands an empty
  // arena.
  ops::AttentionPlanWorkspace plan;
  INFERX_ASSIGN_OR_RETURN(plan.plan,
                          Tensor::Empty(DataType::kInt32,
                                        Shape({3 * max_tokens + 1}), &scratch));
  INFERX_RETURN_IF_ERROR(ops::BeginAttentionStep(
      *op_, *attn.kv_indptr, *attn.last_page_len, block_size, attn.num_tokens,
      attn.num_seqs, plan));

  // The profile runs COMM-FREE through the world-of-one backend: tensor-
  // parallel ranks initialize sequentially (CreateGroup builds them one by
  // one), so a real collective here would deadlock the loopback barrier.
  // The measurements -- arena bytes, cuBLAS workspaces -- are identical
  // either way; collective backends allocate nothing from this arena.
  SingleRankComm profile_comm;
  ForwardContext ctx(*op_, attn, pool, profile_comm, scratch, plan);
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
  // `pool` dies here: its bytes return to the device before AllocateKvPool.
}

Status ModelRunner::AllocateKvPool(int64_t kv_bytes, const CacheConfig& cache) {
  const ModelSpec& spec = model_->GetModelSpec();
  const int64_t per_block = BlockBytesAllLayers(spec.state.paged_kv, cache.block_size);
  if (per_block <= 0) {
    return InvalidArgumentError("cannot size a KV pool for a model with no paged state");
  }

  int64_t num_blocks = -1;
  if (kv_bytes >= 0) {
    num_blocks = kv_bytes / per_block;
  } else if (cache.kv_cache_memory_bytes > 0) {
    num_blocks = cache.kv_cache_memory_bytes / per_block;
  } else {
    num_blocks = cache.num_kv_blocks;
  }
  if (num_blocks <= 0) {
    return ResourceExhaustedError("KV budget of ", kv_bytes,
                                  " bytes fits zero blocks (one block is ", per_block, ")");
  }
  INFERX_ASSIGN_OR_RETURN(
      auto pool, KvBlockPool::Create(num_blocks, cache.block_size, spec.state.paged_kv,
                                     spec.device));
  kv_pool_.emplace(std::move(pool));
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
    state.blocks = fresh.block_ids;
    state.num_computed = fresh.num_computed_tokens;
    state.params = fresh.sampling_params;
    requests_[fresh.request_id] = std::move(state);
  }
  for (const CachedRequestUpdate& update : step.scheduled_cached_reqs) {
    auto it = requests_.find(update.request_id);
    if (it == requests_.end()) {
      return fail(InternalError("scheduler updated request ", update.request_id,
                                " that was never introduced"));
    }
    it->second.blocks.insert(it->second.blocks.end(), update.new_block_ids.begin(),
                             update.new_block_ids.end());
  }

  if (step.scheduled.empty()) return OkStatus();
  const int64_t block_size = kv_pool_->BlockSize();

  // Flatten the step: token stream, positions, per-sequence CSRs, and the
  // concatenated page tables. `kv_len` includes this step's new tokens --
  // their cache slots are written during this forward.
  std::vector<int32_t> input_ids, positions, batch_indices, kv_indices, last_page_len,
      sample_rows;
  std::vector<int32_t> qo_indptr{0}, kv_indptr{0};
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
    const int seq = static_cast<int>(batch.size());
    for (int64_t t = state.num_computed; t < end; ++t) {
      input_ids.push_back(state.tokens[t]);
      positions.push_back(static_cast<int32_t>(t));
      batch_indices.push_back(seq);
    }
    sample_rows.push_back(static_cast<int32_t>(total_new + scheduled.num_new_tokens - 1));
    total_new += scheduled.num_new_tokens;

    const int64_t pages = (end + block_size - 1) / block_size;
    if (pages > static_cast<int64_t>(state.blocks.size())) {
      return fail(InternalError("request ", scheduled.request_id, " needs ", pages,
                                " KV pages but holds ", state.blocks.size()));
    }
    kv_indices.insert(kv_indices.end(), state.blocks.begin(),
                      state.blocks.begin() + pages);
    qo_indptr.push_back(static_cast<int32_t>(total_new));
    kv_indptr.push_back(static_cast<int32_t>(kv_indices.size()));
    last_page_len.push_back(static_cast<int32_t>((end - 1) % block_size + 1));

    batch.push_back(&state);
    prompt_lens.push_back(static_cast<int64_t>(state.tokens.size()) - state.rng_offset);
  }

  // Device mirrors of the plan, then the attention kernel selection pass.
  INFERX_ASSIGN_OR_RETURN(Tensor ids_tensor, UploadInt32(*runtime_, stream_, input_ids));
  INFERX_ASSIGN_OR_RETURN(Tensor positions_tensor, UploadInt32(*runtime_, stream_, positions));
  INFERX_ASSIGN_OR_RETURN(Tensor batch_tensor, UploadInt32(*runtime_, stream_, batch_indices));
  INFERX_ASSIGN_OR_RETURN(Tensor qo_tensor, UploadInt32(*runtime_, stream_, qo_indptr));
  INFERX_ASSIGN_OR_RETURN(Tensor kv_indptr_tensor, UploadInt32(*runtime_, stream_, kv_indptr));
  INFERX_ASSIGN_OR_RETURN(Tensor kv_indices_tensor, UploadInt32(*runtime_, stream_, kv_indices));
  INFERX_ASSIGN_OR_RETURN(Tensor last_page_tensor, UploadInt32(*runtime_, stream_, last_page_len));
  INFERX_ASSIGN_OR_RETURN(Tensor sample_rows_tensor, UploadInt32(*runtime_, stream_, sample_rows));

  StepInputs inputs;
  inputs.token_ids = std::move(ids_tensor);
  inputs.positions = std::move(positions_tensor);
  AttentionMetadata attn;
  attn.batch_indices = std::move(batch_tensor);
  attn.qo_indptr = std::move(qo_tensor);
  attn.kv_indptr = std::move(kv_indptr_tensor);
  attn.kv_indices = std::move(kv_indices_tensor);
  attn.last_page_len = std::move(last_page_tensor);
  attn.host_qo_indptr = absl::MakeConstSpan(qo_indptr);
  attn.num_tokens = static_cast<int>(total_new);
  attn.num_seqs = static_cast<int>(batch.size());

  INFERX_RETURN_IF_ERROR(ops::BeginAttentionStep(
      *op_, *attn.kv_indptr, *attn.last_page_len, block_size, attn.num_tokens,
      attn.num_seqs, attention_plan_));

  const ModelSpec& spec = model_->GetModelSpec();
  ForwardContext ctx(*op_, attn, *kv_pool_, *comm_, *scratch_, attention_plan_);
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
  return OkStatus();
}

}  // namespace models
}  // namespace inferx
