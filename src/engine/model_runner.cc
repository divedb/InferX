#include "inferx/engine/model_runner.h"

#include <algorithm>
#include <future>
#include <limits>
#include <utility>

#include "inferx/dist/nccl_comm.h"
#include "inferx/engine/worker.h"
#include "inferx/models/registry.h"
#include "inferx/ops/attention/cache_spec_bridge.h"
#include "inferx/ops/attention/op.h"
#include "inferx/ops/attention/registry.h"
#include "inferx/util/hf_config.h"

namespace inferx {
using namespace models;

namespace {

/// Uploads one host int32 array as a device tensor.
StatusOr<Tensor> UploadInt32(DeviceRuntime& runtime, DeviceStream stream,
                             const std::vector<int32_t>& values) {
  INFERX_ASSIGN_OR_RETURN(
      Tensor tensor,
      Tensor::Empty(DataType::kInt32, Shape({static_cast<int64_t>(values.size())}),
                    runtime.device()));
  if (values.empty()) return tensor;
  INFERX_RETURN_IF_ERROR(runtime.Memory().CopyAsync(tensor.Data(), values.data(),
                                                    values.size() * sizeof(int32_t),
                                                    CopyKind::kHostToDevice, stream));
  return tensor;
}

}  // namespace

ModelRunner::ModelRunner(std::unique_ptr<models::Model> model, DeviceRuntime* runtime,
                         DeviceStream stream, CommBackend* comm, bool is_driver_rank)
    : runtime_(runtime),
      stream_(stream),
      comm_(comm),
      is_driver_rank_(is_driver_rank),
      model_(std::move(model)) {
  op_ = std::make_unique<ops::OpContext>(*runtime_, stream_);
}

ModelRunner::~ModelRunner() = default;

Status ModelRunner::LoadModel(const models::ModelLoadOptions& options) {
  if (!model_) return FailedPreconditionError("LoadModel requires a local runner");
  INFERX_ASSIGN_OR_RETURN(auto checkpoint, models::CheckpointReader::Open(options.directory));
  return model_->LoadWeights(checkpoint);
}

Status ModelRunner::ProfileRun(const SchedulerConfig& scheduler) {
  if (!model_) return FailedPreconditionError("ProfileRun requires a local runner");
  if (scheduler.max_num_seqs <= 0 || scheduler.max_num_batched_tokens <= 0)
    return InvalidArgumentError("profile requires positive sequence and token limits");
  max_num_seqs_ = scheduler.max_num_seqs;
  const auto& spec = model_->GetModelSpec();
  ScratchArena arena = ScratchArena::Measuring(spec.device);
  INFERX_RETURN_IF_ERROR(
      RunProfileStep(arena, scheduler.max_num_batched_tokens,
                     std::min<int64_t>(max_num_seqs_, scheduler.max_num_batched_tokens)));
  INFERX_RETURN_IF_ERROR(arena.Freeze(spec.device, 0.25));
  scratch_ = std::make_unique<ScratchArena>(std::move(arena));
  if (is_driver_rank_) {
    INFERX_ASSIGN_OR_RETURN(
        sampler_, sampling::Sampler::Create(max_num_seqs_, spec.vocab_size, spec.device));
  }
  return OkStatus();
}

Status ModelRunner::ApplyAcceptedTokens(RequestId id, TokenId token) {
  if (!input_batch_) return FailedPreconditionError("local input batch is not initialized");
  INFERX_RETURN_IF_ERROR(input_batch_->AppendToken(id, token));
  input_batch_->AdvanceRngOffset(*input_batch_->RowOf(id));
  return OkStatus();
}

StatusOr<ModelRunnerOutput> ModelRunner::ExecuteModel(const SchedulerOutput& step) {
  if (!workers_.empty()) return ExecuteGroup(step);
  if (!attention_ || !input_batch_) return FailedPreconditionError("runner is not initialized");
  ModelRunnerOutput output;
  INFERX_RETURN_IF_ERROR(RunStep(step, &output));
  return output;
}

Status ModelRunner::RunProfileStep(Allocator& scratch, int64_t max_tokens, int64_t max_seqs) {
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
  INFERX_ASSIGN_OR_RETURN(
      std::unique_ptr<attn::AttentionRuntime> runtime,
      attn::AttentionRuntime::Create(std::move(registry), config, model_->GetAttentionSpecs()));

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
    const int64_t chunk =
        tokens_per_seq + (request_id <= static_cast<uint64_t>(max_tokens % max_seqs) ? 1 : 0);
    INFERX_RETURN_IF_ERROR(runtime->Cache().Reserve(request_id, GrowthTarget{0, chunk}));
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
  INFERX_ASSIGN_OR_RETURN(inputs.token_ids, UploadInt32(*runtime_, stream_, token_ids));
  std::vector<std::int32_t> positions(static_cast<size_t>(token_ids.size()), 0);
  size_t offset = 0;
  for (int32_t len : query_lens)
    for (int32_t pos = 0; pos < len; ++pos) positions[offset++] = pos;
  INFERX_ASSIGN_OR_RETURN(inputs.positions, UploadInt32(*runtime_, stream_, positions));

  // The profile runs COMM-FREE through the world-of-one backend: tensor-
  // parallel ranks initialize sequentially (CreateGroup builds them one by
  // one), so a real collective here would deadlock the loopback barrier.
  // The measurements -- arena bytes, cuBLAS workspaces -- are identical
  // either way; collective backends allocate nothing from this arena.
  SingleRankComm profile_comm;
  ForwardContext ctx(*op_, step, profile_comm, scratch, absl::MakeConstSpan(ops));
  INFERX_ASSIGN_OR_RETURN(
      Tensor hidden,
      Tensor::Empty(spec.dtype,
                    Shape({static_cast<int64_t>(token_ids.size()), spec.hidden_size}),
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
                      Shape({static_cast<int64_t>(sample_rows.size()), spec.hidden_size}),
                      &scratch));
    INFERX_ASSIGN_OR_RETURN(
        Tensor logits,
        Tensor::Empty(spec.dtype,
                      Shape({static_cast<int64_t>(sample_rows.size()), spec.vocab_size}),
                      &scratch));
    (void)sampled;
    (void)logits;
  }
  return runtime_->Streams().SynchronizeStream(stream_);
  // `runtime` dies here: its bytes return to the device before
  // CreateAttention sizes the real pools.
}

Status ModelRunner::InitializeKVCache(int64_t kv_bytes, const CacheConfig& cache,
                                      const SchedulerConfig& scheduler) {
  if (!scratch_) return FailedPreconditionError("cache initialization requires profiling");
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
  INFERX_ASSIGN_OR_RETURN(
      attention_,
      attn::AttentionRuntime::Create(std::move(registry), config, model_->GetAttentionSpecs()));

  ops_.clear();
  const std::vector<attn::AttentionSpec> specs = model_->GetAttentionSpecs();
  for (size_t l = 0; l < specs.size(); ++l) {
    INFERX_ASSIGN_OR_RETURN(attn::AttentionOp op, attention_->MakeOp(static_cast<int64_t>(l)));
    ops_.push_back(std::move(op));
  }
  input_batch_ = std::make_unique<InputBatch>(max_num_seqs_, config.max_model_len + 1);
  return OkStatus();
}

Status ModelRunner::CaptureGraphs(const ExecutionConfig& execution) const {
  if (execution.enable_cuda_graphs) {
    return UnimplementedError("CUDA graph replay is not wired into the new model layer yet");
  }
  return OkStatus();
}

Status ModelRunner::RunStep(const SchedulerOutput& step, ModelRunnerOutput* out) {
  for (RequestId id : step.finished_request_ids) input_batch_->Remove(id);
  for (RequestId id : step.preempted_request_ids) input_batch_->Remove(id);
  for (const auto& fresh : step.scheduled_new_reqs) {
    InputBatch::NewRequest req;
    req.id = fresh.request_id;
    req.prompt_tokens.assign(fresh.prompt_token_ids.begin(), fresh.prompt_token_ids.end());
    req.num_computed = fresh.num_computed_tokens;
    req.params = fresh.sampling_params;
    INFERX_RETURN_IF_ERROR(input_batch_->Add(req));
  }
  // CachedRequestUpdate's watermark includes this step, so advance it only
  // after execution. Accepted peer tokens arrive through ApplyAcceptedTokens.
  input_batch_->Condense();

  if (step.scheduled.empty()) return OkStatus();
  const int64_t block_size = attention_->GetPlan().block_size;
  const std::uint16_t num_groups = attention_->NumGroups();

  // Flatten the step: token stream and positions. Page tables are NOT
  // mirrored here -- they are read from the cache facade below, the single
  // owner (docs/attention.md §8.3).
  std::vector<int32_t> input_ids, positions, sample_rows;
  std::vector<int32_t> query_lens, seq_lens;
  std::vector<int> batch;
  std::vector<int64_t> prompt_lens;
  int64_t total_new = 0;

  for (const ScheduledRequest& scheduled : step.scheduled) {
    auto found = input_batch_->RowOf(scheduled.request_id);
    if (!found)
      return InternalError("scheduled request has no runner state: ", scheduled.request_id);
    const int row = *found;
    const int64_t end = input_batch_->NumComputed(row) + scheduled.num_new_tokens;
    if (scheduled.num_new_tokens <= 0 || end > input_batch_->NumTokens(row))
      return InternalError("scheduled token range exceeds request history");
    for (int64_t t = input_batch_->NumComputed(row); t < end; ++t) {
      input_ids.push_back(input_batch_->TokensAt(row)[t]);
      positions.push_back(static_cast<int32_t>(t));
    }
    sample_rows.push_back(static_cast<int32_t>(total_new + scheduled.num_new_tokens - 1));
    query_lens.push_back(static_cast<std::int32_t>(scheduled.num_new_tokens));
    seq_lens.push_back(static_cast<std::int32_t>(end));
    total_new += scheduled.num_new_tokens;
    batch.push_back(row);
    prompt_lens.push_back(input_batch_->NumTokens(row));
  }

  INFERX_ASSIGN_OR_RETURN(Tensor ids_tensor, UploadInt32(*runtime_, stream_, input_ids));
  INFERX_ASSIGN_OR_RETURN(Tensor positions_tensor, UploadInt32(*runtime_, stream_, positions));
  INFERX_ASSIGN_OR_RETURN(Tensor sample_rows_tensor,
                          UploadInt32(*runtime_, stream_, sample_rows));

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

  if (std::getenv("INFERX_PARITY_BLOCKS")) {
    std::fprintf(stderr, "RSTEP");
    for (size_t i = 0; i < batch.size(); ++i) {
      const int row = batch[i];
      std::fprintf(stderr, " req=%llu nc=%d nt=%d q=%d s=%d blocks=",
                   (unsigned long long)input_batch_->IdAt(row), input_batch_->NumComputed(row),
                   input_batch_->NumTokens(row), query_lens[i], seq_lens[i]);
      const auto& table = step.group_block_ids[0];
      size_t begin = 0;
      for (size_t j = 0; j < i; ++j)
        begin += static_cast<size_t>((seq_lens[j] + block_size - 1) / block_size);
      const size_t count = static_cast<size_t>((seq_lens[i] + block_size - 1) / block_size);
      for (size_t j = 0; j < count; ++j)
        std::fprintf(stderr, "%d,", table[begin + j]);
      size_t tok_begin = 0;
      for (size_t j = 0; j < i; ++j) tok_begin += static_cast<size_t>(query_lens[j]);
      std::fprintf(stderr, " ids=[%d,%d..%d]", input_ids[tok_begin],
                   input_ids[tok_begin + 1], input_ids[tok_begin + static_cast<size_t>(query_lens[i]) - 1]);
    }
    std::fprintf(stderr, "\n");
  }
  INFERX_ASSIGN_OR_RETURN(attn::StepContext step_ctx, attention_->BeginStep(desc, *op_));

  const ModelSpec& spec = model_->GetModelSpec();
  ForwardContext ctx(*op_, step_ctx, *comm_, *scratch_, absl::MakeConstSpan(ops_));
  INFERX_ASSIGN_OR_RETURN(
      Tensor hidden,
      Tensor::Empty(spec.dtype, Shape({static_cast<int64_t>(total_new), spec.hidden_size}),
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
    std::vector<std::vector<TokenId>> histories(batch.size());
    for (size_t i = 0; i < batch.size(); ++i) {
      const int batch_row = batch[i];
      histories[i].assign(
          input_batch_->TokensAt(batch_row),
          input_batch_->TokensAt(batch_row) + input_batch_->NumTokens(batch_row));
      sampling::SamplingMetadata::PerRequest row;
      row.params = &input_batch_->ParamsAt(batch_row);
      row.rng_offset = input_batch_->RngOffset(batch_row);
      row.history = &histories[i];
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
    const int row = batch[i];
    const int computed = input_batch_->NumComputed(row) + step.scheduled[i].num_new_tokens;
    INFERX_RETURN_IF_ERROR(input_batch_->SetNumComputed(input_batch_->IdAt(row), computed));
    SampledTokens sample;
    sample.request_id = step.scheduled[i].request_id;
    if (sampled[i] >= 0 && computed >= prompt_lens[i]) {
      sample.token_ids.push_back(sampled[i]);
      INFERX_RETURN_IF_ERROR(ApplyAcceptedTokens(sample.request_id, sampled[i]));
    }
    out->samples.push_back(std::move(sample));
  }
  // Prefix publication is the scheduler's job (UpdateFromOutput): the
  // facade is single-threaded with it (docs/attention.md §9).
  return OkStatus();
}

}  // namespace inferx

namespace inferx {
ModelRunner::ModelRunner() = default;
attn::AttentionRuntime* ModelRunner::Attention() {
  return workers_.empty() ? attention_.get() : workers_.front()->Runner()->Attention();
}
const models::ModelSpec& ModelRunner::GetModelSpec() const {
  return workers_.empty() ? model_->GetModelSpec() : workers_.front()->Runner()->GetModelSpec();
}
void ModelRunner::EnableLogitsCapture() {
  if (workers_.empty())
    capture_logits_ = true;
  else
    workers_.front()->Runner()->EnableLogitsCapture();
}
const std::vector<Tensor>& ModelRunner::CapturedLogits() const {
  return workers_.empty() ? captured_logits_ : workers_.front()->Runner()->CapturedLogits();
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(const ModelConfig& model,
                                                           const CacheConfig& cache,
                                                           const SchedulerConfig& scheduler,
                                                           const ExecutionConfig& execution,
                                                           const ParallelConfig& parallel,
                                                           std::unique_ptr<CommBackend> comm) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  if (comm || parallel.tensor_parallel_size == 1) {
    INFERX_ASSIGN_OR_RETURN(auto worker, Worker::Create(model, parallel, std::move(comm)));
    std::vector<std::unique_ptr<Worker>> ranks;
    ranks.push_back(std::move(worker));
    return CreateGroup(std::move(ranks), cache, scheduler, execution, [](const Status&) {});
  }

  // Multi-rank world construction: one rank-local runner per device.
  if (parallel.tensor_parallel_rank != 0) {
    return InvalidArgumentError("the tensor-parallel coordinator must have rank zero");
  }
  if (model.device.device != "cuda") {
    return InvalidArgumentError("multi-GPU tensor parallelism requires CUDA");
  }
  if (execution.enable_cuda_graphs) {
    return UnimplementedError("tensor-parallel serving currently requires eager execution");
  }
  std::vector<int> ids = model.device.device_ids;
  if (ids.empty()) {
    for (int rank = 0; rank < parallel.tensor_parallel_size; ++rank) ids.push_back(rank);
  }
  if (ids.size() != static_cast<size_t>(parallel.tensor_parallel_size)) {
    return InvalidArgumentError("device-ids must contain one device per tensor-parallel rank");
  }
  std::vector<Device> devices;
  for (const int id : ids) {
    if (id > std::numeric_limits<int8_t>::max()) {
      return InvalidArgumentError("CUDA device ordinal is out of range");
    }
    devices.push_back(Device::Cuda(static_cast<int8_t>(id)));
  }
  // Reject unknown architectures before creating the communication world.
  INFERX_ASSIGN_OR_RETURN(auto hf, HfConfig::FromFile(model.model_dir + "/config.json"));
  INFERX_RETURN_IF_ERROR(models::ArchitectureRegistry::Builtins()
                             .Resolve(hf.ModelType(), hf.Architecture())
                             .status());
  INFERX_ASSIGN_OR_RETURN(auto world, NcclWorld::Create(devices));
  std::vector<std::unique_ptr<Worker>> ranks;
  for (int rank = 0; rank < parallel.tensor_parallel_size; ++rank) {
    ModelConfig local = model;
    local.device.device_ids = {ids[rank]};
    ParallelConfig rank_parallel;
    rank_parallel.tensor_parallel_size = parallel.tensor_parallel_size;
    rank_parallel.tensor_parallel_rank = rank;
    auto made = Worker::Create(local, rank_parallel, world->MakeRank(rank));
    if (!made.ok()) {
      world->Abort(made.status());
      return made.status();
    }
    ranks.push_back(*std::move(made));
  }
  return CreateGroup(
      std::move(ranks), cache, scheduler, execution,
      [world](const Status& status) { world->Abort(status); },
      [world] { return world->CheckHealth(); });
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::CreateGroup(
    std::vector<std::unique_ptr<Worker>> ranks, const CacheConfig& cache,
    const SchedulerConfig& scheduler, const ExecutionConfig& execution,
    std::function<void(const Status&)> abort, std::function<Status()> check_health) {
  if (ranks.empty() || !abort)
    return InvalidArgumentError("rank group needs workers and abort");
  for (size_t i = 0; i < ranks.size(); ++i) {
    if (!ranks[i]) return InvalidArgumentError("rank group contains a null worker");
    if (ranks.size() > 1 &&
        (ranks[i]->Parallel().tensor_parallel_rank != static_cast<int>(i) ||
         ranks[i]->Parallel().tensor_parallel_size != static_cast<int>(ranks.size())))
      return InvalidArgumentError("rank group topology disagrees");
  }
  auto runner = std::unique_ptr<ModelRunner>(new ModelRunner());
  runner->workers_ = std::move(ranks);
  runner->abort_ = std::move(abort);
  runner->check_health_ = std::move(check_health);
  auto initialize = [&]() -> Status {
    int64_t budget = std::numeric_limits<int64_t>::max();
    for (auto& worker : runner->workers_) {
      worker->SetFailureHooks(runner->abort_, runner->check_health_);
      INFERX_RETURN_IF_ERROR(worker->InitDevice());
      INFERX_RETURN_IF_ERROR(worker->LoadModel());
      INFERX_ASSIGN_OR_RETURN(auto available,
                              worker->DetermineAvailableMemory(cache, scheduler));
      budget = std::min(budget, available);
    }
    const auto& driver = runner->workers_.front()->Runner()->GetModelSpec();
    for (const auto& worker : runner->workers_) {
      const auto& spec = worker->Runner()->GetModelSpec();
      if (spec.architecture != driver.architecture || spec.vocab_size != driver.vocab_size ||
          spec.hidden_size != driver.hidden_size || spec.num_layers != driver.num_layers ||
          spec.max_position_embeddings != driver.max_position_embeddings ||
          spec.dtype != driver.dtype || spec.eos_token_ids != driver.eos_token_ids)
        return InvalidArgumentError("rank group model specifications disagree");
    }
    if (runner->workers_.size() == 1 && budget >= 0 &&
        runner->workers_.front()->Parallel().tensor_parallel_size > 1) {
      INFERX_ASSIGN_OR_RETURN(budget, runner->workers_.front()->AgreeAvailableMemory(budget));
    }
    // All local ranks have profiled before cache allocation. No collective
    // runs during sequential initialization.
    for (auto& worker : runner->workers_) {
      INFERX_RETURN_IF_ERROR(worker->InitializeFromConfig(budget, cache, scheduler));
      INFERX_RETURN_IF_ERROR(worker->WarmUp(execution));
    }
    return OkStatus();
  };
  const Status status = initialize();
  if (!status.ok()) {
    runner->abort_(status);
    return status;
  }
  return runner;
}

StatusOr<ModelRunnerOutput> ModelRunner::ExecuteGroup(const SchedulerOutput& output) {
  if (check_health_) INFERX_RETURN_IF_ERROR(check_health_());
  if (workers_.size() == 1) return workers_.front()->ExecuteModel(output);

  // Dispatch the same logical step to every rank. Each rank's runner
  // validates and binds its rank-local state; a failure on any rank
  // poisons the group rather than retrying mutated state.
  std::vector<std::future<StatusOr<ModelRunnerOutput>>> pending;
  try {
    pending.reserve(workers_.size() - 1);
    for (size_t rank = 1; rank < workers_.size(); ++rank) {
      pending.push_back(std::async(
          std::launch::async,
          [runner = workers_[rank].get(), &output] { return runner->ExecuteModel(output); }));
    }
  } catch (const std::exception& error) {
    const Status failure = InternalError("could not start rank execution: ", error.what());
    abort_(failure);
    return failure;  // Futures join before the borrowed output expires.
  }
  auto result = workers_.front()->ExecuteModel(output);
  Status failure = result.status();
  for (auto& future : pending) {
    auto peer_result = future.get();
    if (failure.ok() && !peer_result.ok()) failure = peer_result.status();
  }
  if (!failure.ok()) {
    abort_(failure);
    return failure;
  }
  // Only rank zero draws tokens. Share those exact ids with every rank,
  // including stochastic sampling, before admitting the next scheduler step.
  for (const auto& sample : result->samples) {
    if (sample.token_ids.empty()) continue;
    for (size_t rank = 1; rank < workers_.size(); ++rank) {
      const auto status = workers_[rank]->Runner()->ApplyAcceptedTokens(
          sample.request_id, sample.token_ids.back());
      if (!status.ok()) {
        abort_(status);
        return status;
      }
    }
  }
  return result;
}

}  // namespace inferx
