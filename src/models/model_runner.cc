#include "inferx/models/model_runner.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/logging.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/nccl_comm.h"
#include "inferx/ops/execution_context.h"
#include "inferx/sampling/sampler.h"

namespace inferx {
namespace {

struct RunnerRequestState {
  std::vector<int32_t> block_ids;
  std::vector<TokenId> prompt;
  sampling::SamplingParams params;
  uint64_t generated = 0;  ///< Tokens sampled so far; the request's RNG offset.
  int num_computed = 0;
  TokenId last_sampled = -1;
  /// Recurrent-state slot; -1 while unassigned or when unneeded.
  int recurrent_slot = -1;
};

}  // namespace

struct ModelRunnerImpl {
  struct DecodeGraph {
    GraphExec exec;
    std::optional<Tensor> logits;
    std::optional<sampling::SamplerOutput> output;
    bool warmed = false;
  };

  ModelConfig config;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  DeviceId device;  ///< Resolved once from config.device.PrimaryDevice().
  DeviceRuntime* runtime = nullptr;
  Stream stream;
  std::unique_ptr<Model> model;
  std::unique_ptr<dist::CommBackend> comm;
  std::unique_ptr<KvBlockPool> pool;
  std::unique_ptr<RecurrentStatePool> recurrent_pool;
  /// Free recurrent slots, LIFO; empty when the model has no recurrent layers.
  std::vector<int32_t> free_recurrent_slots;
  ModelState model_state;
  absl::flat_hash_map<RequestId, RunnerRequestState> states;
  std::optional<Tensor> token_ids, positions, batch_indices, qo_indptr, kv_indptr;
  std::optional<Tensor> kv_indices, last_page_len, logit_rows, recurrent_indices;
  std::optional<Tensor> input_storage;
  int32_t* host_inputs = nullptr;
  std::unique_ptr<sampling::Sampler> sampler;
  int32_t* host_samples = nullptr;
  absl::flat_hash_map<int, DecodeGraph> decode_graphs;
  Status failure;
  bool sample_output = true;
  std::function<void(const Status&)> abort;
  std::function<Status()> check_health;
  DeviceEvent completion;
  std::vector<std::unique_ptr<ModelRunner>> peers;

  Status WaitForStream() {
    if (!check_health) return runtime->SynchronizeStream(stream);
    INFERX_RETURN_IF_ERROR(runtime->RecordEvent(completion, stream));

    while (true) {
      INFERX_RETURN_IF_ERROR(check_health());
      INFERX_ASSIGN_OR_RETURN(bool done, runtime->QueryEvent(completion));

      if (done) return OkStatus();

      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  ~ModelRunnerImpl() {
    if (runtime == nullptr) return;

    (void)runtime->Activate();

    // Drain queued work even after an operation failed before the runner's sync.
    if (stream.handle != nullptr) (void)runtime->SynchronizeStream(stream);

    for (auto& [batch, graph] : decode_graphs) {
      if (graph.exec.handle != nullptr) (void)runtime->DestroyGraph(graph.exec);
    }

    if (host_samples != nullptr) (void)runtime->FreePinnedHost(host_samples);
    if (host_inputs != nullptr) (void)runtime->FreePinnedHost(host_inputs);
    if (completion.handle != nullptr) (void)runtime->DestroyEvent(completion);
    if (stream.handle != nullptr) (void)runtime->DestroyStream(stream);
  }

  StatusOr<ModelRunnerOutput> Execute(const SchedulerOutput& output);
};

ModelRunner::ModelRunner(std::unique_ptr<ModelRunnerImpl> impl) : impl_(std::move(impl)) {}
ModelRunner::~ModelRunner() = default;
KvBlockPool* ModelRunner::kv_pool() { return impl_->pool.get(); }
const CheckpointConfig& ModelRunner::checkpoint_config() const {
  return impl_->model->config();
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(
    const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
    const ExecutionConfig& execution, const ParallelConfig& parallel,
    std::unique_ptr<dist::CommBackend> comm) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  if (!comm && parallel.tensor_parallel_size > 1) {
    if (parallel.tensor_parallel_rank != 0) {
      return InvalidArgumentError("the tensor-parallel coordinator must have rank zero");
    }
    if (model.device.device_type != "cuda") {
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
      return InvalidArgumentError(
          "device-ids must contain one device per tensor-parallel rank");
    }
    std::vector<DeviceId> devices;
    for (const int id : ids) {
      if (id > std::numeric_limits<int8_t>::max()) {
        return InvalidArgumentError("CUDA device ordinal is out of range");
      }
      devices.push_back(DeviceId::Cuda(static_cast<int8_t>(id)));
    }
    INFERX_ASSIGN_OR_RETURN(auto checkpoint,
                            CheckpointConfig::FromFile(model.model_dir + "/config.json"));
    if (checkpoint.model_type != "qwen3" || checkpoint.architectures != "Qwen3ForCausalLM") {
      return UnimplementedError("multi-GPU serving currently supports dense Qwen3 checkpoints");
    }
    const int size = parallel.tensor_parallel_size;
    if (checkpoint.num_attention_heads <= 0 || checkpoint.num_key_value_heads <= 0 ||
        checkpoint.intermediate_size <= 0 || checkpoint.vocab_size <= 0 ||
        checkpoint.num_attention_heads % size != 0 ||
        checkpoint.intermediate_size % size != 0 || checkpoint.vocab_size % size != 0 ||
        (checkpoint.num_key_value_heads >= size ? checkpoint.num_key_value_heads % size != 0
                                                : size % checkpoint.num_key_value_heads != 0)) {
      return InvalidArgumentError("Qwen3 dimensions do not support this tensor-parallel size");
    }
    INFERX_ASSIGN_OR_RETURN(auto world, dist::NcclWorld::Create(devices));
    std::vector<std::unique_ptr<ModelRunner>> ranks;
    for (int rank = 0; rank < parallel.tensor_parallel_size; ++rank) {
      ModelConfig local = model;
      local.device.device_ids = {ids[rank]};
      auto made =
          Create(local, cache, scheduler, execution,
                 ParallelConfig{parallel.tensor_parallel_size, rank}, world->MakeRank(rank));
      if (!made.ok()) {
        world->Abort(made.status());
        return made.status();
      }
      ranks.push_back(*std::move(made));
    }
    return CreateGroup(
        std::move(ranks), [world](const Status& status) { world->Abort(status); },
        [world] { return world->CheckHealth(); });
  }
  if (!comm) comm = std::make_unique<dist::SingleRankComm>();
  if (comm->size() != parallel.tensor_parallel_size ||
      comm->rank() != parallel.tensor_parallel_rank) {
    return InvalidArgumentError("runner communicator must match the model's parallel topology");
  }
  if (scheduler.max_num_batched_tokens <= 0 || scheduler.max_num_seqs <= 0) {
    return InvalidArgumentError("runner requires positive token and sequence capacities");
  }
  INFERX_ASSIGN_OR_RETURN(
      auto loaded,
      Model::Load(model.model_dir, model.device.PrimaryDevice(),
                  scheduler.max_num_batched_tokens, scheduler.max_num_seqs, parallel));

  return Create(model, cache, scheduler, execution, std::move(loaded), std::move(comm));
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(
    const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
    const ExecutionConfig& execution, std::unique_ptr<Model> loaded,
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
  const auto requirements = loaded->StateRequirements();
  if (requirements.empty() ||
      requirements.size() != static_cast<size_t>(loaded->config().num_hidden_layers)) {
    return InvalidArgumentError("model must declare state for each decoder layer");
  }
  std::vector<KvLayout> layouts;
  std::vector<RecurrentStateSpec> recurrent_specs;
  for (size_t i = 0; i < requirements.size(); ++i) {
    if (const auto* spec = std::get_if<PagedKvStateSpec>(&requirements[i])) {
      layouts.push_back(spec->layout);
    } else {
      recurrent_specs.push_back(std::get<RecurrentStateSpec>(requirements[i]));
    }
  }
  auto impl = std::make_unique<ModelRunnerImpl>();
  impl->config = model;
  impl->cache = cache;
  impl->scheduler = scheduler;
  impl->execution = execution;
  impl->model = std::move(loaded);
  impl->comm = std::move(comm);
  impl->device = device;
  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(impl->runtime->Activate());
  INFERX_ASSIGN_OR_RETURN(impl->stream, impl->runtime->CreateStream());
  const auto& mc = impl->model->config();
  INFERX_ASSIGN_OR_RETURN(
      auto pool, KvBlockPool::Create(cache.num_kv_blocks, cache.block_size, layouts, device));
  impl->pool = std::make_unique<KvBlockPool>(std::move(pool));
  if (!recurrent_specs.empty()) {
    INFERX_ASSIGN_OR_RETURN(
        auto recurrent,
        RecurrentStatePool::Create(recurrent_specs, scheduler.max_num_seqs, device));
    impl->recurrent_pool = std::make_unique<RecurrentStatePool>(std::move(recurrent));
    impl->free_recurrent_slots.resize(scheduler.max_num_seqs);
    for (int64_t slot = 0; slot < scheduler.max_num_seqs; ++slot) {
      impl->free_recurrent_slots[static_cast<size_t>(slot)] =
          static_cast<int32_t>(scheduler.max_num_seqs - 1 - slot);
    }
  }
  INFERX_LOG(INFO) << "runner ready: device=" << device.ToString()
                   << " model=" << model.model_dir << " kv_blocks=" << cache.num_kv_blocks
                   << " block_size=" << cache.block_size
                   << " max_tokens=" << scheduler.max_num_batched_tokens
                   << " max_seqs=" << scheduler.max_num_seqs;
  impl->model_state.paged_kv = impl->pool.get();
  impl->model_state.recurrent = impl->recurrent_pool.get();
  int64_t next_recurrent_layer = 0;
  int64_t next_paged_layer = 0;
  for (size_t i = 0; i < requirements.size(); ++i) {
    if (std::holds_alternative<PagedKvStateSpec>(requirements[i])) {
      impl->model_state.layers.push_back(PagedKvState{next_paged_layer++});
    } else {
      impl->model_state.layers.push_back(RecurrentState{next_recurrent_layer++});
    }
  }
  auto alloc = [&](int64_t size) {
    return Tensor::Empty(DataType::kInt32, Shape({size}), device);
  };
  const int64_t T = scheduler.max_num_batched_tokens;
  const int64_t S = scheduler.max_num_seqs;
  // Fixed offsets keep graph pointers stable. One pinned upload replaces eight
  // synchronous copies; Run drains the stream before this staging area is reused.
  INFERX_ASSIGN_OR_RETURN(impl->input_storage, alloc(3 * T + 5 * S + 2 + cache.num_kv_blocks));
  int64_t offset = 0;
  auto input_view = [&](int64_t size) -> StatusOr<Tensor> {
    INFERX_ASSIGN_OR_RETURN(auto view, impl->input_storage->Slice(offset, offset + size));
    offset += size;
    return view;
  };
  INFERX_ASSIGN_OR_RETURN(impl->token_ids, input_view(T));
  INFERX_ASSIGN_OR_RETURN(impl->positions, input_view(T));
  INFERX_ASSIGN_OR_RETURN(impl->batch_indices, input_view(T));
  INFERX_ASSIGN_OR_RETURN(impl->qo_indptr, input_view(S + 1));
  INFERX_ASSIGN_OR_RETURN(impl->kv_indptr, input_view(S + 1));
  INFERX_ASSIGN_OR_RETURN(impl->kv_indices, input_view(cache.num_kv_blocks));
  INFERX_ASSIGN_OR_RETURN(impl->last_page_len, input_view(S));
  INFERX_ASSIGN_OR_RETURN(impl->logit_rows, input_view(S));
  INFERX_ASSIGN_OR_RETURN(impl->recurrent_indices, input_view(S));
  if (device.IsCuda()) {
    INFERX_ASSIGN_OR_RETURN(void* inputs,
                            impl->runtime->AllocatePinnedHost(impl->input_storage->NBytes()));
    impl->host_inputs = static_cast<int32_t*>(inputs);
    std::memset(inputs, 0, impl->input_storage->NBytes());
    INFERX_ASSIGN_OR_RETURN(void* host, impl->runtime->AllocatePinnedHost(S * sizeof(int32_t)));
    impl->host_samples = static_cast<int32_t*>(host);
  }
  INFERX_ASSIGN_OR_RETURN(
      impl->sampler, sampling::Sampler::Create(scheduler.max_num_seqs, mc.vocab_size, device));
  return std::unique_ptr<ModelRunner>(new ModelRunner(std::move(impl)));
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::CreateGroup(
    std::vector<std::unique_ptr<ModelRunner>> ranks, std::function<void(const Status&)> abort,
    std::function<Status()> check_health) {
  if (ranks.empty() || !abort)
    return InvalidArgumentError("rank group needs runners and abort");
  for (size_t rank = 0; rank < ranks.size(); ++rank) {
    if (!ranks[rank]) return InvalidArgumentError("rank group contains a null runner");
    auto& local = *ranks[rank]->impl_;
    const auto& first = *ranks.front()->impl_;
    if (!local.peers.empty() || local.comm->rank() != static_cast<int>(rank) ||
        local.comm->size() != static_cast<int>(ranks.size()) ||
        local.scheduler.max_num_seqs != first.scheduler.max_num_seqs ||
        local.scheduler.max_num_batched_tokens != first.scheduler.max_num_batched_tokens ||
        local.cache.num_kv_blocks != first.cache.num_kv_blocks ||
        local.cache.block_size != first.cache.block_size ||
        local.model->config().vocab_size != first.model->config().vocab_size) {
      const auto status = InvalidArgumentError("rank group topology or capacities disagree");
      abort(status);
      return status;
    }
    if (local.execution.enable_cuda_graphs) {
      const auto status = UnimplementedError("rank groups currently require eager execution");
      abort(status);
      return status;
    }
    local.abort = abort;
    local.check_health = check_health;
    if (check_health) {
      auto event = local.runtime->CreateEvent(false);
      if (!event.ok()) {
        abort(event.status());
        return event.status();
      }
      local.completion = *event;
    }
    local.sample_output = rank == 0;
    if (rank != 0) local.sampler.reset();
  }
  auto coordinator = std::move(ranks.front());
  for (size_t rank = 1; rank < ranks.size(); ++rank) {
    coordinator->impl_->peers.push_back(std::move(ranks[rank]));
  }
  return coordinator;
}

StatusOr<ModelRunnerOutput> ModelRunnerImpl::Execute(const SchedulerOutput& output) {
  ModelRunnerOutput result;
  // Process removals before admission so a finished ID can safely be reused.
  ops::ExecutionContext lifecycle_ctx(*runtime, stream);
  for (RequestId id : output.finished_request_ids) {
    const auto it = states.find(id);
    if (it != states.end() && it->second.recurrent_slot >= 0) {
      free_recurrent_slots.push_back(it->second.recurrent_slot);
    }
    states.erase(id);
  }
  for (const auto& nr : output.scheduled_new_reqs) {
    RunnerRequestState state;
    state.block_ids = nr.block_ids;
    state.prompt.assign(nr.prompt_token_ids.begin(), nr.prompt_token_ids.end());
    state.params = nr.sampling_params;
    state.num_computed = nr.num_computed_tokens;
    if (recurrent_pool != nullptr) {
      if (free_recurrent_slots.empty()) {
        return ResourceExhaustedError("recurrent state slots are exhausted");
      }
      state.recurrent_slot = free_recurrent_slots.back();
      free_recurrent_slots.pop_back();
      // A fresh sequence starts from zero state; stream-ordered so the
      // step's kernels (and captured graphs) read zeros first.
      INFERX_RETURN_IF_ERROR(recurrent_pool->ResetSlot(lifecycle_ctx, state.recurrent_slot));
    }
    if (!states.emplace(nr.request_id, std::move(state)).second) {
      return InvalidArgumentError("duplicate runner request ", nr.request_id);
    }
  }
  absl::flat_hash_map<RequestId, int> scheduled_ends;
  for (const auto& cu : output.scheduled_cached_reqs) {
    auto it = states.find(cu.request_id);
    if (it == states.end()) return InternalError("unknown cached request ", cu.request_id);
    auto& blocks = it->second.block_ids;
    blocks.insert(blocks.end(), cu.new_block_ids.begin(), cu.new_block_ids.end());
    // Scheduler watermarks include this allocation. Keep the runner's start
    // position until Forward succeeds; never treat that end as the start.
    scheduled_ends[cu.request_id] = cu.num_computed_tokens;
  }
  const int batch = output.scheduled.size();
  if (batch == 0) return result;
  if (batch > scheduler.max_num_seqs) return InvalidArgumentError("batch exceeds max_num_seqs");

  bool pure_decode = true;
  std::vector<int32_t> tokens, positions_h, batches;
  std::vector<int32_t> qo{0}, kv{0}, blocks, last_lengths, rows;
  const auto& mc = model->config();
  for (int i = 0; i < batch; ++i) {
    const auto& sr = output.scheduled[i];
    auto it = states.find(sr.request_id);
    if (it == states.end()) return InternalError("unknown scheduled request ", sr.request_id);
    const auto& state = it->second;
    const int start = state.num_computed;
    const int chunk = sr.num_new_tokens;
    if (start < 0 || chunk <= 0 || chunk > scheduler.max_num_batched_tokens ||
        tokens.size() + chunk > static_cast<size_t>(scheduler.max_num_batched_tokens)) {
      return InvalidArgumentError("invalid token allocation for request ", sr.request_id);
    }
    const int64_t end = static_cast<int64_t>(start) + chunk;
    if (end > mc.max_position_embeddings || end > std::numeric_limits<int32_t>::max()) {
      return InvalidArgumentError("request ", sr.request_id, " exceeds model context length");
    }
    auto expected = scheduled_ends.find(sr.request_id);
    if (expected != scheduled_ends.end() && expected->second != end) {
      return InternalError("computed-token watermark mismatch for request ", sr.request_id);
    }
    const int64_t prompt_len = state.prompt.size();
    pure_decode = pure_decode && chunk == 1 && start >= prompt_len;
    if (end <= prompt_len) {
      tokens.insert(tokens.end(), state.prompt.begin() + start, state.prompt.begin() + end);
    } else {
      if (chunk != 1 || start < prompt_len || state.last_sampled < 0) {
        return InternalError("inconsistent decode allocation for request ", sr.request_id);
      }
      tokens.push_back(state.last_sampled);
    }
    for (int j = 0; j < chunk; ++j) {
      positions_h.push_back(start + j);
      batches.push_back(i);
    }
    qo.push_back(tokens.size());
    const int64_t pages = pool->BlocksForTokens(end);
    if (static_cast<int64_t>(state.block_ids.size()) < pages ||
        blocks.size() + pages > static_cast<size_t>(cache.num_kv_blocks)) {
      return InternalError("insufficient KV block capacity for request ", sr.request_id);
    }
    for (int64_t j = 0; j < pages; ++j) {
      const int32_t block = state.block_ids[j];
      if (block < 0 || block >= cache.num_kv_blocks)
        return InvalidArgumentError("invalid KV block");
      blocks.push_back(block);
    }
    kv.push_back(blocks.size());
    last_lengths.push_back((end - 1) % cache.block_size + 1);
    rows.push_back(tokens.size() - 1);
  }
  for (int32_t token : tokens) {
    if (token < 0 || token >= mc.vocab_size)
      return InvalidArgumentError("token outside vocabulary");
  }
  auto upload = [&](const Tensor& dst, const std::vector<int32_t>& src) {
    if (host_inputs != nullptr) {
      const auto offset = dst.DataAs<int32_t>() - input_storage->DataAs<int32_t>();
      std::memcpy(host_inputs + offset, src.data(), src.size() * sizeof(int32_t));
      return OkStatus();
    }
    return runtime->Copy(dst.Data(), src.data(), src.size() * sizeof(int32_t),
                         CopyKind::kHostToDevice);
  };
  INFERX_RETURN_IF_ERROR(upload(*token_ids, tokens));
  INFERX_RETURN_IF_ERROR(upload(*positions, positions_h));
  INFERX_RETURN_IF_ERROR(upload(*batch_indices, batches));
  INFERX_RETURN_IF_ERROR(upload(*qo_indptr, qo));
  INFERX_RETURN_IF_ERROR(upload(*kv_indptr, kv));
  INFERX_RETURN_IF_ERROR(upload(*kv_indices, blocks));
  INFERX_RETURN_IF_ERROR(upload(*last_page_len, last_lengths));
  INFERX_RETURN_IF_ERROR(upload(*logit_rows, rows));
  if (recurrent_pool != nullptr) {
    std::vector<int32_t> slots;
    slots.reserve(batch);
    for (int i = 0; i < batch; ++i) {
      const auto it = states.find(output.scheduled[i].request_id);
      if (it == states.end() || it->second.recurrent_slot < 0) {
        return InternalError("missing recurrent slot for a scheduled request");
      }
      slots.push_back(it->second.recurrent_slot);
    }
    INFERX_RETURN_IF_ERROR(upload(*recurrent_indices, slots));
  }
  if (host_inputs != nullptr) {
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(input_storage->Data(), host_inputs,
                                              input_storage->NBytes(), CopyKind::kHostToDevice,
                                              stream));
  }
  // ModelInput carries exact-size views of the capacity buffers; the padded
  // tails of the upload buffers are never visible to the model.
  INFERX_ASSIGN_OR_RETURN(Tensor token_ids_v, token_ids->Slice(0, tokens.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor positions_v, positions->Slice(0, tokens.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor batch_indices_v, batch_indices->Slice(0, tokens.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor qo_indptr_v, qo_indptr->Slice(0, batch + 1));
  INFERX_ASSIGN_OR_RETURN(Tensor kv_indptr_v, kv_indptr->Slice(0, batch + 1));
  INFERX_ASSIGN_OR_RETURN(Tensor kv_indices_v, kv_indices->Slice(0, blocks.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor last_page_len_v, last_page_len->Slice(0, batch));
  INFERX_ASSIGN_OR_RETURN(Tensor logit_rows_v, logit_rows->Slice(0, batch));
  std::optional<Tensor> recurrent_v;
  if (recurrent_pool != nullptr) {
    INFERX_ASSIGN_OR_RETURN(recurrent_v, recurrent_indices->Slice(0, batch));
  }
  ModelInput input{
      std::move(token_ids_v),
      AttentionBatch{std::move(positions_v), std::move(batch_indices_v), std::move(qo_indptr_v),
                     std::move(kv_indptr_v), std::move(kv_indices_v),
                     std::move(last_page_len_v), absl::MakeConstSpan(qo),
                     absl::MakeConstSpan(kv), static_cast<int>(tokens.size()), batch},
      std::move(logit_rows_v)};
  input.attention.recurrent_indices = std::move(recurrent_v);
  ops::ExecutionContext ctx(*runtime, stream);
  std::optional<Tensor> logits;
  std::optional<sampling::SamplerOutput> sampled;
  INFERX_VLOG(2) << "step: tokens=" << tokens.size() << " seqs=" << batch
                 << " pure_decode=" << pure_decode;
  // Per-request sampling knobs plus RNG offsets, resolved for this batch.
  std::vector<sampling::SamplingMetadata::PerRequest> per(batch);
  for (int i = 0; i < batch; ++i) {
    const RunnerRequestState& state = states.at(output.scheduled[i].request_id);
    per[i] = {&state.params, state.generated};
  }
  const sampling::SamplingMetadata metadata =
      sampling::SamplingMetadata::Build(mc.vocab_size, absl::MakeConstSpan(per));
  if (pure_decode && execution.enable_cuda_graphs && device.IsCuda() &&
      model->SupportsCudaGraphs()) {
    auto& graph = decode_graphs[batch];
    if (!graph.warmed) {
      // Let cuBLAS initialize algorithms/workspace for this exact shape before capture.
      INFERX_ASSIGN_OR_RETURN(logits, model->Forward(input, model_state, ctx, *comm));
      graph.warmed = true;
    } else {
      if (graph.exec.handle == nullptr) {
        INFERX_VLOG(1) << "capturing decode graph for batch " << batch;
        INFERX_RETURN_IF_ERROR(runtime->BeginCapture(stream));
        auto captured = model->Forward(input, model_state, ctx, *comm);
        absl::StatusOr<sampling::SamplerOutput> samples =
            captured.ok() ? sampler->Sample(ctx, *captured, metadata)
                          : absl::StatusOr<sampling::SamplerOutput>(captured.status());
        // End the capture even on failure, so teardown never leaves a stream capturing.
        auto instantiated = runtime->EndCaptureAndInstantiate(stream);
        if (!captured.ok() || !samples.ok()) {
          if (instantiated.ok()) (void)runtime->DestroyGraph(*instantiated);
          return captured.ok() ? samples.status() : captured.status();
        }
        INFERX_ASSIGN_OR_RETURN(graph.exec, std::move(instantiated));
        graph.output = *std::move(samples);
        graph.logits = *std::move(captured);
      }
      INFERX_RETURN_IF_ERROR(runtime->LaunchGraph(graph.exec, stream));
      logits = graph.logits;
      sampled = graph.output;  // Graph replay already produced the samples.
    }
  } else {
    INFERX_ASSIGN_OR_RETURN(logits, model->Forward(input, model_state, ctx, *comm));
    if (sample_output) {
      INFERX_ASSIGN_OR_RETURN(sampled, sampler->Sample(ctx, *logits, metadata));
    }
  }
  if (!sample_output) {
    INFERX_RETURN_IF_ERROR(WaitForStream());
    for (const auto& sr : output.scheduled) {
      auto& state = states.at(sr.request_id);
      state.num_computed += sr.num_new_tokens;
      ++state.generated;
    }
    return result;
  }
  // The graph warm-up iteration runs Forward eagerly without capture; sample
  // its logits through the normal path.
  if (!sampled.has_value()) {
    INFERX_ASSIGN_OR_RETURN(sampled, sampler->Sample(ctx, *logits, metadata));
  }
  if (logits->Rank() != 2 || logits->Dim(0) != batch ||
      (logits->GetDataType() != DataType::kBFloat16 &&
       logits->GetDataType() != DataType::kFloat32) ||
      logits->Device() != device) {
    return InternalError("model returned malformed logits for the scheduled batch");
  }
  if (sampled->sampled_token_ids.Dim(0) != batch)
    return InternalError("sampler returned malformed results for the scheduled batch");
  std::vector<int32_t> samples;
  if (host_samples != nullptr) {
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(host_samples, sampled->sampled_token_ids.Data(),
                                              batch * sizeof(int32_t), CopyKind::kDeviceToHost,
                                              stream));
    INFERX_RETURN_IF_ERROR(WaitForStream());
    samples.assign(host_samples, host_samples + batch);
  } else {
    // CPU device: tensors are host-accessible once the stream has drained.
    INFERX_RETURN_IF_ERROR(WaitForStream());
    const int32_t* ids = sampled->sampled_token_ids.DataAs<int32_t>();
    samples.assign(ids, ids + batch);
  }
  for (int i = 0; i < batch; ++i) {
    const auto& sr = output.scheduled[i];
    result.samples.push_back({sr.request_id, {samples[i]}, std::nullopt});
    auto& state = states.at(sr.request_id);
    state.num_computed += sr.num_new_tokens;
    ++state.generated;
    state.last_sampled = samples[i];
  }
  return result;
}

StatusOr<ModelRunnerOutput> ModelRunner::Run(const SchedulerOutput& output) {
  if (impl_->peers.empty()) return RunLocal(output);
  if (!impl_->failure.ok()) return FailedPreconditionError(impl_->failure.message());
  std::vector<std::future<StatusOr<ModelRunnerOutput>>> pending;
  try {
    for (const auto& peer : impl_->peers) {
      pending.push_back(std::async(
          std::launch::async, [rank = peer.get(), &output] { return rank->RunLocal(output); }));
    }
  } catch (const std::exception& error) {
    impl_->failure = InternalError("could not start rank execution: ", error.what());
    impl_->abort(impl_->failure);
    return impl_->failure;  // Futures join before the borrowed output expires.
  }
  auto result = RunLocal(output);
  Status failure = result.status();
  for (auto& future : pending) {
    auto peer_result = future.get();
    if (failure.ok() && !peer_result.ok()) failure = peer_result.status();
  }
  if (!failure.ok()) {
    impl_->failure = failure;
    impl_->abort(failure);
    return failure;
  }
  // Only rank zero draws a token. Share those exact ids with every rank,
  // including stochastic sampling, before admitting the next scheduler step.
  for (const auto& sample : result->samples) {
    for (auto& peer : impl_->peers) {
      peer->impl_->states.at(sample.request_id).last_sampled = sample.token_ids.back();
    }
  }
  return result;
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

}  // namespace inferx
