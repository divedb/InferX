#include "inferx/models/model_runner.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
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
};

}  // namespace

struct ModelRunnerImpl {
  ModelConfig config;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  DeviceRuntime* runtime = nullptr;
  Stream stream;
  std::unique_ptr<Model> model;
  std::unique_ptr<KvBlockPool> pool;
  ModelState model_state;
  absl::flat_hash_map<RequestId, RunnerRequestState> states;
  Tensor token_ids, positions, batch_indices, qo_indptr, kv_indptr;
  Tensor kv_indices, last_page_len, logit_rows;
  Tensor input_storage;
  int32_t* host_inputs = nullptr;
  std::unique_ptr<sampling::Sampler> sampler;
  int32_t* host_samples = nullptr;
  struct DecodeGraph { GraphExec exec; Tensor logits; sampling::SamplerOutput output; bool warmed = false; };
  absl::flat_hash_map<int, DecodeGraph> decode_graphs;
  Status failure;

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
    const ModelConfig& model, const CacheConfig& cache,
    const SchedulerConfig& scheduler, const ExecutionConfig& execution) {
  if (scheduler.max_num_batched_tokens <= 0 || scheduler.max_num_seqs <= 0) {
    return InvalidArgumentError("runner requires positive token and sequence capacities");
  }
  INFERX_ASSIGN_OR_RETURN(auto backend, ops::ParseAttentionBackend(execution.attention_backend));
  INFERX_ASSIGN_OR_RETURN(
      auto loaded, Model::Load(model.model_dir, model.device, scheduler.max_num_batched_tokens,
                               scheduler.max_num_seqs, backend));
  return Create(model, cache, scheduler, execution, std::move(loaded));
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(
    const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
    const ExecutionConfig& execution, std::unique_ptr<Model> loaded) {
  INFERX_RETURN_IF_ERROR(ops::ParseAttentionBackend(execution.attention_backend).status());
  if (!loaded || scheduler.max_num_batched_tokens <= 0 || scheduler.max_num_seqs <= 0 ||
      cache.num_kv_blocks <= 0 || cache.num_kv_blocks > std::numeric_limits<int32_t>::max() ||
      cache.block_size <= 0 || cache.block_size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("runner requires a model, and positive int32 KV capacities");
  }
  const auto requirements = loaded->StateRequirements();
  if (requirements.empty() || requirements.size() !=
                                  static_cast<size_t>(loaded->config().num_hidden_layers)) {
    return InvalidArgumentError("model must declare state for each decoder layer");
  }
  KvLayout layout;
  for (size_t i = 0; i < requirements.size(); ++i) {
    const auto* spec = std::get_if<PagedKvStateSpec>(&requirements[i]);
    if (!spec) {
      return UnimplementedError("runner does not yet manage recurrent layer state");
    }
    if (i == 0) layout = spec->layout;
    if (spec->layout.entries_per_token != layout.entries_per_token ||
        spec->layout.kv_heads != layout.kv_heads || spec->layout.head_dim != layout.head_dim ||
        spec->layout.dtype != layout.dtype) {
      return UnimplementedError("runner requires a uniform paged KV layout across layers");
    }
  }
  auto impl = std::make_unique<ModelRunnerImpl>();
  impl->config = model;
  impl->cache = cache;
  impl->scheduler = scheduler;
  impl->execution = execution;
  impl->model = std::move(loaded);
  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(model.device));
  INFERX_RETURN_IF_ERROR(impl->runtime->Activate());
  INFERX_ASSIGN_OR_RETURN(impl->stream, impl->runtime->CreateStream());
  const auto& mc = impl->model->config();
  INFERX_ASSIGN_OR_RETURN(auto pool,
                          KvBlockPool::Create(mc.num_hidden_layers, cache.num_kv_blocks,
                                              cache.block_size, layout, model.device));
  impl->pool = std::make_unique<KvBlockPool>(std::move(pool));
  impl->model_state.paged_kv = impl->pool.get();
  for (size_t i = 0; i < requirements.size(); ++i) {
    impl->model_state.layers.push_back(PagedKvState{static_cast<int64_t>(i)});
  }
  auto alloc = [&](int64_t size) {
    return Tensor::Empty(DataType::kInt32, Shape({size}), model.device);
  };
  const int64_t T = scheduler.max_num_batched_tokens;
  const int64_t S = scheduler.max_num_seqs;
  // Fixed offsets keep graph pointers stable. One pinned upload replaces eight
  // synchronous copies; Run drains the stream before this staging area is reused.
  INFERX_ASSIGN_OR_RETURN(impl->input_storage, alloc(3 * T + 4 * S + 2 + cache.num_kv_blocks));
  int64_t offset = 0;
  auto input_view = [&](int64_t size) -> StatusOr<Tensor> {
    INFERX_ASSIGN_OR_RETURN(auto view, impl->input_storage.Slice(offset, offset + size));
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
  if (model.device.IsCuda()) {
    INFERX_ASSIGN_OR_RETURN(void* inputs,
        impl->runtime->AllocatePinnedHost(impl->input_storage.NBytes()));
    impl->host_inputs = static_cast<int32_t*>(inputs);
    std::memset(inputs, 0, impl->input_storage.NBytes());
    INFERX_ASSIGN_OR_RETURN(void* host, impl->runtime->AllocatePinnedHost(S * sizeof(int32_t)));
    impl->host_samples = static_cast<int32_t*>(host);
  }
  INFERX_ASSIGN_OR_RETURN(impl->sampler,
                          sampling::Sampler::Create(scheduler.max_num_seqs, mc.vocab_size, model.device));
  return std::unique_ptr<ModelRunner>(new ModelRunner(std::move(impl)));
}

StatusOr<ModelRunnerOutput> ModelRunnerImpl::Execute(const SchedulerOutput& output) {
  ModelRunnerOutput result;
  // Process removals before admission so a finished ID can safely be reused.
  for (RequestId id : output.finished_request_ids) states.erase(id);
  for (const auto& nr : output.scheduled_new_reqs) {
    RunnerRequestState state;
    state.block_ids = nr.block_ids;
    state.prompt.assign(nr.prompt_token_ids.begin(), nr.prompt_token_ids.end());
    state.params = nr.sampling_params;
    state.num_computed = nr.num_computed_tokens;
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
  auto upload = [&](Tensor& dst, const std::vector<int32_t>& src) {
    if (host_inputs != nullptr) {
      const auto offset = dst.DataAs<int32_t>() - input_storage.DataAs<int32_t>();
      std::memcpy(host_inputs + offset, src.data(), src.size() * sizeof(int32_t));
      return OkStatus();
    }
    return runtime->Copy(dst.Data(), src.data(), src.size() * sizeof(int32_t),
                         CopyKind::kHostToDevice);
  };
  INFERX_RETURN_IF_ERROR(upload(token_ids, tokens));
  INFERX_RETURN_IF_ERROR(upload(positions, positions_h));
  INFERX_RETURN_IF_ERROR(upload(batch_indices, batches));
  INFERX_RETURN_IF_ERROR(upload(qo_indptr, qo));
  INFERX_RETURN_IF_ERROR(upload(kv_indptr, kv));
  INFERX_RETURN_IF_ERROR(upload(kv_indices, blocks));
  INFERX_RETURN_IF_ERROR(upload(last_page_len, last_lengths));
  INFERX_RETURN_IF_ERROR(upload(logit_rows, rows));
  if (host_inputs != nullptr) {
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(input_storage.Data(), host_inputs,
        input_storage.NBytes(), CopyKind::kHostToDevice, stream));
  }
  // ModelInput carries exact-size views of the capacity buffers; the padded
  // tails of the upload buffers are never visible to the model.
  ModelInput input;
  INFERX_ASSIGN_OR_RETURN(input.token_ids, token_ids.Slice(0, tokens.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor positions_v, positions.Slice(0, tokens.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor batch_indices_v, batch_indices.Slice(0, tokens.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor qo_indptr_v, qo_indptr.Slice(0, batch + 1));
  INFERX_ASSIGN_OR_RETURN(Tensor kv_indptr_v, kv_indptr.Slice(0, batch + 1));
  INFERX_ASSIGN_OR_RETURN(Tensor kv_indices_v, kv_indices.Slice(0, blocks.size()));
  INFERX_ASSIGN_OR_RETURN(Tensor last_page_len_v, last_page_len.Slice(0, batch));
  INFERX_ASSIGN_OR_RETURN(input.logit_rows, logit_rows.Slice(0, batch));
  input.attention = {std::move(positions_v),
                     std::move(batch_indices_v),
                     std::move(qo_indptr_v),
                     std::move(kv_indptr_v),
                     std::move(kv_indices_v),
                     std::move(last_page_len_v),
                     absl::MakeConstSpan(qo),
                     absl::MakeConstSpan(kv),
                     static_cast<int>(tokens.size()),
                     batch};
  ops::ExecutionContext ctx(*runtime, stream);
  Tensor logits;
  sampling::SamplerOutput sampled;
  // Per-request sampling knobs plus RNG offsets, resolved for this batch.
  std::vector<sampling::SamplingMetadata::PerRequest> per(batch);
  for (int i = 0; i < batch; ++i) {
    const RunnerRequestState& state = states.at(output.scheduled[i].request_id);
    per[i] = {&state.params, state.generated};
  }
  const sampling::SamplingMetadata metadata =
      sampling::SamplingMetadata::Build(mc.vocab_size, absl::MakeConstSpan(per));
  if (pure_decode && execution.enable_cuda_graphs && config.device.IsCuda() &&
      model->SupportsCudaGraphs()) {
    auto& graph = decode_graphs[batch];
    if (!graph.warmed) {
      // Let cuBLAS initialize algorithms/workspace for this exact shape before capture.
      INFERX_ASSIGN_OR_RETURN(logits, model->Forward(input, model_state, ctx));
      graph.warmed = true;
    } else {
      if (graph.exec.handle == nullptr) {
        INFERX_RETURN_IF_ERROR(runtime->BeginCapture(stream));
        auto captured = model->Forward(input, model_state, ctx);
        Status sampling_status;
        if (captured.ok()) {
          sampling_status = sampler->Sample(ctx, *captured, metadata, graph.output);
        }
        // End the capture even on failure, so teardown never leaves a stream capturing.
        auto instantiated = runtime->EndCaptureAndInstantiate(stream);
        if (!captured.ok() || !sampling_status.ok()) {
          if (instantiated.ok()) (void)runtime->DestroyGraph(*instantiated);
          return captured.ok() ? sampling_status : captured.status();
        }
        INFERX_ASSIGN_OR_RETURN(graph.exec, std::move(instantiated));
        graph.logits = *std::move(captured);
      }
      INFERX_RETURN_IF_ERROR(runtime->LaunchGraph(graph.exec, stream));
      logits = graph.logits;
      sampled = graph.output;  // Graph replay already produced the samples.
    }
  } else {
    INFERX_ASSIGN_OR_RETURN(logits, model->Forward(input, model_state, ctx));
    INFERX_RETURN_IF_ERROR(sampler->Sample(ctx, logits, metadata, sampled));
  }
  // The graph warm-up iteration runs Forward eagerly without capture; sample
  // its logits through the normal path.
  if (!sampled.IsDefined()) {
    INFERX_RETURN_IF_ERROR(sampler->Sample(ctx, logits, metadata, sampled));
  }
  if (!logits.IsDefined() || logits.Rank() != 2 || logits.Dim(0) != batch ||
      (logits.GetDataType() != DataType::kBFloat16 &&
       logits.GetDataType() != DataType::kFloat) ||
      logits.Device() != config.device) {
    return InternalError("model returned malformed logits for the scheduled batch");
  }
  if (!sampled.IsDefined() || sampled.sampled_token_ids.Dim(0) != batch)
    return InternalError("sampler returned malformed results for the scheduled batch");
  std::vector<int32_t> samples;
  if (host_samples != nullptr) {
    INFERX_RETURN_IF_ERROR(runtime->CopyAsync(host_samples, sampled.sampled_token_ids.Data(),
        batch * sizeof(int32_t), CopyKind::kDeviceToHost, stream));
    INFERX_RETURN_IF_ERROR(runtime->SynchronizeStream(stream));
    samples.assign(host_samples, host_samples + batch);
  } else {
    // CPU device: tensors are host-accessible once the stream has drained.
    INFERX_RETURN_IF_ERROR(runtime->SynchronizeStream(stream));
    const int32_t* ids = sampled.sampled_token_ids.DataAs<int32_t>();
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
  if (!impl_->failure.ok()) {
    return FailedPreconditionError("runner failed previously: ", impl_->failure.message());
  }
  auto result = impl_->Execute(output);
  if (!result.ok()) {
    // A scheduler step and cache writes cannot be rolled back safely. Drain
    // queued work and require reconstruction instead of retrying partial state.
    (void)impl_->runtime->SynchronizeStream(impl_->stream);
    impl_->failure = result.status();
  }
  return result;
}

}  // namespace inferx
