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

namespace inferx {
namespace {

struct RunnerRequestState {
  std::vector<int32_t> block_ids;
  std::vector<TokenId> prompt;
  int num_computed = 0;
  TokenId last_sampled = -1;
};

float Bf16BitsToFloat(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

struct ModelRunnerImpl {
  ModelRunnerConfig config;
  DeviceRuntime* runtime = nullptr;
  Stream stream;
  std::unique_ptr<Model> model;
  std::unique_ptr<KvBlockPool> pool;
  absl::flat_hash_map<RequestId, RunnerRequestState> states;
  Tensor token_ids, positions, batch_indices, qo_indptr, kv_indptr;
  Tensor kv_indices, last_page_len, logit_rows;
  Status failure;

  ~ModelRunnerImpl() {
    if (runtime == nullptr) return;
    (void)runtime->Activate();
    // Drain queued work even after an operation failed before the runner's sync.
    if (stream.handle != nullptr) (void)runtime->SynchronizeStream(stream);
    if (stream.handle != nullptr) (void)runtime->DestroyStream(stream);
  }

  StatusOr<ModelRunnerOutput> Execute(const SchedulerOutput& output);
  Status GreedySample(const Tensor& logits, int batch, std::vector<int32_t>& samples);
};

ModelRunner::ModelRunner(std::unique_ptr<ModelRunnerImpl> impl) : impl_(std::move(impl)) {}
ModelRunner::~ModelRunner() = default;
KvBlockPool* ModelRunner::kv_pool() { return impl_->pool.get(); }
const ModelConfig& ModelRunner::model_config() const { return impl_->model->config(); }

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(const ModelRunnerConfig& config) {
  if (config.max_num_batched_tokens <= 0 || config.max_num_seqs <= 0) {
    return InvalidArgumentError("runner requires positive token and sequence capacities");
  }
  INFERX_ASSIGN_OR_RETURN(
      auto model, Model::Load(config.model_dir, config.device, config.max_num_batched_tokens,
                              config.max_num_seqs));
  return Create(config, std::move(model));
}

StatusOr<std::unique_ptr<ModelRunner>> ModelRunner::Create(const ModelRunnerConfig& config,
                                                           std::unique_ptr<Model> model) {
  if (!model || config.max_num_batched_tokens <= 0 || config.max_num_seqs <= 0 ||
      config.num_kv_blocks <= 0 || config.num_kv_blocks > std::numeric_limits<int32_t>::max() ||
      config.block_size <= 0 || config.block_size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("runner requires a model, and positive int32 KV capacities");
  }
  auto impl = std::make_unique<ModelRunnerImpl>();
  impl->config = config;
  impl->model = std::move(model);
  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(config.device));
  INFERX_RETURN_IF_ERROR(impl->runtime->Activate());
  INFERX_ASSIGN_OR_RETURN(impl->stream, impl->runtime->CreateStream());
  const auto& mc = impl->model->config();
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = mc.num_key_value_heads;
  layout.head_dim = mc.head_dim;
  layout.dtype = DataType::kBFloat16;
  INFERX_ASSIGN_OR_RETURN(auto pool,
                          KvBlockPool::Create(mc.num_hidden_layers, config.num_kv_blocks,
                                              config.block_size, layout, config.device));
  impl->pool = std::make_unique<KvBlockPool>(std::move(pool));
  auto alloc = [&](int64_t size) {
    return Tensor::Empty(DataType::kInt32, Shape({size}), config.device);
  };
  const int64_t T = config.max_num_batched_tokens;
  const int64_t S = config.max_num_seqs;
  INFERX_ASSIGN_OR_RETURN(impl->token_ids, alloc(T));
  INFERX_ASSIGN_OR_RETURN(impl->positions, alloc(T));
  INFERX_ASSIGN_OR_RETURN(impl->batch_indices, alloc(T));
  INFERX_ASSIGN_OR_RETURN(impl->qo_indptr, alloc(S + 1));
  INFERX_ASSIGN_OR_RETURN(impl->kv_indptr, alloc(S + 1));
  INFERX_ASSIGN_OR_RETURN(impl->kv_indices, alloc(config.num_kv_blocks));
  INFERX_ASSIGN_OR_RETURN(impl->last_page_len, alloc(S));
  INFERX_ASSIGN_OR_RETURN(impl->logit_rows, alloc(S));
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
  if (batch > config.max_num_seqs) return InvalidArgumentError("batch exceeds max_num_seqs");

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
    if (start < 0 || chunk <= 0 || chunk > config.max_num_batched_tokens ||
        tokens.size() + chunk > static_cast<size_t>(config.max_num_batched_tokens)) {
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
        blocks.size() + pages > static_cast<size_t>(config.num_kv_blocks)) {
      return InternalError("insufficient KV block capacity for request ", sr.request_id);
    }
    for (int64_t j = 0; j < pages; ++j) {
      const int32_t block = state.block_ids[j];
      if (block < 0 || block >= config.num_kv_blocks)
        return InvalidArgumentError("invalid KV block");
      blocks.push_back(block);
    }
    kv.push_back(blocks.size());
    last_lengths.push_back((end - 1) % config.block_size + 1);
    rows.push_back(tokens.size() - 1);
  }
  for (int32_t token : tokens) {
    if (token < 0 || token >= mc.vocab_size)
      return InvalidArgumentError("token outside vocabulary");
  }
  auto upload = [&](Tensor& dst, const std::vector<int32_t>& src) {
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
  ModelInput input;
  input.token_ids = token_ids;
  input.attention = {positions,
                     batch_indices,
                     qo_indptr,
                     kv_indptr,
                     kv_indices,
                     last_page_len,
                     absl::MakeConstSpan(qo),
                     absl::MakeConstSpan(kv),
                     static_cast<int>(tokens.size()),
                     batch};
  input.logit_rows = logit_rows;
  ops::ExecutionContext ctx(*runtime, stream);
  INFERX_ASSIGN_OR_RETURN(Tensor logits, model->Forward(input, *pool, ctx));
  if (!logits.IsDefined() || logits.Rank() != 2 || logits.Dim(0) != batch ||
      (logits.GetDataType() != DataType::kBFloat16 &&
       logits.GetDataType() != DataType::kFloat) ||
      logits.Device() != config.device) {
    return InternalError("model returned malformed logits for the scheduled batch");
  }
  std::vector<int32_t> samples;
  INFERX_RETURN_IF_ERROR(GreedySample(logits, batch, samples));
  for (int i = 0; i < batch; ++i) {
    const auto& sr = output.scheduled[i];
    result.samples.push_back({sr.request_id, {samples[i]}, std::nullopt});
    auto& state = states.at(sr.request_id);
    state.num_computed += sr.num_new_tokens;
    state.last_sampled = samples[i];
  }
  return result;
}

Status ModelRunnerImpl::GreedySample(const Tensor& logits, int batch,
                                     std::vector<int32_t>& samples) {
  INFERX_RETURN_IF_ERROR(runtime->SynchronizeStream(stream));
  const int64_t vocab = logits.Dim(1);
  const bool is_bf16 = logits.GetDataType() == DataType::kBFloat16;
  const size_t elem = is_bf16 ? sizeof(uint16_t) : sizeof(float);
  std::vector<std::byte> host(static_cast<size_t>(logits.NBytes()));
  INFERX_RETURN_IF_ERROR(
      runtime->Copy(host.data(), logits.Data(), host.size(), CopyKind::kDeviceToHost));
  auto value_at = [&](int64_t row, int64_t col) -> float {
    const std::byte* p = host.data() + (row * vocab + col) * elem;
    if (!is_bf16) {
      float f = 0.0f;
      std::memcpy(&f, p, sizeof(f));
      return f;
    }
    uint16_t h = 0;
    std::memcpy(&h, p, sizeof(h));
    return Bf16BitsToFloat(h);
  };
  samples.resize(batch);
  for (int i = 0; i < batch; ++i) {
    // Ties resolve to the lowest index. Host-side for now; sampling becomes
    // an op when kernels for it land.
    float best = value_at(i, 0);
    int32_t best_idx = 0;
    for (int64_t j = 1; j < vocab; ++j) {
      const float value = value_at(i, j);
      if (value > best) {
        best = value;
        best_idx = static_cast<int32_t>(j);
      }
    }
    samples[i] = best_idx;
  }
  return OkStatus();
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
