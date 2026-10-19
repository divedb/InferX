#include "inferx/models/model_runner.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace inferx {
namespace models {

// ---------------------------------------------------------------------------
// Assumed APIs. Everything below that touches the device, the model, the
// attention runtime, or the sampler goes through the calls listed here, so
// adapting to the real signatures is a local edit.
//
// SchedulerOutput (vLLM-shaped):
//   step.scheduled_new_reqs     [{req_id, prompt_token_ids, num_computed_tokens,
//                                 sampling_params}]
//   step.scheduled_cached_reqs  [{req_id, new_token_ids, num_computed_tokens}]
//   step.finished_req_ids       RequestId set/vector
//   step.num_scheduled_tokens   unordered_map<RequestId, int>
//   step.total_num_scheduled_tokens
//   ModelRunnerOutput { std::vector<RequestId> req_ids;
//                       std::vector<std::vector<TokenId>> sampled_token_ids; }
//
// Device:
//   Tensor::FromHostI32(runtime, stream, data, shape)
//   runtime->Synchronize(stream), runtime->ResetPeakMemory(),
//   runtime->PeakMemoryBytes(), runtime->AllocatedBytes()
//
// Model:
//   Load(options), WeightsBytes(), Forward(op, ops, token_ids, positions, meta),
//   ComputeLogits(op, hidden, logits_indices)
//
// Attention (attn::AttentionRuntime):
//   Create(spec, kv_config, scheduler, runtime, stream), CreateForProfile(...),
//   MakeOp(layer), BlockTable(req_id), BlockSize(), MaxBlocksPerSeq(),
//   BuildMetadata(req_ids, num_tokens, num_computed)
//
// Sampler:
//   Sample(logits, params, rng_offsets) -> StatusOr<std::vector<TokenId>>
//
// ScratchArena::Create(runtime, stream, spec, max_tokens, max_seqs);
// ops::OpContext(runtime, stream, comm, allocator)
// ---------------------------------------------------------------------------

#define MR_RETURN_IF_ERROR(expr)   \
  do {                             \
    ::inferx::Status _st = (expr); \
    if (!_st.ok()) return _st;     \
  } while (0)

namespace {

StatusOr<Tensor> UploadI32(DeviceRuntime* runtime, DeviceStream stream,
                           const std::vector<int32_t>& data, std::vector<int64_t> shape) {
  return Tensor::FromHostI32(runtime, stream, data, std::move(shape));
}

}  // namespace

ModelRunner::ModelRunner(std::unique_ptr<Model> model, DeviceRuntime* runtime,
                         DeviceStream stream, CommBackend* comm, bool is_driver_rank,
                         RunnerDebugOptions debug)
    : runtime_(runtime),
      stream_(stream),
      comm_(comm),
      is_driver_rank_(is_driver_rank),
      debug_(debug),
      model_(std::move(model)) {}

ModelRunner::~ModelRunner() = default;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

Status ModelRunner::LoadModel(const LoadOptions& options) { return model_->Load(options); }

StatusOr<ProfileResult> ModelRunner::ProfileRun(const SchedulerConfig& scheduler) {
  const int64_t max_tokens = scheduler.max_num_batched_tokens;
  const int64_t max_seqs = scheduler.max_num_seqs;

  auto scratch = ScratchArena::Create(runtime_, stream_, GetModelSpec(), max_tokens, max_seqs);
  if (!scratch.ok()) return scratch.status();
  scratch_ = std::move(*scratch);
  op_ = std::make_unique<ops::OpContext>(runtime_, stream_, comm_, scratch_->allocator());

  // Throwaway attention runtime: just enough cache to run the largest step.
  auto profile_attn =
      attn::AttentionRuntime::CreateForProfile(GetModelSpec(), scheduler, runtime_, stream_);
  if (!profile_attn.ok()) return profile_attn.status();

  runtime_->ResetPeakMemory();
  const int64_t before = runtime_->AllocatedBytes();

  // Worst-case shape: one sequence per slot, tokens split evenly.
  const int64_t per_seq = std::max<int64_t>(1, max_tokens / std::max<int64_t>(1, max_seqs));
  std::vector<RequestId> ids;
  std::vector<int> num_tokens(max_seqs, static_cast<int>(per_seq));
  std::vector<int> num_computed(max_seqs, 0);
  for (int64_t i = 0; i < max_seqs; ++i) ids.push_back(static_cast<RequestId>(i));

  std::vector<int32_t> tokens(max_tokens, 0), positions(max_tokens, 0);
  for (int64_t s = 0, t = 0; s < max_seqs; ++s)
    for (int64_t i = 0; i < per_seq && t < max_tokens; ++i, ++t) positions[t] = i;
  std::vector<int32_t> last_idx(max_seqs);
  for (int64_t s = 0; s < max_seqs; ++s)
    last_idx[s] = static_cast<int32_t>((s + 1) * per_seq - 1);

  auto tok_t = UploadI32(runtime_, stream_, tokens, {max_tokens});
  if (!tok_t.ok()) return tok_t.status();
  auto pos_t = UploadI32(runtime_, stream_, positions, {max_tokens});
  if (!pos_t.ok()) return pos_t.status();
  auto idx_t = UploadI32(runtime_, stream_, last_idx, {max_seqs});
  if (!idx_t.ok()) return idx_t.status();

  auto meta = (*profile_attn)->BuildMetadata(ids, num_tokens, num_computed);
  if (!meta.ok()) return meta.status();

  std::vector<attn::AttentionOp> profile_ops;
  for (int i = 0; i < GetModelSpec().num_layers; ++i)
    profile_ops.push_back((*profile_attn)->MakeOp(i));

  auto hidden = model_->Forward(*op_, profile_ops, *tok_t, *pos_t, *meta);
  if (!hidden.ok()) return hidden.status();
  auto logits = model_->ComputeLogits(*op_, *hidden, *idx_t);
  if (!logits.ok()) return logits.status();
  MR_RETURN_IF_ERROR(runtime_->Synchronize(stream_));

  ProfileResult result;
  result.weights_bytes = model_->WeightsBytes();
  result.peak_activation_bytes = std::max<int64_t>(0, runtime_->PeakMemoryBytes() - before);
  return result;
}

Status ModelRunner::InitializeKVCache(const KVCacheConfig& kv,
                                      const SchedulerConfig& scheduler) {
  if (!scratch_) return Internal("InitializeKVCache called before ProfileRun");

  auto attention =
      attn::AttentionRuntime::Create(GetModelSpec(), kv, scheduler, runtime_, stream_);
  if (!attention.ok()) return attention.status();
  attention_ = std::move(*attention);

  ops_.clear();
  for (int layer = 0; layer < GetModelSpec().num_layers; ++layer) {
    ops_.push_back(attention_->MakeOp(layer));
  }

  input_batch_ = std::make_unique<InputBatch>(static_cast<int>(scheduler.max_num_seqs),
                                              GetModelSpec().max_model_len);
  if (is_driver_rank_) sampler_ = std::make_unique<sampling::Sampler>(runtime_, stream_);
  return OkStatus();
}

Status ModelRunner::CaptureGraphs() {
  // CUDA graphs are not wired into the new model layer yet.
  return OkStatus();
}

// ---------------------------------------------------------------------------
// Per-step execution
// ---------------------------------------------------------------------------

StatusOr<ModelRunnerOutput> ModelRunner::ExecuteModel(const SchedulerOutput& step) {
  if (!attention_ || !input_batch_) {
    return Internal("ExecuteModel called before InitializeKVCache");
  }

  MR_RETURN_IF_ERROR(UpdateStates(step));

  ModelRunnerOutput out;
  if (step.total_num_scheduled_tokens == 0) return out;

  StepInputs inputs;
  MR_RETURN_IF_ERROR(PrepareInputs(step, &inputs));

  auto hidden = Forward(inputs);
  if (!hidden.ok()) return hidden.status();
  auto logits = ComputeLogits(*hidden, inputs);
  if (!logits.ok()) return logits.status();

  if (debug_.capture_logits) captured_logits_.push_back(*logits);

  MR_RETURN_IF_ERROR(Sample(*logits, inputs, &out));

  // Advance computed counts only after sampling, which needs the old values
  // to tell whether a sequence reached the tip of its token history.
  for (size_t i = 0; i < inputs.rows.size(); ++i) {
    const int row = inputs.rows[i];
    MR_RETURN_IF_ERROR(input_batch_->SetNumComputed(
        inputs.req_ids[i], input_batch_->NumComputed(row) + inputs.num_tokens[i]));
  }

  MR_RETURN_IF_ERROR(runtime_->Synchronize(stream_));
  return out;
}

Status ModelRunner::UpdateStates(const SchedulerOutput& step) {
  // Finished first so their rows can be reused by new requests.
  for (RequestId id : step.finished_req_ids) input_batch_->Remove(id);

  for (const auto& r : step.scheduled_new_reqs) {
    InputBatch::NewRequest req;
    req.id = r.req_id;
    req.prompt_tokens = r.prompt_token_ids;
    req.num_computed = r.num_computed_tokens;
    req.params = r.sampling_params;
    MR_RETURN_IF_ERROR(input_batch_->Add(req));
  }

  for (const auto& r : step.scheduled_cached_reqs) {
    if (!input_batch_->RowOf(r.req_id)) {
      return NotFound("cached request " + std::to_string(r.req_id) + " is not in the batch");
    }
    // The driver rank appended its own samples in Sample(); only other ranks
    // learn accepted tokens through the scheduler diff.
    if (!is_driver_rank_) {
      for (TokenId t : r.new_token_ids) {
        MR_RETURN_IF_ERROR(input_batch_->AppendToken(r.req_id, t));
      }
    }
    // The scheduler is authoritative for how much KV exists.
    MR_RETURN_IF_ERROR(input_batch_->SetNumComputed(r.req_id, r.num_computed_tokens));
  }

  input_batch_->Condense();
  return OkStatus();
}

Status ModelRunner::PrepareInputs(const SchedulerOutput& step, StepInputs* inputs) {
  // Scheduler order: new requests, then cached requests.
  for (const auto& r : step.scheduled_new_reqs) inputs->req_ids.push_back(r.req_id);
  for (const auto& r : step.scheduled_cached_reqs) inputs->req_ids.push_back(r.req_id);

  const int num_seqs = static_cast<int>(inputs->req_ids.size());
  const int block_size = attention_->BlockSize();
  const int max_blocks = attention_->MaxBlocksPerSeq();

  std::vector<int32_t> token_ids, positions, slot_mapping, query_start_loc{0}, logits_idx;
  std::vector<int32_t> block_tables(static_cast<size_t>(num_seqs) * max_blocks, 0);
  token_ids.reserve(step.total_num_scheduled_tokens);
  positions.reserve(step.total_num_scheduled_tokens);
  slot_mapping.reserve(step.total_num_scheduled_tokens);

  for (int s = 0; s < num_seqs; ++s) {
    const RequestId id = inputs->req_ids[s];
    auto row = input_batch_->RowOf(id);
    if (!row) return Internal("scheduled request " + std::to_string(id) + " has no batch row");

    auto it = step.num_scheduled_tokens.find(id);
    if (it == step.num_scheduled_tokens.end()) {
      return Internal("request " + std::to_string(id) + " has no scheduled token count");
    }
    const int n = it->second;
    const int start = input_batch_->NumComputed(*row);
    if (start + n > input_batch_->NumTokens(*row)) {
      return Internal("request " + std::to_string(id) + " schedules past its token history");
    }

    const std::vector<int32_t>& table = attention_->BlockTable(id);
    for (int i = 0; i < n; ++i) {
      const int pos = start + i;
      const int block_idx = pos / block_size;
      if (block_idx >= static_cast<int>(table.size())) {
        return Internal("request " + std::to_string(id) + " has no block for position " +
                        std::to_string(pos));
      }
      token_ids.push_back(input_batch_->TokensAt(*row)[pos]);
      positions.push_back(pos);
      slot_mapping.push_back(table[block_idx] * block_size + pos % block_size);
    }
    std::copy(table.begin(), table.begin() + std::min<size_t>(table.size(), max_blocks),
              block_tables.begin() + static_cast<size_t>(s) * max_blocks);

    query_start_loc.push_back(query_start_loc.back() + n);
    logits_idx.push_back(query_start_loc.back() - 1);
    inputs->rows.push_back(*row);
    inputs->num_tokens.push_back(n);
  }

  const int64_t total = static_cast<int64_t>(token_ids.size());
  auto upload = [&](const std::vector<int32_t>& v, std::vector<int64_t> shape,
                    Tensor* dst) -> Status {
    auto t = UploadI32(runtime_, stream_, v, std::move(shape));
    if (!t.ok()) return t.status();
    *dst = std::move(*t);
    return OkStatus();
  };
  MR_RETURN_IF_ERROR(upload(token_ids, {total}, &inputs->token_ids));
  MR_RETURN_IF_ERROR(upload(positions, {total}, &inputs->positions));
  MR_RETURN_IF_ERROR(upload(slot_mapping, {total}, &inputs->slot_mapping));
  MR_RETURN_IF_ERROR(upload(block_tables, {num_seqs, max_blocks}, &inputs->block_tables));
  MR_RETURN_IF_ERROR(upload(query_start_loc, {num_seqs + 1}, &inputs->query_start_loc));
  MR_RETURN_IF_ERROR(upload(logits_idx, {num_seqs}, &inputs->logits_indices));
  return OkStatus();
}

StatusOr<Tensor> ModelRunner::Forward(const StepInputs& inputs) {
  std::vector<int> num_computed;
  num_computed.reserve(inputs.rows.size());
  for (int row : inputs.rows) num_computed.push_back(input_batch_->NumComputed(row));

  auto meta = attention_->BuildMetadata(inputs.req_ids, inputs.num_tokens, num_computed);
  if (!meta.ok()) return meta.status();
  return model_->Forward(*op_, ops_, inputs.token_ids, inputs.positions, *meta);
}

StatusOr<Tensor> ModelRunner::ComputeLogits(const Tensor& hidden, const StepInputs& inputs) {
  return model_->ComputeLogits(*op_, hidden, inputs.logits_indices);
}

Status ModelRunner::Sample(const Tensor& logits, const StepInputs& inputs,
                           ModelRunnerOutput* out) {
  const size_t num_seqs = inputs.rows.size();
  out->req_ids = inputs.req_ids;
  out->sampled_token_ids.assign(num_seqs, {});

  // Non-driver ranks only maintain state for their shard.
  if (!sampler_) return OkStatus();

  std::vector<sampling::SamplingParams> params;
  std::vector<int64_t> rng_offsets;
  for (int row : inputs.rows) {
    params.push_back(input_batch_->ParamsAt(row));
    rng_offsets.push_back(input_batch_->RngOffset(row));
  }

  auto sampled = sampler_->Sample(logits, params, rng_offsets);
  if (!sampled.ok()) return sampled.status();

  for (size_t i = 0; i < num_seqs; ++i) {
    const int row = inputs.rows[i];
    // A chunked prefill that has not reached the tip produces no token.
    const bool at_tip =
        input_batch_->NumComputed(row) + inputs.num_tokens[i] == input_batch_->NumTokens(row);
    if (!at_tip) continue;

    const TokenId token = (*sampled)[i];
    MR_RETURN_IF_ERROR(input_batch_->AppendToken(inputs.req_ids[i], token));
    input_batch_->AdvanceRngOffset(row);
    out->sampled_token_ids[i].push_back(token);
  }
  return OkStatus();
}

#undef MR_RETURN_IF_ERROR

}  // namespace models
}  // namespace inferx