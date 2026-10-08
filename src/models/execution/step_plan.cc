#include "inferx/models/execution/step_plan.h"

#include <limits>

#include "inferx/models/execution/state_storage.h"  // KvBlockPool::BlocksForTokens

namespace inferx::models::execution {
namespace {

Status CheckChunk(const PlanLimits& lim, const PackedStep& packed, RequestId id, int start,
                  int chunk) {
  if (start < 0 || chunk <= 0 || chunk > lim.max_tokens ||
      packed.tokens.size() + chunk > static_cast<size_t>(lim.max_tokens)) {
    return InvalidArgumentError("invalid token allocation for request ", id);
  }
  return OkStatus();
}

Status CheckEnd(const PlanLimits& lim, const ScheduledEnds& expected, RequestId id, int end) {
  if (end > lim.max_context || end > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("request ", id, " exceeds model context length");
  }
  const auto it = expected.find(id);
  if (it != expected.end() && it->second != end) {
    return InternalError("computed-token watermark mismatch for request ", id);
  }
  return OkStatus();
}

/// Prompt chunks copy from the prompt; decode feeds back the last sampled token.
Status AppendTokens(const RequestRecord& rec, RequestId id, int start, int end, int chunk,
                    int64_t prompt_len, std::vector<int32_t>& tokens) {
  if (end <= prompt_len) {
    tokens.insert(tokens.end(), rec.prompt.begin() + start, rec.prompt.begin() + end);
    return OkStatus();
  }
  if (chunk != 1 || start < prompt_len || rec.last_sampled < 0) {
    return InternalError("inconsistent decode allocation for request ", id);
  }
  tokens.push_back(rec.last_sampled);
  return OkStatus();
}

void AppendPositions(int start, int chunk, int batch_index, PackedStep& p) {
  for (int j = 0; j < chunk; ++j) {
    p.positions.push_back(start + j);
    p.batch_indices.push_back(batch_index);
  }
}

Status AppendKvPages(const PlanLimits& lim, KvBlockPool& kv, const RequestRecord& rec,
                     RequestId id, int end, PackedStep& p) {
  const int64_t pages = kv.BlocksForTokens(end);
  if (static_cast<int64_t>(rec.block_ids.size()) < pages ||
      static_cast<int64_t>(p.kv_indices.size()) + pages > lim.num_kv_blocks) {
    return InternalError("insufficient KV block capacity for request ", id);
  }
  for (int64_t j = 0; j < pages; ++j) {
    const int32_t block = rec.block_ids[j];
    if (block < 0 || block >= lim.num_kv_blocks) return InvalidArgumentError("invalid KV block");
    p.kv_indices.push_back(block);
  }
  p.kv_indptr.push_back(static_cast<int32_t>(p.kv_indices.size()));
  p.last_page_len.push_back(static_cast<int32_t>((end - 1) % lim.block_size + 1));
  return OkStatus();
}

Status PlanRequest(const PlanLimits& lim, KvBlockPool& kv, const ScheduledEnds& expected,
                   RequestRecord& rec, RequestId id, int chunk, StepPlan& plan) {
  PackedStep& p = plan.packed;
  const int start = rec.committed;
  INFERX_RETURN_IF_ERROR(CheckChunk(lim, p, id, start, chunk));
  const int end = start + chunk;
  INFERX_RETURN_IF_ERROR(CheckEnd(lim, expected, id, end));

  // Output-row selection: only allocations completing the prompt or
  // continuing decode produce a usable token. Mid-prompt prefill chunks
  // compute state and report an empty sample.
  const int64_t prompt_len = rec.prompt.size();
  const bool sample = end >= prompt_len;
  plan.pure_decode = plan.pure_decode && sample && chunk == 1 && start >= prompt_len;

  INFERX_RETURN_IF_ERROR(AppendTokens(rec, id, start, end, chunk, prompt_len, p.tokens));
  AppendPositions(start, chunk, plan.batch(), p);
  p.qo_indptr.push_back(static_cast<int32_t>(p.tokens.size()));
  INFERX_RETURN_IF_ERROR(AppendKvPages(lim, kv, rec, id, end, p));
  if (sample) p.logit_rows.push_back(static_cast<int32_t>(p.tokens.size() - 1));

  plan.requests.push_back(StepRequest{&rec, start, chunk, end, sample});
  return OkStatus();
}

Status CheckVocab(const PlanLimits& lim, const PackedStep& p) {
  for (int32_t token : p.tokens) {
    if (token < 0 || token >= lim.vocab_size) {
      return InvalidArgumentError("token outside vocabulary");
    }
  }
  return OkStatus();
}

Status AttachRecurrentSlots(StepPlan& plan) {
  std::vector<int32_t> slots;
  slots.reserve(plan.requests.size());
  for (const auto& step : plan.requests) {
    if (step.record->recurrent_slot < 0) {
      return InternalError("missing recurrent slot for a scheduled request");
    }
    slots.push_back(step.record->recurrent_slot);
  }
  plan.packed.recurrent_slots = std::move(slots);
  return OkStatus();
}

}  // namespace

StatusOr<StepPlan> BuildStepPlan(const SchedulerOutput& output, RequestStateTable& requests,
                                 KvBlockPool& kv, const PlanLimits& limits,
                                 const ScheduledEnds& expected_ends) {
  const int batch = static_cast<int>(output.scheduled.size());
  if (batch > limits.max_seqs) return InvalidArgumentError("batch exceeds max_num_seqs");

  StepPlan plan;
  plan.requests.reserve(batch);
  plan.packed.tokens.reserve(output.total_num_scheduled_tokens);
  plan.packed.qo_indptr.push_back(0);
  plan.packed.kv_indptr.push_back(0);

  for (const auto& sr : output.scheduled) {
    auto* record = requests.Find(sr.request_id);
    if (record == nullptr) return InternalError("unknown scheduled request ", sr.request_id);
    INFERX_RETURN_IF_ERROR(PlanRequest(limits, kv, expected_ends, *record, sr.request_id,
                                       sr.num_new_tokens, plan));
  }
  INFERX_RETURN_IF_ERROR(CheckVocab(limits, plan.packed));
  if (limits.has_recurrent) INFERX_RETURN_IF_ERROR(AttachRecurrentSlots(plan));
  return plan;
}

}  // namespace inferx::models::execution
