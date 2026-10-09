#include "inferx/engine/scheduler.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace inferx {

Scheduler::Scheduler(SchedulerConfig config, KvBlockPool* pool,
                     std::vector<TokenId> eos_token_ids)
    : config_(config), cache_manager_(pool), eos_token_ids_(std::move(eos_token_ids)),
      waiting_(static_cast<size_t>(config.queue_capacity)) {}

Status Scheduler::AddRequest(Request request) {
  INFERX_RETURN_IF_ERROR(request.sampling_params().Validate());
  Status status = waiting_.Push(std::move(request));
  if (!status.ok()) {
    ++stats_.num_rejected;
  }
  return status;
}

StatusOr<SchedulerOutput> Scheduler::Schedule() {
  SchedulerOutput output;
  output.finished_request_ids = std::move(pending_finish_notifications_);
  pending_finish_notifications_.clear();
  int budget = config_.max_num_batched_tokens;

  // Running requests first: each gets one token per step, or the next prompt
  // chunk when its prompt is longer than what earlier steps computed. A
  // request the pool cannot cover is skipped this step, not preempted.
  for (size_t i = 0; i < running_.size() && budget > 0; ++i) {
    Request& request = running_[i];
    const int remaining_prompt =
        static_cast<int>(request.prompt().size()) - request.num_computed_tokens();
    const int chunk = remaining_prompt > 0 ? std::min(budget, remaining_prompt) : 1;

    auto granted = cache_manager_.Reserve(request.id(), request.num_computed_tokens() + chunk);
    if (!granted.ok()) {
      continue;
    }

    CachedRequestUpdate update;
    update.request_id = request.id();
    update.new_block_ids = std::move(*granted);
    update.num_computed_tokens = request.num_computed_tokens() + chunk;
    output.scheduled_cached_reqs.push_back(std::move(update));

    output.scheduled.push_back({request.id(), chunk});
    output.total_num_scheduled_tokens += chunk;
    budget -= chunk;
    request.set_num_computed_tokens(request.num_computed_tokens() + chunk);
  }

  // Admit waiting requests in arrival order while slots, budget, and KV
  // blocks allow. The first chunk is capped by the remaining budget, so a
  // prompt longer than the budget enters as chunked prefill.
  while (!waiting_.empty() && budget > 0 &&
         static_cast<int>(running_.size()) < config_.max_num_seqs) {
    StatusOr<Request> popped = waiting_.Pop();
    if (!popped.ok()) {
      return popped.status();
    }
    Request request = std::move(*popped);
    const int chunk =
        std::min(budget, static_cast<int>(request.prompt().size()));

    auto granted = cache_manager_.Reserve(request.id(), chunk);
    if (!granted.ok()) {
      // Not enough KV for even the first chunk: stop admitting this step and
      // put the request back at the head of the queue.
      waiting_.PushFront(std::move(request));
      break;
    }

    NewRequestData data;
    data.request_id = request.id();
    data.prompt_token_ids = absl::MakeConstSpan(request.prompt());
    data.sampling_params = request.sampling_params();
    data.block_ids = std::move(*granted);
    data.num_computed_tokens = 0;
    output.scheduled_new_reqs.push_back(std::move(data));

    output.scheduled.push_back({request.id(), chunk});
    output.total_num_scheduled_tokens += chunk;
    budget -= chunk;
    request.set_num_computed_tokens(chunk);

    running_index_[request.id()] = running_.size();
    running_.push_back(std::move(request));
  }

  if (!output.IsEmpty()) {
    ++stats_.steps;
    stats_.scheduled_tokens_total += output.total_num_scheduled_tokens;
  }
  return output;
}

Status Scheduler::UpdateFromOutput(const SchedulerOutput& output,
                                   const ModelRunnerOutput& result) {
  if (result.samples.size() != output.scheduled.size()) {
    return InvalidArgumentError("runner returned ", result.samples.size(), " samples for ",
                                output.scheduled.size(), " scheduled requests");
  }

  for (size_t i = 0; i < output.scheduled.size(); ++i) {
    const RequestId id = output.scheduled[i].request_id;
    auto it = running_index_.find(id);
    if (it == running_index_.end()) {
      return InternalError("sampled request ", id, " is not running");
    }
    Request& request = running_[it->second];

    // Samples answer the last token of each allocation, but only allocations
    // that completed the prompt (or continued decode) produce a usable
    // token; mid-prompt chunks are prefill compute whose sample is dropped.
    const bool prompt_complete =
        request.num_computed_tokens() >= static_cast<int>(request.prompt().size());
    const SampledTokens& sample = result.samples[i];
    if (!prompt_complete || sample.token_ids.empty()) {
      continue;
    }
    if (sample.token_ids.size() != 1) {
      return InternalError("expected one sampled token, got ", sample.token_ids.size());
    }
    const TokenId token = sample.token_ids.front();
    INFERX_RETURN_IF_ERROR(request.AppendOutput(absl::MakeConstSpan(&token, 1)));

    const sampling::SamplingParams& params = request.sampling_params();
    bool finish = false;
    FinishReason reason = FinishReason::kStopped;
    if (static_cast<int>(request.output().size()) >= request.max_new_tokens()) {
      finish = true;
      reason = FinishReason::kLengthCapped;
    } else if ((!params.ignore_eos &&
                std::find(eos_token_ids_.begin(), eos_token_ids_.end(), token) !=
                    eos_token_ids_.end()) ||
               (request.output().size() >= params.min_tokens &&
                std::find(params.stop_token_ids.begin(), params.stop_token_ids.end(),
                          token) != params.stop_token_ids.end())) {
      finish = true;
    }
    if (sample.finish_reason.has_value() && !finish) {
      finish = true;
      reason = *sample.finish_reason;
    }

    if (!finish) {
      continue;
    }
    const size_t index = it->second;
    INFERX_RETURN_IF_ERROR(cache_manager_.Release(id));
    request.Finish(reason);
    pending_finish_notifications_.push_back(id);
    finished_.push_back(std::move(running_[index]));
    running_index_.erase(id);
    // Swap-remove; repair the moved request's index entry.
    if (index + 1 < running_.size()) {
      running_[index] = std::move(running_.back());
      running_index_[running_[index].id()] = index;
    }
    running_.pop_back();
    ++stats_.num_finished;
  }
  return OkStatus();
}

std::optional<Request> Scheduler::PopFinished() {
  if (finished_.empty()) {
    return std::nullopt;
  }
  Request request = std::move(finished_.front());
  finished_.pop_front();
  return request;
}

std::vector<uint64_t> Scheduler::AbortRequests(absl::Span<const uint64_t> request_ids) {
  std::vector<uint64_t> aborted;
  for (const uint64_t id : request_ids) {
    if (waiting_.Erase(id)) {
      aborted.push_back(id);
      continue;
    }
    auto it = running_index_.find(id);
    if (it == running_index_.end()) {
      continue;
    }
    const size_t index = it->second;
    Request& request = running_[index];
    const Status released = cache_manager_.Release(id);
    assert(released.ok());
    (void)released;
    request.Finish(FinishReason::kAborted);
    pending_finish_notifications_.push_back(id);
    finished_.push_back(std::move(running_[index]));
    running_index_.erase(id);
    if (index + 1 < running_.size()) {
      running_[index] = std::move(running_.back());
      running_index_[running_[index].id()] = index;
    }
    running_.pop_back();
    ++stats_.num_finished;
    aborted.push_back(id);
  }
  return aborted;
}

SchedulerStats Scheduler::Stats() const {
  SchedulerStats stats = stats_;
  stats.num_running = static_cast<int>(running_.size());
  stats.num_waiting = static_cast<int>(waiting_.size());
  stats.kv_cache_usage = cache_manager_.usage();
  return stats;
}

}  // namespace inferx
