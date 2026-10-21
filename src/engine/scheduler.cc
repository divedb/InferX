#include "inferx/engine/scheduler.h"

#include <algorithm>
#include <cassert>
#include <utility>

namespace inferx {

Scheduler::Scheduler(SchedulerConfig config, std::vector<TokenId> eos_token_ids,
                     attn::AttentionRuntime& attention)
    : config_(config), attention_(&attention),
      eos_token_ids_(std::move(eos_token_ids)),
      waiting_(static_cast<size_t>(config.queue_capacity)) {}

Scheduler::~Scheduler() {
  if (attention_ == nullptr) return;
  for (const Request& request : running_) {
    const Status released = attention_->Cache().Release(request.GetRequestId());
    assert(released.ok());
    (void)released;
  }
}

Status Scheduler::AddRequest(Request request) {
  INFERX_RETURN_IF_ERROR(request.GetSamplingParams().Validate());
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

  // Prefill and replay chunks end on block boundaries while prefix caching
  // is on, so every chunk publishes complete blocks immediately instead of
  // stranding KV in an unhashable partial block until the sequence finishes
  // (vLLM's block-aligned split; SGLang page-aligns chunk ends the same way).
  const auto clip_chunk = [this](int64_t base, int chunk, int total_len) -> int {
    if (!attention_->Cache().PrefixCachingEnabled()) return chunk;
    const int64_t end = base + chunk;
    if (end >= total_len) return chunk;  // The final chunk may end anywhere.
    const int64_t block_size = attention_->GetPlan().block_size;
    const int64_t aligned = end / block_size * block_size;
    return aligned > base ? static_cast<int>(aligned - base) : chunk;
  };

  // Running requests first: each gets one token per step, or the next chunk
  // when existing tokens remain uncomputed. That covers both prefill
  // continuation and preemption replay: a resumed victim must recompute its
  // accepted output too, so the target is NumTokens() (prompt + retained
  // output), not the prompt length (backlog C05). A request the pool cannot
  // cover preempts newer requests to make room (vLLM V1 recompute
  // preemption); when no victim can help it waits a step.
  for (size_t i = 0; i < running_.size() && budget > 0; ++i) {
    Request& request = running_[i];
    const int remaining_tokens = request.NumTokens() - request.NumComputedTokens();
    const int chunk = remaining_tokens > 0
                          ? clip_chunk(request.NumComputedTokens(),
                                       std::min(budget, remaining_tokens), request.NumTokens())
                          : 1;

    const int64_t target = request.NumComputedTokens() + chunk;
    Status granted =
        attention_->Cache().Reserve(request.GetRequestId(), GrowthTarget{target - chunk, target});
    while (!granted.ok() &&
           PreemptNewestAfter(i, &output.preempted_request_ids)) {
      granted = attention_->Cache().Reserve(request.GetRequestId(),
                                            GrowthTarget{target - chunk, target});
    }
    if (!granted.ok()) {
      continue;
    }

    // Block tables are not mirrored into the output: the runner reads them
    // from the facade, their single owner (docs/attention.md §8.3).
    CachedRequestUpdate update;
    update.request_id = request.GetRequestId();
    update.num_computed_tokens = static_cast<int>(target);
    output.scheduled_cached_reqs.push_back(std::move(update));

    output.scheduled.push_back({request.GetRequestId(), chunk});
    output.total_num_scheduled_tokens += chunk;
    budget -= chunk;
    request.SetNumComputedTokens(request.NumComputedTokens() + chunk);
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
    // Adopt the cached prefix first: its blocks are pinned and its tokens
    // count as computed, so the first chunk only schedules the uncached
    // tail. The sequence hashed is prompt plus retained output for a
    // resumed victim; the cap keeps the sequence's last token recomputing
    // so its logits drive the next sample.
    const int total_tokens = request.NumTokens();
    const StatusOr<CacheManager::PrefixAdoption> adopted = attention_->Cache().AdoptPrefix(
        request.GetRequestId(), request.AllTokenIds(), request.CacheSalt(), total_tokens - 1);
    if (!adopted.ok()) return adopted.status();
    const CacheManager::PrefixAdoption& adoption = *adopted;
    if (attention_->Cache().PrefixCachingEnabled()) {
      ++stats_.prefix_queries;
      stats_.prefix_queried_tokens += total_tokens;
      if (adoption.cached_tokens > 0) {
        ++stats_.prefix_hits;
        stats_.prefix_hit_tokens += adoption.cached_tokens;
      }
      request.SetNumCachedTokens(static_cast<int>(adoption.cached_tokens));
    }
    const int chunk =
        clip_chunk(adoption.cached_tokens,
                   std::min(budget, total_tokens - static_cast<int>(adoption.cached_tokens)),
                   total_tokens);

    const Status granted = attention_->Cache().Reserve(
        request.GetRequestId(), GrowthTarget{adoption.cached_tokens,
                                            adoption.cached_tokens + chunk});
    if (!granted.ok()) {
      // Not enough KV for even the first chunk: stop admitting this step and
      // put the request back at the head of the queue. The adoption must
      // come with it, so release the pinned blocks first.
      const Status released = attention_->Cache().Release(request.GetRequestId());
      assert(released.ok());
      (void)released;
      waiting_.PushFront(std::move(request));
      break;
    }

    NewRequestData data;
    data.request_id = request.GetRequestId();
    data.prompt_token_ids = request.AllTokenIds();
    data.sampling_params = request.GetSamplingParams();
    data.num_computed_tokens = static_cast<int>(adoption.cached_tokens);
    output.scheduled_new_reqs.push_back(std::move(data));

    output.scheduled.push_back({request.GetRequestId(), chunk});
    output.total_num_scheduled_tokens += chunk;
    budget -= chunk;
    request.SetNumComputedTokens(static_cast<int>(adoption.cached_tokens) + chunk);

    // Admission completes the kWaiting -> kRunning transition (C05/D3);
    // only a running request may later be preempted or accept samples.
    const Status admitted = request.MarkRunning();
    assert(admitted.ok());
    (void)admitted;
    running_index_[request.GetRequestId()] = running_.size();
    running_.push_back(std::move(request));
  }

  // The immutable snapshot the runner executes against: each scheduled
  // request's whole per-group table, in batch order. Publishing after all
  // reserves means the facade is not read again while a step runs async.
  for (const ScheduledRequest& scheduled : output.scheduled) {
    for (std::uint16_t g = 0; g < attention_->NumGroups() &&
                             g < static_cast<std::uint16_t>(kMaxSchedulerGroups);
         ++g) {
      const auto blocks = attention_->Cache().GroupBlocks(scheduled.request_id, g);
      auto& table = output.group_block_ids[g];
      table.insert(table.end(), blocks.begin(), blocks.end());
    }
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

    // The scheduled chunk just completed: every full block below the
    // watermark now holds computed KV and can join the prefix index. Tokens
    // are the full sequence minus the token sampled below -- including a
    // resumed victim's retained output, whose KV the replay recomputed.
    {
      const Status published = attention_->Cache().PublishComputed(
          id, request.Prompt(), request.Output(), request.NumComputedTokens());
      if (!published.ok()) return published;
    }

    // Samples answer the last token of each allocation, but only allocations
    // that completed every existing token (prompt, plus retained output for
    // a resumed victim) produce a usable token; replay chunks are prefill
    // compute whose sample is dropped, so old tokens are never re-emitted.
    const bool sequence_complete = request.NumComputedTokens() >= request.NumTokens();
    const SampledTokens& sample = result.samples[i];
    if (!sequence_complete || sample.token_ids.empty()) {
      continue;
    }
    if (sample.token_ids.size() != 1) {
      return InternalError("expected one sampled token, got ", sample.token_ids.size());
    }
    const TokenId token = sample.token_ids.front();
    INFERX_RETURN_IF_ERROR(request.AppendOutput(absl::MakeConstSpan(&token, 1)));

    const sampling::SamplingParams& params = request.GetSamplingParams();
    bool finish = false;
    FinishReason reason = FinishReason::kStopped;
    if ((!params.ignore_eos && std::find(eos_token_ids_.begin(), eos_token_ids_.end(),
                                                token) != eos_token_ids_.end()) ||
               (request.NumOutputTokens() >= static_cast<int>(params.min_tokens) &&
                std::find(params.stop_token_ids.begin(), params.stop_token_ids.end(), token) !=
                    params.stop_token_ids.end())) {
      finish = true;
    }
    if (sample.finish_reason.has_value()) {
      finish = true;
      reason = *sample.finish_reason;
    }
    if (!finish && request.NumOutputTokens() >= request.MaxNewTokens()) {
      finish = true;
      reason = FinishReason::kLengthCapped;
    }

    if (!finish) {
      continue;
    }
    const size_t index = it->second;
    INFERX_RETURN_IF_ERROR(attention_->Cache().Release(id));
    INFERX_RETURN_IF_ERROR(request.Finish(reason));
    pending_finish_notifications_.push_back(id);
    finished_.push_back(std::move(running_[index]));
    running_index_.erase(id);
    // Swap-remove; repair the moved request's index entry.
    if (index + 1 < running_.size()) {
      running_[index] = std::move(running_.back());
      running_index_[running_[index].GetRequestId()] = index;
    }
    running_.pop_back();
    ++stats_.num_finished;
  }
  return OkStatus();
}

bool Scheduler::PreemptNewestAfter(size_t keep, std::vector<RequestId>* preempted) {
  if (keep + 1 >= running_.size()) return false;  // Nothing newer to yield.

  const size_t victim_index = running_.size() - 1;
  const RequestId victim_id = running_[victim_index].GetRequestId();
  Request victim = std::move(running_[victim_index]);
  running_.pop_back();
  running_index_.erase(victim.GetRequestId());

  // Cached prefix blocks survive the release: preemption costs the private
  // tail, not the shared prefix, and resumption re-adopts what it can.
  const Status released = attention_->Cache().Release(victim.GetRequestId());
  assert(released.ok());
  (void)released;
  // Admission marked the victim kRunning, so the transition is legal; the
  // watermark resets while the accepted output is kept for replay (C05).
  const Status transition = victim.Preempt();
  assert(transition.ok());
  (void)transition;
  const Status requeued = victim.MarkWaiting();
  assert(requeued.ok());
  (void)requeued;
  waiting_.PushFront(std::move(victim));

  preempted->push_back(victim_id);
  ++stats_.num_preempted;
  return true;
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
    const Status released = attention_->Cache().Release(id);
    assert(released.ok());
    (void)released;
    const Status finished = request.Finish(FinishReason::kAborted);
    assert(finished.ok());
    (void)finished;
    pending_finish_notifications_.push_back(id);
    finished_.push_back(std::move(running_[index]));
    running_index_.erase(id);
    if (index + 1 < running_.size()) {
      running_[index] = std::move(running_.back());
      running_index_[running_[index].GetRequestId()] = index;
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
  stats.kv_cache_usage = attention_->Cache().Usage();
  return stats;
}

}  // namespace inferx
