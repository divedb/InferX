#include "inferx/engine/engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>

#include "inferx/api/openai.h"
#include "inferx/core/device_runtime.h"
#include "inferx/model/deepseek_v2.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/gpt_oss.h"
#include "inferx/observe/metrics.h"
#include "inferx/support/log.h"
#include "kv_autosize.h"
#include "model_runner.h"
#include "qwen2_runner.h"
#include "sync_model_runner.h"

namespace inferx::engine {
namespace {

using model::ForwardBatch;
using scheduler::FinishReason;
using scheduler::RequestId;
using scheduler::Scheduler;
using scheduler::TokenDelta;

// How long the loop sleeps when there is nothing to run. Short enough that a
// newly submitted request is picked up promptly, long enough that an idle
// server is not spinning a core -- and only reached when the scheduler is
// genuinely empty, since otherwise the loop is bounded by the step itself.
constexpr auto kIdleWait = std::chrono::milliseconds(2);

// When an idle engine receives a burst, Submit calls arrive one by one even if
// their callers regard them as concurrent. Waking on the first and immediately
// forming a batch makes membership depend on thread timing; a faster prefill
// made that visible as different greedy continuations for identical bursts.
// Wait for a short quiet period only at the idle-to-busy transition. Decode
// steps and arrivals while work is already running pay no delay.
constexpr auto kBatchCoalesceWait = std::chrono::milliseconds(1);

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

// Full sampling for one host-resident logits row, used by the synchronous
// model paths (DeepSeek-V2, gpt-oss) that ship logits back anyway. Mirrors
// the device sampler's semantics: penalties and min-tokens masks mutate the
// row first, then greedy or a seeded truncated draw, then logprobs at
// temperature 1 over the post-penalty logits.
int32_t HostSampleRow(float* row, int64_t vocab,
                      const model::ForwardBatch& batch, size_t i,
                      SampledLogprob* logprob_out) {
  using FB = model::ForwardBatch;
  const auto at_f = [&](const std::vector<float>& v, float fallback) {
    return i < v.size() ? v[i] : fallback;
  };

  const float presence = at_f(batch.presence_penalty, 0.0f);
  const float frequency = at_f(batch.frequency_penalty, 0.0f);
  const float repetition = at_f(batch.repetition_penalty, 1.0f);
  const size_t hist_cap = static_cast<size_t>(FB::kPenaltyHistoryCap);
  if ((presence != 0.0f || frequency != 0.0f || repetition != 1.0f) &&
      batch.penalty_history_ids.size() >= (i + 1) * hist_cap) {
    for (size_t j = 0; j < hist_cap; ++j) {
      const int32_t id = batch.penalty_history_ids[i * hist_cap + j];
      if (id < 0 || static_cast<int64_t>(id) >= vocab) continue;
      float& value = row[id];
      if (repetition != 1.0f) {
        value = value > 0.0f ? value / repetition : value * repetition;
      }
      value -= presence;
      value -= frequency * static_cast<float>(
                               batch.penalty_history_counts[i * hist_cap + j]);
    }
  }

  const size_t mask_cap = static_cast<size_t>(FB::kMaskCap);
  if (batch.mask_token_ids.size() >= (i + 1) * mask_cap) {
    for (size_t j = 0; j < mask_cap; ++j) {
      const int32_t id = batch.mask_token_ids[i * mask_cap + j];
      if (id >= 0 && static_cast<int64_t>(id) < vocab) {
        row[id] = -std::numeric_limits<float>::infinity();
      }
    }
  }

  const float temperature = at_f(batch.temperature, 0.0f);
  int32_t chosen = 0;
  if (temperature <= 0.0f) {
    chosen = static_cast<int32_t>(std::max_element(row, row + vocab) - row);
  } else {
    const float top_p = at_f(batch.top_p, 1.0f);
    const int32_t top_k = i < batch.top_k.size() ? batch.top_k[i] : 0;
    const float min_p = at_f(batch.min_p, 0.0f);
    const uint64_t seed = i < batch.seeds.size() ? batch.seeds[i] : 0;

    const float inv_t = 1.0f / temperature;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) {
      max_logit = std::max(max_logit, row[v] * inv_t);
    }
    std::vector<std::pair<float, int32_t>> probs;
    probs.reserve(static_cast<size_t>(vocab));
    double total = 0.0;
    for (int64_t v = 0; v < vocab; ++v) {
      const float p = std::exp(row[v] * inv_t - max_logit);
      total += p;
      probs.emplace_back(p, static_cast<int32_t>(v));
    }

    // Descending by probability; index breaks ties so the order is total.
    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });

    // The surviving prefix under all three truncations at once.
    size_t keep = probs.size();
    if (top_k > 0) keep = std::min<size_t>(keep, static_cast<size_t>(top_k));
    if (min_p > 0.0f) {
      const double floor_prob = static_cast<double>(min_p) * probs[0].first;
      size_t n = 0;
      while (n < keep && probs[n].first >= floor_prob) ++n;
      keep = std::max<size_t>(n, 1);
    }
    double mass = 0.0;
    if (top_p < 1.0f) {
      size_t n = 0;
      while (n < keep) {
        mass += probs[n].first / total;
        ++n;
        if (mass >= top_p) break;
      }
      keep = std::max<size_t>(n, 1);
    } else {
      for (size_t n = 0; n < keep; ++n) mass += probs[n].first / total;
    }

    std::mt19937_64 rng(seed);
    const double target =
        std::uniform_real_distribution<double>(0.0, mass)(rng);
    double running = 0.0;
    chosen = probs[keep - 1].second;
    for (size_t n = 0; n < keep; ++n) {
      running += probs[n].first / total;
      if (running >= target) {
        chosen = probs[n].second;
        break;
      }
    }
  }

  const int32_t logprob_k =
      i < batch.logprobs_k.size() ? batch.logprobs_k[i] : -1;
  if (logprob_out != nullptr && logprob_k >= 0) {
    float raw_max = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) raw_max = std::max(raw_max, row[v]);
    double raw_total = 0.0;
    for (int64_t v = 0; v < vocab; ++v) {
      raw_total += std::exp(row[v] - raw_max);
    }
    const float log_z = static_cast<float>(std::log(raw_total)) + raw_max;

    logprob_out->present = true;
    logprob_out->logprob = row[chosen] - log_z;

    const size_t want =
        std::min<size_t>(static_cast<size_t>(logprob_k),
                         static_cast<size_t>(FB::kMaxTopLogprobs));
    if (want > 0) {
      std::vector<int32_t> order(static_cast<size_t>(vocab));
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + want, order.end(),
                        [&](int32_t a, int32_t b) {
                          return row[a] != row[b] ? row[a] > row[b] : a < b;
                        });
      for (size_t j = 0; j < want; ++j) {
        logprob_out->top.emplace_back(order[j], row[order[j]] - log_z);
      }
    }
  }

  return chosen;
}

struct ServingMetrics {
  observe::Registry registry;
  std::shared_ptr<observe::Counter> accepted;
  std::array<std::shared_ptr<observe::Counter>, 6> finished;
  std::shared_ptr<observe::Counter> prompt_tokens;
  std::shared_ptr<observe::Counter> generation_tokens;
  std::shared_ptr<observe::Histogram> queue;
  std::shared_ptr<observe::Histogram> ttft;
  std::shared_ptr<observe::Histogram> itl;
  std::shared_ptr<observe::Histogram> duration;
  std::shared_ptr<observe::Histogram> prompt_length;
  std::shared_ptr<observe::Histogram> generation_length;
  std::vector<std::shared_ptr<observe::Histogram>> prefill_step;
  std::vector<std::shared_ptr<observe::Histogram>> decode_step;

  ServingMetrics() {
    accepted = registry.AddCounter(
        "inferx_requests_total", "Requests by terminal outcome and reason.",
        {{"finish_reason", ""}, {"outcome", "accepted"}});
    for (int i = 1; i <= 5; ++i) {
      const auto reason = static_cast<FinishReason>(i);
      std::string outcome = "failed";
      if (reason == FinishReason::kStopToken ||
          reason == FinishReason::kMaxTokens) {
        outcome = "completed";
      } else if (reason == FinishReason::kCancelled) {
        outcome = "cancelled";
      }
      finished[static_cast<size_t>(i)] = registry.AddCounter(
          "inferx_requests_total", "Requests by terminal outcome and reason.",
          {{"finish_reason", scheduler::FinishReasonName(reason)},
           {"outcome", std::move(outcome)}});
    }
    prompt_tokens = registry.AddCounter("inferx_prompt_tokens_total",
                                        "Accepted prompt tokens.");
    generation_tokens = registry.AddCounter("inferx_generation_tokens_total",
                                            "Generated output tokens.");
    queue = registry.AddHistogram(
        "inferx_request_queue_seconds", "Time from acceptance to admission.",
        {.0001, .001, .005, .01, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60});
    ttft = registry.AddHistogram("inferx_time_to_first_token_seconds",
                                 "Time from acceptance to first output token.",
                                 {.001, .002, .005, .01, .02, .04, .06, .08, .1,
                                  .2, .4, .8, 1, 2, 4, 8, 16, 32, 64});
    itl = registry.AddHistogram(
        "inferx_inter_token_latency_seconds",
        "Time between consecutive generated tokens for a request.",
        {.001, .002, .004, .006, .008, .01, .015, .02, .025, .03, .04, .06, .08,
         .1, .2, .4, 1, 2});
    duration = registry.AddHistogram(
        "inferx_request_duration_seconds",
        "Time from acceptance to terminal completion.",
        {.1, .25, .5, 1, 2.5, 5, 10, 20, 40, 60, 120, 300, 600});
    const std::vector<double> token_buckets = {
        1,   2,   4,    8,    16,   32,   64,    128,
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
    prompt_length = registry.AddHistogram("inferx_request_prompt_tokens",
                                          "Prompt tokens per accepted request.",
                                          token_buckets);
    generation_length = registry.AddHistogram(
        "inferx_request_generation_tokens",
        "Generated tokens per terminal request.", token_buckets);
  }

  void RecordFinish(FinishReason reason, int32_t generated,
                    std::chrono::steady_clock::time_point submitted) {
    const size_t index = static_cast<size_t>(reason);
    if (index > 0 && index < finished.size()) finished[index]->Increment();
    generation_length->Observe(generated);
    duration->Observe(
        SecondsSince(submitted, std::chrono::steady_clock::now()));
  }

  void ConfigureRanks(const std::vector<RankTelemetry>& ranks) {
    const std::vector<double> buckets = {.00001, .000025, .00005, .0001, .00025,
                                         .0005,  .001,    .0025,  .005,  .01,
                                         .025,   .05,     .1};
    for (const RankTelemetry& rank : ranks) {
      const observe::Labels prefill_labels = {
          {"phase", "prefill"}, {"rank", std::to_string(rank.rank)}};
      const observe::Labels decode_labels = {
          {"phase", "decode"}, {"rank", std::to_string(rank.rank)}};
      prefill_step.push_back(registry.AddHistogram(
          "inferx_engine_step_seconds", "Device step duration by rank.",
          buckets, prefill_labels));
      decode_step.push_back(registry.AddHistogram(
          "inferx_engine_step_seconds", "Device step duration by rank.",
          buckets, decode_labels));
    }
  }

  void RecordRankSteps(const std::vector<RankTelemetry>& ranks, bool prefill) {
    auto& histograms = prefill ? prefill_step : decode_step;
    for (size_t i = 0; i < ranks.size() && i < histograms.size(); ++i) {
      histograms[i]->Observe(ranks[i].last_step_device_ms / 1000.0);
    }
  }
};

std::string RenderRankMetrics(const std::vector<RankTelemetry>& ranks) {
  if (ranks.empty()) return {};
  observe::Registry registry;
  const std::string backend = comm::CommBackendName(ranks.front().backend);
  registry
      .AddGauge("inferx_communicator_info",
                "Active communicator backend and world size.",
                {{"backend", backend},
                 {"world_size", std::to_string(ranks.front().world_size)}})
      ->Set(1);
  for (const RankTelemetry& rank : ranks) {
    const observe::Labels rank_label = {{"rank", std::to_string(rank.rank)}};
    registry
        .AddGauge("inferx_rank_healthy", "Whether the rank is operational.",
                  rank_label)
        ->Set(rank.healthy ? 1 : 0);
    registry
        .AddGauge("inferx_rank_last_progress_seconds",
                  "Seconds since the rank last completed worker work.",
                  rank_label)
        ->Set(rank.last_progress_age_seconds);
    registry
        .AddGauge("inferx_rank_last_step_seconds",
                  "Most recent device step duration for the rank.", rank_label)
        ->Set(rank.last_step_device_ms / 1000.0);
    registry
        .AddCounter("inferx_rank_timeouts_total", "Rank worker timeouts.",
                    rank_label)
        ->Increment(rank.timeouts);
    const observe::Labels collective_labels = {
        {"backend", backend},
        {"op", "all_reduce_sum"},
        {"rank", std::to_string(rank.rank)}};
    registry
        .AddCounter("inferx_collectives_total", "Collective calls by rank.",
                    collective_labels)
        ->Increment(rank.communication.all_reduce_calls);
    registry
        .AddCounter("inferx_collective_bytes_total",
                    "Collective payload bytes by rank.", collective_labels)
        ->Increment(rank.communication.all_reduce_bytes);
    registry
        .AddCounter("inferx_collective_failures_total",
                    "Collective failures by rank.", collective_labels)
        ->Increment(rank.communication.collective_failures);
    auto latency = registry.AddHistogram(
        "inferx_collective_seconds",
        "Sampled CUDA-event collective duration by rank.",
        std::vector<double>(comm::kCollectiveLatencyBuckets.begin(),
                            comm::kCollectiveLatencyBuckets.end()),
        collective_labels);
    latency->SetSnapshot(
        std::vector<uint64_t>(rank.communication.latency_buckets.begin(),
                              rank.communication.latency_buckets.end()),
        rank.communication.latency_count,
        rank.communication.latency_sum_seconds);
    registry
        .AddCounter("inferx_collective_timing_samples_dropped_total",
                    "Collective timing samples dropped before observation.",
                    collective_labels)
        ->Increment(rank.communication.timing_samples_dropped);
    registry
        .AddCounter("inferx_collective_timing_graph_skips_total",
                    "Collective timing samples skipped during graph capture.",
                    collective_labels)
        ->Increment(rank.communication.timing_graph_skips);
    registry
        .AddCounter("inferx_communicator_aborts_total",
                    "Communicator abort calls by rank.",
                    {{"backend", backend}, {"rank", std::to_string(rank.rank)}})
        ->Increment(rank.communication.aborts);
  }
  return registry.Render();
}

}  // namespace

// --- Engine ------------------------------------------------------------------

struct Engine::Impl {
  EngineConfig config;

  // Exactly one of these is populated, decided once in Engine::Create from the
  // checkpoint's declared architecture. Qwen2Runner hides whether execution is
  // direct TP=1 or dispatched to NCCL rank workers.
  //
  // The gpt-oss path is the Phase 3 slow serve: GptOssModel has only Forward(),
  // no paged KV, no continuous batching. It is batch-1, full recompute per
  // token, and exists so the engine can serve the checkpoint end to end before
  // Phase 4 makes it fast.
  std::unique_ptr<ModelRunner> model_runner;

  // The declared architecture, kept because composition needs it after Create
  // (chat-template selection) and the populated pointer alone cannot
  // distinguish the two synchronous models.
  model::Architecture architecture = model::Architecture::kQwen2;

  // Null in the gpt-oss path, which does not continuous-batch. Held by pointer
  // rather than by value for the same reason the models are: the Qwen2 path
  // keeps its direct access pattern, the gpt-oss path pays nothing for it.
  std::unique_ptr<Scheduler> scheduler;

  std::unique_ptr<tokenizer::Tokenizer> tokenizer;

  std::string model_name;

  // Everything below is shared between the request threads and the loop.
  mutable std::mutex mutex;
  std::condition_variable wake;

  struct Pending {
    RequestId id;
    std::vector<int32_t> prompt;
    scheduler::SamplingParams params;
    std::chrono::steady_clock::time_point submitted;

    // Held here rather than in `params` because the scheduler never sees text
    // (§3.1); matching these is this layer's job.
    std::vector<std::string> stop;

    std::shared_ptr<Generation> generation;
  };

  std::deque<Pending> intake;
  std::atomic<bool> stopping{false};
  RequestId next_id = 1;

  Engine::Stats stats;
  ServingMetrics metrics;

  std::thread loop;

  // Per-request state the *scheduler* must not hold, because holding it would
  // mean holding text (§3.1). Only the loop thread touches this.
  struct Active {
    std::shared_ptr<Generation> generation;
    std::unique_ptr<tokenizer::IncrementalDecoder> decoder;

    /// Everything decoded so far, and how much of it has been emitted. The gap
    /// between the two is text withheld because it might yet turn out to be
    /// the start of a stop sequence.
    std::string text;
    size_t emitted = 0;

    std::vector<std::string> stop;
    bool stop_hit = false;
    /// Keep the matched stop string in the output instead of truncating
    /// before it (vLLM's include_stop_str_in_output).
    bool include_stop_str = false;
    /// Emit one event per token even when it decodes to no text, so each
    /// token's logprob reaches the client.
    bool want_logprobs = false;
    int32_t generated = 0;
    std::chrono::steady_clock::time_point submitted;
    std::chrono::steady_clock::time_point last_token;
    bool admitted = false;
    bool produced_token = false;
  };

  std::unordered_map<RequestId, Active> active;

  Impl() = default;

  // Routes one sampled token to its request: detokenize, apply stop sequences,
  // emit whatever is now safe to send.
  void Deliver(const TokenDelta& delta) {
    const auto it = active.find(delta.id);

    if (it == active.end()) return;

    Active& state = it->second;

    const auto now = std::chrono::steady_clock::now();
    if (state.produced_token) {
      metrics.itl->Observe(SecondsSince(state.last_token, now));
    } else {
      metrics.ttft->Observe(SecondsSince(state.submitted, now));
      state.produced_token = true;
    }
    state.last_token = now;

    ++state.generated;
    metrics.generation_tokens->Increment();

    state.text += state.decoder->Push(delta.token);

    // Every emitted event carries this step's token and, when asked for, its
    // logprob -- the client-side logprobs array is per token, not per text
    // delta.
    const auto attach = [&](Generation::Event* event) {
      event->generated = state.generated;
      event->token = delta.token;
      if (delta.has_logprob) {
        event->has_logprob = true;
        event->logprob = delta.logprob;
        event->top_logprobs = delta.top_logprobs;
      }
    };

    // A stop sequence ends the generation even though the scheduler, which
    // never sees text, has no idea it happened. Cancelling here retires the
    // sequence on the next step and frees its blocks.
    if (!state.stop.empty()) {
      // The earliest match, and its needle: include_stop_str_in_output keeps
      // the matched text, so the offset alone is not enough.
      size_t at = std::string::npos;
      size_t match_len = 0;
      for (const std::string& needle : state.stop) {
        if (needle.empty()) continue;
        const size_t pos = state.text.find(needle);
        if (pos < at) {
          at = pos;
          match_len = needle.size();
        }
      }

      if (at != std::string::npos) {
        state.text.resize(state.include_stop_str ? at + match_len : at);
        state.stop_hit = true;

        if (state.emitted < state.text.size() || state.want_logprobs) {
          Generation::Event event;
          event.text =
              state.text.substr(std::min(state.emitted, state.text.size()));
          attach(&event);

          state.generation->Emit(std::move(event));
          state.emitted = state.text.size();
        }

        // Tell whichever path owns this request to retire it. The Qwen2 loop
        // learns through the scheduler and retires on the next step; the
        // gpt-oss loop has no scheduler, so retiring here is what lets its
        // post-Deliver `active.find(id) == end()` check notice the stop.
        if (scheduler != nullptr) {
          scheduler->Cancel(delta.id);
        } else {
          Retire(delta.id, FinishReason::kStopToken);
        }
        return;
      }
    }

    // Withhold any suffix that could still grow into a stop sequence. Once
    // bytes are on the wire they cannot be recalled, and emitting the start of
    // a stop string means the client sees text it asked to have removed.
    const size_t hold = api::StopSequenceHoldback(state.text, state.stop);
    const size_t safe = state.text.size() - hold;

    if (safe > state.emitted || state.want_logprobs) {
      Generation::Event event;
      if (safe > state.emitted) {
        event.text = state.text.substr(state.emitted, safe - state.emitted);
        state.emitted = safe;
      }
      attach(&event);

      state.generation->Emit(std::move(event));
    }
  }

  void Retire(RequestId id, FinishReason reason) {
    const auto it = active.find(id);

    if (it == active.end()) return;

    Active& state = it->second;

    // Flush whatever the incremental decoder is still holding, unless a stop
    // sequence already truncated the output -- in which case the held bytes are
    // part of the stop string and must not be sent.
    if (!state.stop_hit) {
      state.text += state.decoder->Flush();

      if (state.emitted < state.text.size()) {
        Generation::Event event;
        event.text = state.text.substr(state.emitted);
        event.generated = state.generated;

        state.generation->Emit(std::move(event));
        state.emitted = state.text.size();
      }
    }

    // A stop *string* is a stop, not a cancellation: the request finished the
    // way the caller asked it to.
    const FinishReason reported =
        state.stop_hit ? FinishReason::kStopToken : reason;

    state.generation->Finish(reported, state.generated);
    metrics.RecordFinish(reported, state.generated, state.submitted);
    active.erase(it);
  }

  void RecordAdmissions() {
    const auto now = std::chrono::steady_clock::now();
    for (const RequestId id : scheduler->TakeAdmitted()) {
      const auto it = active.find(id);
      if (it == active.end() || it->second.admitted) continue;
      it->second.admitted = true;
      metrics.queue->Observe(SecondsSince(it->second.submitted, now));
    }
  }

  // Drives one short request all the way through, before anything real is
  // served.
  //
  // This exists for CUDA graph capture, and the reason is subtle enough to be
  // worth stating: capture records device addresses, and several things the
  // step depends on -- the FP8 activation buffer, FlashInfer's workspace, the
  // cuBLASLt plan for each shape -- are allocated on first use. Capturing
  // before that first use bakes in addresses that are not yet the ones the
  // model will run with, and the result is not an error but a graph that
  // replays against the wrong memory and emits fluent nonsense. Measured: with
  // capture before warm-up, "The capital of France is" continued " Paris" and
  // then repeated a single token forever.
  //
  // Running it through the scheduler rather than poking the model directly
  // keeps the KV bookkeeping honest -- the sequence retires normally and its
  // blocks go back to the pool, so the warm-up leaves no trace.
  Status Warmup() {
    // Long enough to span several blocks and to make the sequences grow into a
    // new one while decoding, so the warm-up drives the same code paths real
    // traffic will.
    constexpr int64_t kWarmupPromptTokens = 40;
    constexpr int32_t kWarmupSteps = 12;

    scheduler::SamplingParams params;
    params.max_tokens = kWarmupSteps;

    // As many sequences as will ever run at once, because the shapes that get
    // captured are per batch size and every one of them has to have been
    // *executed* before it is recorded.
    for (int64_t i = 0; i < config.scheduler.max_running; ++i) {
      // Token 0 exists in every vocabulary; the values do not matter, only the
      // shapes they drive.
      std::vector<int32_t> prompt(kWarmupPromptTokens, 0);

      INFERX_RETURN_IF_ERROR(
          scheduler->AddRequest(next_id++, std::move(prompt), params));
    }

    ForwardBatch batch;
    std::vector<int32_t> sampled;

    while (scheduler->HasWork()) {
      INFERX_RETURN_IF_ERROR(scheduler->PrepareStep(&batch));

      if (batch.token_ids.empty()) break;

      std::vector<SampledLogprob> logprobs;
      INFERX_RETURN_IF_ERROR(model_runner->Step(batch, &sampled, &logprobs));
      INFERX_RETURN_IF_ERROR(scheduler->CommitStep(sampled, nullptr));

      (void)scheduler->TakeCompleted();
    }

    (void)scheduler->TakeCompleted();

    return OkStatus();
  }

  void Run() {
    RunModel();

    // Requests can still be in the intake queue when shutdown wins the race
    // with the loop. Close those streams and account for them just like active
    // cancellations; otherwise their consumers wait forever and accepted
    // requests never receive a terminal outcome.
    std::deque<Pending> abandoned;
    {
      std::lock_guard<std::mutex> lock(mutex);
      abandoned.swap(intake);
    }
    for (Pending& pending : abandoned) {
      pending.generation->Finish(FinishReason::kCancelled, 0);
      metrics.RecordFinish(FinishReason::kCancelled, 0, pending.submitted);
    }

    // Shutting down: nothing else will ever produce for these streams.
    for (auto& [id, state] : active) {
      state.generation->Finish(FinishReason::kCancelled, state.generated);
      metrics.RecordFinish(FinishReason::kCancelled, state.generated,
                           state.submitted);
    }

    active.clear();
  }

  // Continuous batching over a paged KV cache. Architecture runners differ in
  // how a step produces samples, while admission, scheduling, commit, delivery,
  // completion, and accounting remain shared here.
  void RunModel() {
    ForwardBatch batch;
    std::vector<int32_t> sampled;
    std::vector<TokenDelta> deltas;

    while (!stopping.load(std::memory_order_relaxed)) {
      {
        std::unique_lock<std::mutex> lock(mutex);

        if (!intake.empty() && !scheduler->HasWork()) {
          size_t observed = intake.size();
          auto quiet_until =
              std::chrono::steady_clock::now() + kBatchCoalesceWait;

          while (!stopping.load(std::memory_order_relaxed)) {
            if (wake.wait_until(lock, quiet_until) == std::cv_status::timeout) {
              break;
            }

            if (intake.size() != observed) {
              observed = intake.size();
              quiet_until =
                  std::chrono::steady_clock::now() + kBatchCoalesceWait;
            }
          }
        }

        while (!intake.empty()) {
          Pending pending = std::move(intake.front());
          intake.pop_front();

          // Admission can fail -- a prompt longer than max_seq_len, say -- and
          // the caller is blocked on the stream, so the refusal has to arrive
          // there rather than only in a log.
          const Status added = scheduler->AddRequest(
              pending.id, std::move(pending.prompt), pending.params);

          if (!added.ok()) {
            lock.unlock();
            pending.generation->Finish(FinishReason::kOutOfMemory, 0);
            metrics.RecordFinish(FinishReason::kOutOfMemory, 0,
                                 pending.submitted);
            lock.lock();
            continue;
          }

          Active state;
          state.generation = pending.generation;
          state.decoder = std::make_unique<tokenizer::IncrementalDecoder>(
              tokenizer.get(),
              /*skip_special=*/!pending.params.keep_special_tokens);
          state.stop = std::move(pending.stop);
          state.include_stop_str = pending.params.include_stop_str_in_output;
          state.want_logprobs = pending.params.want_logprobs;
          state.submitted = pending.submitted;

          active.emplace(pending.id, std::move(state));
        }

        if (!scheduler->HasWork()) {
          wake.wait_for(lock, kIdleWait);
          continue;
        }
      }

      // Disconnects, noticed once per step. A cancelled request is retired by
      // the scheduler on the step after this one.
      for (const auto& [id, state] : active) {
        if (state.generation->cancelled()) scheduler->Cancel(id);
      }

      if (const Status prepared = scheduler->PrepareStep(&batch);
          !prepared.ok()) {
        FailAll(prepared);
        continue;
      }
      RecordAdmissions();

      if (!batch.token_ids.empty()) {
        const bool wants_logprobs =
            std::any_of(batch.logprobs_k.begin(), batch.logprobs_k.end(),
                        [](int32_t k) { return k >= 0; });

        std::vector<SampledLogprob> raw_logprobs;
        std::vector<scheduler::StepLogprob> step_logprobs;
        const Status stepped =
            model_runner->Step(batch, &sampled, &raw_logprobs);
        if (stepped.ok() && wants_logprobs) {
          step_logprobs.resize(raw_logprobs.size());
          for (size_t i = 0; i < raw_logprobs.size(); ++i) {
            step_logprobs[i] = {raw_logprobs[i].present,
                                raw_logprobs[i].logprob,
                                std::move(raw_logprobs[i].top)};
          }
        }

        // A device error is not recoverable per-request: the KV cache state is
        // now unknown, so every in-flight sequence is suspect. Failing them all
        // and stopping is honest; continuing would serve corrupted output.
        if (!stepped.ok()) {
          FailAll(stepped);
          stopping.store(true, std::memory_order_relaxed);
          break;
        }

        if (const Status committed = scheduler->CommitStep(
                sampled, &deltas, wants_logprobs ? &step_logprobs : nullptr);
            !committed.ok()) {
          FailAll(committed);
          stopping.store(true, std::memory_order_relaxed);
          break;
        }

        for (const TokenDelta& delta : deltas) Deliver(delta);
        metrics.RecordRankSteps(model_runner->telemetry(),
                                batch.num_tokens() > batch.num_seqs);

        std::lock_guard<std::mutex> lock(mutex);
        ++stats.steps;
        stats.tokens_generated += static_cast<int64_t>(deltas.size());
        stats.last_step_ms = model_runner->last_step_device_ms();
      }

      for (const scheduler::Completion& completion :
           scheduler->TakeCompleted()) {
        Retire(completion.id, completion.reason);
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        stats.running = scheduler->num_running();
        stats.waiting = scheduler->num_waiting();
        stats.blocks_in_use = scheduler->blocks_in_use();
        stats.preemptions = scheduler->preemptions();
        stats.cached_blocks = scheduler->cached_blocks();
        stats.prefix_hit_tokens = scheduler->prefix_hit_tokens();
        stats.prefix_miss_tokens = scheduler->prefix_miss_tokens();
        stats.evicted_blocks = scheduler->evicted_blocks();
      }
    }
  }

  void FailAll(const Status& why) {
    // A model-step failure is fatal to every active sequence. Keep the reason
    // visible in the server log: the HTTP streams can only express a generic
    // finish reason, and discarding `why` made GPU/kernel failures look like a
    // clean shutdown during serving benchmarks.
    LOG(ERROR) << "engine step failed: " << why;

    for (auto& [id, state] : active) {
      state.generation->Finish(FinishReason::kOutOfMemory, state.generated);
      metrics.RecordFinish(FinishReason::kOutOfMemory, state.generated,
                           state.submitted);
    }

    active.clear();
  }
};

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Engine::~Engine() {
  if (impl_ == nullptr) return;

  impl_->stopping.store(true, std::memory_order_relaxed);
  impl_->wake.notify_all();

  if (impl_->loop.joinable()) impl_->loop.join();
}

StatusOr<std::unique_ptr<Engine>> Engine::Create(const EngineConfig& config) {
  if (config.model_dir.empty()) {
    return InvalidArgumentError("model_dir is empty");
  }
  if (config.fp8_weights && config.int4_weights) {
    return InvalidArgumentError(
        "fp8_weights and int4_weights are mutually exclusive");
  }
  if (config.tensor_parallel_size != 1 && config.tensor_parallel_size != 2) {
    return InvalidArgumentError("tensor_parallel_size must be 1 or 2, got ",
                                config.tensor_parallel_size);
  }
  if (static_cast<int>(config.devices.size()) != config.tensor_parallel_size) {
    return InvalidArgumentError("devices has ", config.devices.size(),
                                " entries but tensor_parallel_size is ",
                                config.tensor_parallel_size);
  }
  if (config.comm_backend != "single" && config.comm_backend != "nccl") {
    return InvalidArgumentError("unknown communication backend ",
                                config.comm_backend);
  }
  if ((config.tensor_parallel_size == 1) != (config.comm_backend == "single")) {
    return InvalidArgumentError(
        "TP=1 requires comm_backend=single and TP=2 requires nccl");
  }
  if (config.comm_backend == "nccl" &&
      config.device_kind != DeviceKind::kCuda) {
    return InvalidArgumentError("comm_backend=nccl requires device_kind=cuda");
  }
  const DeviceId primary_device = DeviceId::For(
      config.device_kind, static_cast<int8_t>(config.devices.front()));
  INFERX_ASSIGN_OR_RETURN(DeviceRuntime * runtime, RuntimeFor(primary_device));
  INFERX_RETURN_IF_ERROR(runtime->SetDevice(primary_device));

  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<tokenizer::Tokenizer> tok,
                          tokenizer::Tokenizer::LoadFrom(config.model_dir));

  // Architecture dispatch. The checkpoint's config.json says which model class
  // to build, and the two classes expose different contracts: Qwen2Model has
  // the paged-KV / continuous-batching serving interface, GptOssModel has only
  // Forward(). Reading the config here rather than letting Qwen2Model::Load
  // reject gpt-oss keeps the error message honest ("we serve this with a
  // different class") rather than misattributing it to a shape mismatch.
  INFERX_ASSIGN_OR_RETURN(const model::ModelConfig model_config,
                          model::ModelConfig::FromDirectory(config.model_dir));

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->tokenizer = std::move(tok);
  impl->architecture = model_config.architecture;

  impl->model_name =
      config.served_model_name.empty()
          ? std::filesystem::path(config.model_dir).filename().string()
          : config.served_model_name;

  if (model_config.architecture == model::Architecture::kDeepSeekV2) {
    if (config.tensor_parallel_size != 1) {
      return UnimplementedError(
          "tensor parallel serving is currently implemented for Qwen2 only");
    }
    // A quantization request that would be silently ignored is a startup
    // error: the caller asked for behavior this arm cannot deliver.
    if (config.fp8_weights || config.int4_weights || config.fp8_kv_cache) {
      return InvalidArgumentError(
          "--quantization / --kv-cache-dtype are not supported for the "
          "DeepSeek-V2 architecture");
    }
    // --enforce-eager's default is accepted and *skipped* rather than
    // attempted: the unabsorbed MLA path sizes its scratch by context length,
    // so CaptureDecodeGraph is Unimplemented by design (§18.7 D4), and
    // failing startup over a documented limitation would be wrong.
    if (config.capture_graphs) {
      LOG(INFO) << "DeepSeek-V2 serves without CUDA graphs by design; "
                   "graph capture is skipped";
    }
    INFERX_ASSIGN_OR_RETURN(
        model::DeepseekV2Model deepseek,
        model::DeepseekV2Model::Load(config.model_dir, primary_device));

    // The latent cache: 576 elements/token/layer for V2-Lite against a GQA
    // model's thousands, so the same block count buys ~14x the cached context.
    INFERX_ASSIGN_OR_RETURN(
        const int64_t kv_blocks,
        AutosizeKvBlocksOnDevice(
            KvSizingSpec{
                .explicit_blocks = config.kv_blocks,
                .explicit_bytes = config.kv_cache_memory_bytes,
                .gpu_memory_utilization = config.gpu_memory_utilization,
                .block_bytes = deepseek.KvBlockBytes(config.block_size),
                .min_blocks =
                    (config.scheduler.max_seq_len + config.block_size - 1) /
                    config.block_size},
            primary_device));
    INFERX_RETURN_IF_ERROR(
        deepseek.AttachKvCache(kv_blocks, config.block_size));

    INFERX_ASSIGN_OR_RETURN(
        Scheduler scheduler,
        Scheduler::Create(config.scheduler, deepseek.kv_pool()));

    impl->model_runner =
        MakeSyncModelRunner(std::move(deepseek), HostSampleRow);
    impl->scheduler = std::make_unique<Scheduler>(std::move(scheduler));
    impl->stats.blocks_total = kv_blocks;

    INFERX_RETURN_IF_ERROR(impl->model_runner->ReserveActivations(
        config.scheduler.max_batch_tokens));
  } else if (model_config.architecture == model::Architecture::kGptOss) {
    if (config.tensor_parallel_size != 1) {
      return UnimplementedError(
          "tensor parallel serving is currently implemented for Qwen2 only");
    }
    // A quantization request that would be silently ignored is a startup
    // error; gpt-oss weights are MXFP4 from the checkpoint and the Qwen2-path
    // quantization methods do not exist on this class.
    if (config.fp8_weights || config.int4_weights || config.fp8_kv_cache) {
      return InvalidArgumentError(
          "--quantization / --kv-cache-dtype are not supported for the "
          "gpt-oss architecture");
    }

    INFERX_ASSIGN_OR_RETURN(
        model::GptOssModel gpt_oss,
        model::GptOssModel::Load(config.model_dir, primary_device));

    // Attach a paged KV cache so the scheduler can batch multiple sequences.
    // gpt-oss has 24 layers × 8 kv_heads × 64 head_dim × 2 (bf16) = 2048
    // bytes/token/layer; the pool sizing is the same VRAM/latency trade as
    // Qwen2, auto-sized unless pinned.
    INFERX_ASSIGN_OR_RETURN(
        const int64_t kv_blocks,
        AutosizeKvBlocksOnDevice(
            KvSizingSpec{
                .explicit_blocks = config.kv_blocks,
                .explicit_bytes = config.kv_cache_memory_bytes,
                .gpu_memory_utilization = config.gpu_memory_utilization,
                .block_bytes = gpt_oss.KvBlockBytes(config.block_size),
                .min_blocks =
                    (config.scheduler.max_seq_len + config.block_size - 1) /
                    config.block_size},
            primary_device));
    INFERX_RETURN_IF_ERROR(gpt_oss.AttachKvCache(kv_blocks, config.block_size));

    INFERX_ASSIGN_OR_RETURN(
        Scheduler scheduler,
        Scheduler::Create(config.scheduler, gpt_oss.kv_pool()));

    impl->model_runner = MakeSyncModelRunner(std::move(gpt_oss), HostSampleRow);
    impl->scheduler = std::make_unique<Scheduler>(std::move(scheduler));
    impl->stats.blocks_total = kv_blocks;

    // Size the activation scratch once, for the largest batch that will ever
    // run. The scratch only grows, so a later growth inside a step is fine --
    // but doing it up front avoids the first-step reallocation cost.
    INFERX_RETURN_IF_ERROR(impl->model_runner->ReserveActivations(
        config.scheduler.max_batch_tokens));

    if (config.capture_graphs) {
      const int64_t max_blocks =
          (config.scheduler.max_seq_len + config.block_size - 1) /
          config.block_size;
      for (int64_t seqs = config.scheduler.max_running; seqs >= 1; --seqs) {
        INFERX_RETURN_IF_ERROR(
            impl->model_runner->CaptureDecodeGraph(seqs, max_blocks));
      }
    }
  } else {
    Qwen2RunnerConfig runner_config;
    runner_config.model_dir = config.model_dir;
    runner_config.device_kind = config.device_kind;
    runner_config.devices = config.devices;
    runner_config.use_nccl = config.comm_backend == "nccl";
    runner_config.collective_timing_sample_every =
        config.collective_timing_sample_every;
    runner_config.fp8_weights = config.fp8_weights;
    runner_config.int4_weights = config.int4_weights;
    runner_config.fp8_kv_cache = config.fp8_kv_cache;
    runner_config.kv_blocks = config.kv_blocks;
    runner_config.kv_cache_memory_bytes = config.kv_cache_memory_bytes;
    runner_config.gpu_memory_utilization = config.gpu_memory_utilization;
    runner_config.block_size = config.block_size;
    runner_config.max_seq_len = config.scheduler.max_seq_len;
    runner_config.max_sampling_rows = config.scheduler.max_running;
    INFERX_ASSIGN_OR_RETURN(auto model, Qwen2Runner::Create(runner_config));

    INFERX_ASSIGN_OR_RETURN(
        Scheduler scheduler,
        Scheduler::Create(config.scheduler, model->kv_pool()));

    impl->model_runner = std::move(model);
    impl->metrics.ConfigureRanks(impl->model_runner->telemetry());
    impl->scheduler = std::make_unique<Scheduler>(std::move(scheduler));
    impl->stats.blocks_total = impl->model_runner->kv_pool()->num_blocks();

    // One graph per batch size. The block-table width is fixed for the whole
    // scheduler, so the only shape that varies between decode steps is the
    // number of sequences -- which makes the set of graphs small and knowable
    // up front rather than something to capture lazily and hope converges.
    if (config.capture_graphs) {
      // Order matters, and both of these are correctness requirements rather
      // than optimizations. First size the activation buffers for the largest
      // batch that will ever run, because they only grow and a later growth
      // would strand every captured pointer. Then run one real request through,
      // so that everything allocated on first use has been allocated.
      INFERX_RETURN_IF_ERROR(impl->model_runner->ReserveActivations(
          config.scheduler.max_batch_tokens));

      INFERX_RETURN_IF_ERROR(impl->Warmup());

      const int64_t max_blocks =
          (config.scheduler.max_seq_len + config.block_size - 1) /
          config.block_size;

      // Largest shape first, and this order is load-bearing. Capture records
      // device addresses, and preparing a *bigger* batch grows buffers that are
      // sized on demand -- FlashInfer's workspace among them. Capturing 1, 2,
      // 3, 4 in that order therefore strands graphs 1..3 the moment graph 4's
      // preparation reallocates underneath them, leaving only the last one
      // valid.
      //
      // Measured, because it does not announce itself: ascending capture left
      // "The capital of France is" continuing " Paris" and then repeating a
      // single token forever, while the same build with only one shape
      // captured produced " Paris. The capital of Germany is Berlin."
      // Descending, every shape is captured after the buffers have already
      // reached their maximum.
      for (int64_t seqs = config.scheduler.max_running; seqs >= 1; --seqs) {
        // Best-effort: a shape that will not capture still runs un-graphed, and
        // refusing to start the server over it would trade a working slow path
        // for no path at all.
        (void)impl->model_runner->CaptureDecodeGraph(seqs, max_blocks);
      }
    }
  }

  Impl* raw = impl.get();
  auto engine = std::unique_ptr<Engine>(new Engine(std::move(impl)));

  raw->loop = std::thread([raw] { raw->Run(); });

  return engine;
}

StatusOr<std::shared_ptr<Generation>> Engine::Submit(
    std::vector<int32_t> prompt, int32_t max_tokens,
    std::vector<std::string> stop, scheduler::SamplingParams sampling) {
  if (prompt.empty()) return InvalidArgumentError("prompt is empty");

  if (max_tokens <= 0) {
    return InvalidArgumentError("max_tokens must be positive, got ",
                                max_tokens);
  }

  if (impl_->stopping.load(std::memory_order_relaxed)) {
    return FailedPreconditionError("engine is shutting down");
  }

  const auto prompt_len = static_cast<int32_t>(prompt.size());

  if (prompt_len >= impl_->config.scheduler.max_seq_len) {
    return InvalidArgumentError("prompt is ", prompt_len,
                                " tokens, which does not leave room to "
                                "generate within max_seq_len of ",
                                impl_->config.scheduler.max_seq_len);
  }

  auto generation = std::make_shared<Generation>();
  generation->set_prompt_tokens(prompt_len);

  Impl::Pending pending;
  pending.prompt = std::move(prompt);
  pending.submitted = std::chrono::steady_clock::now();
  pending.params = std::move(sampling);
  pending.params.max_tokens = max_tokens;
  // User stop-token ids arrive in `stop_tokens`; EOS joins them unless the
  // request asked to generate through it.
  if (!pending.params.ignore_eos) {
    pending.params.stop_tokens.push_back(impl_->tokenizer->eos_id());
  }
  pending.stop = std::move(stop);
  pending.generation = generation;

  impl_->metrics.accepted->Increment();
  impl_->metrics.prompt_tokens->Increment(static_cast<uint64_t>(prompt_len));
  impl_->metrics.prompt_length->Observe(prompt_len);

  {
    std::lock_guard<std::mutex> lock(impl_->mutex);

    pending.id = impl_->next_id++;
    generation->id_ = pending.id;

    impl_->intake.push_back(std::move(pending));
  }

  impl_->wake.notify_one();

  return generation;
}

const tokenizer::Tokenizer& Engine::tokenizer() const {
  return *impl_->tokenizer;
}

const std::string& Engine::model_name() const { return impl_->model_name; }

model::Architecture Engine::architecture() const { return impl_->architecture; }

Engine::Stats Engine::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

std::string Engine::metrics() const {
  std::string output = impl_->metrics.registry.Render();
  if (impl_->model_runner != nullptr) {
    output += RenderRankMetrics(impl_->model_runner->telemetry());
  }
  return output;
}

}  // namespace inferx::engine
