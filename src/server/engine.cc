#include "inferx/server/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>
#include <unordered_map>
#include <utility>

#include "inferx/api/openai.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/gpt_oss.h"
#include "inferx/model/qwen2.h"

namespace inferx::server {
namespace {

using model::ForwardBatch;
using model::Qwen2Model;
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

}  // namespace

// --- Generation --------------------------------------------------------------

bool Generation::Next(Event* out) {
  std::unique_lock<std::mutex> lock(mutex_);

  ready_.wait(lock, [this] { return !events_.empty() || finished_; });

  if (events_.empty()) return false;

  *out = std::move(events_.front());
  events_.pop_front();

  return true;
}

void Generation::Cancel() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
  }

  // Wakes a consumer blocked in Next. The engine will retire the request on its
  // next step; this only stops the reader waiting for output that is no longer
  // coming.
  ready_.notify_all();
}

bool Generation::cancelled() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cancelled_;
}

void Generation::Emit(Event event) {
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (finished_) return;

    events_.push_back(std::move(event));
  }

  ready_.notify_one();
}

void Generation::Finish(FinishReason reason, int32_t generated) {
  {
    std::lock_guard<std::mutex> lock(mutex_);

    if (finished_) return;

    Event event;
    event.done = true;
    event.reason = reason;
    event.generated = generated;

    events_.push_back(std::move(event));
    finished_ = true;
  }

  ready_.notify_all();
}

// --- Engine ------------------------------------------------------------------

struct Engine::Impl {
  EngineConfig config;

  // Exactly one of these is populated, decided once in Engine::Create from the
  // checkpoint's declared architecture. They are not a variant because both
  // types are move-only and the Qwen2 path reads `model.X()` dozens of times;
  // `qwen2_model->X()` is the smaller diff against that hot path than threading
  // `std::get<Qwen2Model>(model)` through every call site.
  //
  // The gpt-oss path is the Phase 3 slow serve: GptOssModel has only Forward(),
  // no paged KV, no continuous batching. It is batch-1, full recompute per
  // token, and exists so the engine can serve the checkpoint end to end before
  // Phase 4 makes it fast.
  std::unique_ptr<Qwen2Model> qwen2_model;
  std::unique_ptr<model::GptOssModel> gpt_oss_model;

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

    // Held here rather than in `params` because the scheduler never sees text
    // (§3.1); matching these is this layer's job.
    std::vector<std::string> stop;

    std::shared_ptr<Generation> generation;
  };

  std::deque<Pending> intake;
  std::atomic<bool> stopping{false};
  RequestId next_id = 1;

  Engine::Stats stats;

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
    int32_t generated = 0;
  };

  std::unordered_map<RequestId, Active> active;

  Impl() = default;

  bool is_gpt_oss() const { return gpt_oss_model != nullptr; }

  // Routes one sampled token to its request: detokenize, apply stop sequences,
  // emit whatever is now safe to send.
  void Deliver(const TokenDelta& delta) {
    const auto it = active.find(delta.id);

    if (it == active.end()) return;

    Active& state = it->second;

    ++state.generated;
    ++stats.tokens_generated;

    state.text += state.decoder->Push(delta.token);

    // A stop sequence ends the generation even though the scheduler, which
    // never sees text, has no idea it happened. Cancelling here retires the
    // sequence on the next step and frees its blocks.
    if (!state.stop.empty()) {
      const size_t at = api::FindStopSequence(state.text, state.stop);

      if (at != std::string::npos) {
        state.text.resize(at);
        state.stop_hit = true;

        if (state.emitted < state.text.size()) {
          Generation::Event event;
          event.text = state.text.substr(state.emitted);
          event.generated = state.generated;

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

    if (safe > state.emitted) {
      Generation::Event event;
      event.text = state.text.substr(state.emitted, safe - state.emitted);
      event.generated = state.generated;

      state.generation->Emit(std::move(event));
      state.emitted = safe;
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
    active.erase(it);
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

      INFERX_RETURN_IF_ERROR(qwen2_model->StepAsync(batch));
      INFERX_RETURN_IF_ERROR(qwen2_model->AwaitStep(&sampled));
      INFERX_RETURN_IF_ERROR(scheduler->CommitStep(sampled, nullptr));

      (void)scheduler->TakeCompleted();
    }

    (void)scheduler->TakeCompleted();

    return OkStatus();
  }

  void Run() {
    if (is_gpt_oss()) {
      RunGptOss();
    } else {
      RunQwen2();
    }

    // Shutting down: nothing else will ever produce for these streams.
    for (auto& [id, state] : active) {
      state.generation->Finish(FinishReason::kCancelled, state.generated);
    }

    active.clear();
  }

  // The Qwen2 serving loop: continuous batching over a paged KV cache, driven
  // by the scheduler. This is M3-M10's path, unchanged by the gpt-oss work
  // except that the model and scheduler are reached through pointers (so the
  // gpt-oss path, which has neither, can share this struct).
  void RunQwen2() {
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
            if (wake.wait_until(lock, quiet_until) ==
                std::cv_status::timeout) {
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
            lock.lock();
            continue;
          }

          Active state;
          state.generation = pending.generation;
          state.decoder = std::make_unique<tokenizer::IncrementalDecoder>(
              tokenizer.get(), /*skip_special=*/true);
          state.stop = std::move(pending.stop);

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

      if (!batch.token_ids.empty()) {
        Status stepped = qwen2_model->StepAsync(batch);

        if (stepped.ok()) stepped = qwen2_model->AwaitStep(&sampled);

        // A device error is not recoverable per-request: the KV cache state is
        // now unknown, so every in-flight sequence is suspect. Failing them all
        // and stopping is honest; continuing would serve corrupted output.
        if (!stepped.ok()) {
          FailAll(stepped);
          stopping.store(true, std::memory_order_relaxed);
          break;
        }

        if (const Status committed = scheduler->CommitStep(sampled, &deltas);
            !committed.ok()) {
          FailAll(committed);
          stopping.store(true, std::memory_order_relaxed);
          break;
        }

        for (const TokenDelta& delta : deltas) Deliver(delta);

        std::lock_guard<std::mutex> lock(mutex);
        ++stats.steps;
        stats.last_step_ms = qwen2_model->last_step_device_ms();
      }

      for (const scheduler::Completion& completion : scheduler->TakeCompleted()) {
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

  // The gpt-oss serving loop: scheduler-driven continuous batching, but with a
  // synchronous Step + host argmax where the Qwen2 path uses StepAsync +
  // device sampling. The shape of the loop (intake -> PrepareStep -> Step ->
  // CommitStep -> Deliver) is deliberately the same as RunQwen2's, so the two
  // read as one design with the sampling/async parts swapped.
  //
  // The win over the old batch-1 RunGptOss is real: the MXFP4 expert upload
  // (~10 GB over PCIe per layer) is per-forward, not per-token, so a batch of N
  // sequences pays it once rather than N times. Each token in a 4-sequence
  // batch costs ~1/4 the bandwidth-floor latency of the single-sequence path,
  // even before Phase 4's fused GEMM makes the upload itself resident.
  //
  // No CUDA graphs and no device sampling yet: the per-layer
  // cudaDeviceSynchronize inside GptOssModel's MoE path (R-E's bias_keep free)
  // forbids capture, and device sampling is moot while the logits already come
  // back to the host for argmax. Both stack on later, after Phase 4.
  void RunGptOss() {
    std::vector<float> logits;
    std::vector<int32_t> sampled;
    std::vector<TokenDelta> deltas;

    while (!stopping.load(std::memory_order_relaxed)) {
      {
        std::unique_lock<std::mutex> lock(mutex);

        // Same coalesce as the Qwen2 path: when the scheduler is empty, a
        // newly-arrived burst is given a short quiet window to accumulate
        // before the first batch is formed. Without it, batch membership
        // depends on thread timing, which showed up as different greedy
        // continuations for identical bursts.
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

          const Status added = scheduler->AddRequest(
              pending.id, std::move(pending.prompt), pending.params);

          if (!added.ok()) {
            lock.unlock();
            pending.generation->Finish(FinishReason::kOutOfMemory, 0);
            lock.lock();
            continue;
          }

          Active state;
          state.generation = pending.generation;
          state.decoder = std::make_unique<tokenizer::IncrementalDecoder>(
              tokenizer.get(), /*skip_special=*/true);
          state.stop = std::move(pending.stop);

          active.emplace(pending.id, std::move(state));
        }

        if (!scheduler->HasWork()) {
          wake.wait_for(lock, kIdleWait);
          continue;
        }
      }

      // Disconnects, noticed once per step.
      for (const auto& [id, state] : active) {
        if (state.generation->cancelled()) scheduler->Cancel(id);
      }

      ForwardBatch batch;
      if (const Status prepared = scheduler->PrepareStep(&batch);
          !prepared.ok()) {
        FailAll(prepared);
        continue;
      }

      if (batch.token_ids.empty()) {
        for (const scheduler::Completion& completion :
             scheduler->TakeCompleted()) {
          Retire(completion.id, completion.reason);
        }
        continue;
      }

      // Synchronous step: logits come back to the host, argmax runs here, and
      // the sampled tokens go to the scheduler. The Qwen2 path's StepAsync /
      // AwaitStep overlap does not apply -- the gpt-oss step's per-layer
      // cudaDeviceSynchronize already serializes it.
      const Status stepped = gpt_oss_model->Step(batch, &logits);

      if (!stepped.ok()) {
        FailAll(stepped);
        stopping.store(true, std::memory_order_relaxed);
        break;
      }

      // One argmax per requested logits row. The batch carries logits_indices
      // naming which rows to sample, and Step returns them in the same order.
      const int64_t vocab = gpt_oss_model->config().vocab_size;
      sampled.clear();
      sampled.reserve(batch.logits_indices.size());
      for (size_t i = 0; i < batch.logits_indices.size(); ++i) {
        const float* row = logits.data() + i * static_cast<size_t>(vocab);
        const int32_t tok = static_cast<int32_t>(
            std::max_element(row, row + vocab) - row);
        sampled.push_back(tok);
      }

      if (const Status committed = scheduler->CommitStep(sampled, &deltas);
          !committed.ok()) {
        FailAll(committed);
        stopping.store(true, std::memory_order_relaxed);
        break;
      }

      for (const TokenDelta& delta : deltas) Deliver(delta);

      {
        std::lock_guard<std::mutex> lock(mutex);
        ++stats.steps;
        stats.running = scheduler->num_running();
        stats.waiting = scheduler->num_waiting();
        stats.blocks_in_use = scheduler->blocks_in_use();
        stats.preemptions = scheduler->preemptions();
        stats.cached_blocks = scheduler->cached_blocks();
        stats.prefix_hit_tokens = scheduler->prefix_hit_tokens();
        stats.prefix_miss_tokens = scheduler->prefix_miss_tokens();
        stats.evicted_blocks = scheduler->evicted_blocks();
      }

      for (const scheduler::Completion& completion :
           scheduler->TakeCompleted()) {
        Retire(completion.id, completion.reason);
      }
    }
  }

  void FailAll(const Status& why) {
    (void)why;

    for (auto& [id, state] : active) {
      state.generation->Finish(FinishReason::kOutOfMemory, state.generated);
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

  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<tokenizer::Tokenizer> tok,
                          tokenizer::Tokenizer::LoadFromDirectory(
                              config.model_dir));

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

  impl->model_name = config.served_model_name.empty()
                         ? std::filesystem::path(config.model_dir)
                               .filename()
                               .string()
                         : config.served_model_name;

  if (model_config.architecture == model::Architecture::kGptOss) {
    // Phase 3 slow serve. FP8 weights / KV cache, CUDA graphs and the scheduler
    // are Qwen2-path options that GptOssModel does not expose; silently
    // ignoring them keeps `inferx-serve --model <gpt-oss> --fp8` from crashing
    // on a method that does not exist, and the launch log below names the path
    // so the slowness is not a mystery.
    if (config.fp8_weights || config.fp8_kv_cache || config.capture_graphs) {
      // Not an error: the flags are accepted, the gpt-oss path simply does not
      // use them. Logging is the caller's business; the return path is what
      // matters here.
    }

    INFERX_ASSIGN_OR_RETURN(model::GptOssModel gpt_oss,
                            model::GptOssModel::Load(config.model_dir));

    // Attach a paged KV cache so the scheduler can batch multiple sequences.
    // gpt-oss has 24 layers × 8 kv_heads × 64 head_dim × 2 (bf16) = 2048
    // bytes/token/layer; the pool sizing is a VRAM/latency trade the caller
    // makes via --kv-blocks, same as Qwen2.
    INFERX_RETURN_IF_ERROR(gpt_oss.AttachKvCache(config.kv_blocks, config.block_size));

    INFERX_ASSIGN_OR_RETURN(
        Scheduler scheduler,
        Scheduler::Create(config.scheduler, gpt_oss.kv_pool()));

    impl->gpt_oss_model =
        std::make_unique<model::GptOssModel>(std::move(gpt_oss));
    impl->scheduler = std::make_unique<Scheduler>(std::move(scheduler));
    impl->stats.blocks_total = config.kv_blocks;

    // Size the activation scratch once, for the largest batch that will ever
    // run. The scratch only grows, so a later growth inside a step is fine --
    // but doing it up front avoids the first-step reallocation cost.
    INFERX_RETURN_IF_ERROR(
        impl->gpt_oss_model->ReserveActivations(config.scheduler.max_batch_tokens));
  } else {
    INFERX_ASSIGN_OR_RETURN(Qwen2Model model,
                            Qwen2Model::LoadFromDirectory(config.model_dir));

    if (config.fp8_weights) {
      INFERX_RETURN_IF_ERROR(model.QuantizeWeightsToF8());
    }

    if (config.fp8_kv_cache) {
      INFERX_RETURN_IF_ERROR(model.EnableFp8KvCache());
    }

    INFERX_RETURN_IF_ERROR(
        model.AttachKvCache(config.kv_blocks, config.block_size));

    // Device sampling is what makes StepAsync possible: without it the next
    // step's token ids would have to come back through the host first (§5.2).
    INFERX_RETURN_IF_ERROR(
        model.EnableDeviceSampling(config.scheduler.max_running));

    INFERX_ASSIGN_OR_RETURN(
        Scheduler scheduler,
        Scheduler::Create(config.scheduler, model.kv_pool()));

    impl->qwen2_model = std::make_unique<Qwen2Model>(std::move(model));
    impl->scheduler = std::make_unique<Scheduler>(std::move(scheduler));
    impl->stats.blocks_total = config.kv_blocks;

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
      INFERX_RETURN_IF_ERROR(impl->qwen2_model->ReserveActivations(
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
        (void)impl->qwen2_model->CaptureDecodeGraph(seqs, max_blocks);
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
  pending.params = std::move(sampling);
  pending.params.max_tokens = max_tokens;
  pending.params.stop_tokens = {impl_->tokenizer->eos_id()};
  pending.stop = std::move(stop);
  pending.generation = generation;

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

Engine::Stats Engine::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->stats;
}

}  // namespace inferx::server
