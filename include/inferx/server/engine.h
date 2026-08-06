#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/scheduler/scheduler.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx::server {

struct EngineConfig {
  /// Checkpoint directory: safetensors, config.json, tokenizer.json.
  std::string model_dir;

  scheduler::SchedulerConfig scheduler;

  /// Blocks in the KV pool. Together with `block_size` and the model's layer
  /// count this is the whole KV budget, and it is a deployment decision -- how
  /// much VRAM is left after the weights -- not a property of the model.
  int64_t kv_blocks = 4096;
  int64_t block_size = 16;

  /// Tensor-parallel deployment. The first NCCL runtime supports two distinct
  /// GPUs in one process; TP=1 retains the direct single-rank path.
  int tensor_parallel_size = 1;
  std::vector<int> devices{0};
  std::string comm_backend = "single";

  /// Diagnostic CUDA-event timing. Zero disables it; N samples every Nth
  /// collective. Sampling is automatically skipped during graph capture.
  uint64_t collective_timing_sample_every = 0;

  /// Quantize weights to FP8 e4m3 at load. Roughly halves weight bandwidth and
  /// so nearly halves decode latency, at the cost of per-tensor quantization
  /// error. Off by default: it changes the model's output, and that should be
  /// an explicit choice.
  bool fp8_weights = false;

  /// Quantize projection weights to symmetric per-group int4 and run them
  /// through the fused W4A16 kernel. Mutually exclusive with `fp8_weights`;
  /// embeddings, norms, and the tied LM head remain bf16.
  bool int4_weights = false;

  /// Store the KV cache as FP8 e4m3 instead of bf16. Halves KV memory (twice
  /// the concurrency at the same VRAM) at the cost of fp8 quantization error on
  /// K/V. Per-layer dequant scales are frozen from warmup; prefill folds the K
  /// scale into the query and decode into sm_scale. Off by default for the same
  /// reason as `fp8_weights`.
  bool fp8_kv_cache = false;

  /// Capture a CUDA graph per batch size at startup.
  ///
  /// On by default again now that R9 is fixed. The capture probe used to write
  /// its keys and values into physical block 0 -- which the free list hands out
  /// first -- so capturing a graph corrupted the first live sequence's history
  /// in every layer. It now borrows a block from the pool and returns it.
  ///
  /// Costs a few hundred milliseconds of load time and removes per-launch
  /// overhead from every decode step afterwards.
  bool capture_graphs = true;

  /// Reported as the model name in responses. Defaults to the directory's
  /// basename.
  std::string served_model_name;
};

/// \brief One in-flight generation, as the request thread sees it.
///
/// The engine runs a single step loop on its own thread; HTTP handlers run on
/// theirs. This is the queue between them. It is reference-counted because the
/// two ends genuinely have independent lifetimes: a client can disconnect while
/// a step is in flight, and the engine can finish a sequence while the handler
/// is still draining what it already produced.
class Generation {
 public:
  struct Event {
    /// Text produced since the previous event. Often empty -- a token need not
    /// complete a character, and detokenization holds back partial UTF-8.
    std::string text;

    /// Set on the last event, which carries no text.
    bool done = false;

    scheduler::FinishReason reason = scheduler::FinishReason::kNotFinished;

    /// Tokens generated so far, for usage accounting.
    int32_t generated = 0;
  };

  /// \brief Blocks until an event is available, returning false at end of
  ///        stream.
  bool Next(Event* out);

  /// \brief Abandons the generation.
  ///
  /// Safe at any point, including after completion: a disconnect races with
  /// everything (§4, step 10). The engine notices on its next step.
  void Cancel();

  /// \brief Whether the client has gone away.
  bool cancelled() const;

  int32_t prompt_tokens() const { return prompt_tokens_; }

  /// \name Producer side
  ///
  /// Called by the engine loop only. Public because the loop lives in the
  /// engine's implementation struct rather than in `Engine` itself, and
  /// contorting the friendship to hide three methods would buy nothing -- the
  /// header is the contract, and the contract is that clients call `Next` and
  /// `Cancel`.
  /// @{

  /// \brief Queues an event for the consumer.
  void Emit(Event event);

  /// \brief Queues the terminal event and closes the stream.
  ///
  /// Idempotent: a generation cancelled and then completed in the same step
  /// must not deliver two terminal events.
  void Finish(scheduler::FinishReason reason, int32_t generated);

  void set_prompt_tokens(int32_t n) { prompt_tokens_ = n; }

  /// @}

 private:
  friend class Engine;

  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Event> events_;
  bool finished_ = false;
  bool cancelled_ = false;
  int32_t prompt_tokens_ = 0;
  scheduler::RequestId id_ = 0;
};

/// \brief Model, scheduler and tokenizer behind one step loop.
///
/// §3.1's split is preserved rather than blurred: the scheduler still holds all
/// the policy and touches no CUDA, the model still owns all the CUDA and holds
/// no policy, and this class is the seam -- it owns the *thread*, and the order
/// in which the two are called. Everything it adds on top is bookkeeping the
/// scheduler must not have because it would need text to do it: detokenization,
/// and stop strings.
///
/// The loop is §5.2 at depth 1: `StepAsync` issues the step and `AwaitStep`
/// collects it, with the host free in between. It is not yet overlapped -- the
/// double-buffering that would let step N+1 be prepared while step N runs is
/// deferred, and measured at about 5.5% -- so for now the two calls bracket the
/// same step.
class Engine {
 public:
  static StatusOr<std::unique_ptr<Engine>> Create(const EngineConfig& config);

  /// Stops the loop and fails every outstanding generation.
  ~Engine();

  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  /// \brief Submits a prompt and returns the stream it will arrive on.
  ///
  /// \param prompt      Token ids. Must be non-empty and shorter than
  ///                    `max_seq_len`.
  /// \param max_tokens  Generation cap.
  /// \param stop        Text sequences that end the generation.
  /// \param sampling  Temperature, nucleus threshold and seed. Defaults are
  ///                  greedy, so callers predating sampling are unaffected.
  StatusOr<std::shared_ptr<Generation>> Submit(
      std::vector<int32_t> prompt, int32_t max_tokens,
      std::vector<std::string> stop, scheduler::SamplingParams sampling = {});

  /// \brief The tokenizer, for callers that need to encode prompts.
  const tokenizer::Tokenizer& tokenizer() const;

  /// \brief The name to report in responses.
  const std::string& model_name() const;

  struct Stats {
    int64_t running = 0;
    int64_t waiting = 0;
    int64_t blocks_in_use = 0;
    int64_t blocks_total = 0;
    int64_t steps = 0;
    int64_t tokens_generated = 0;
    double last_step_ms = 0.0;
    /// Cumulative §8.2 preemptions. Climbing means the KV pool is undersized
    /// for `max_running`, not that anything is broken -- but every one of them
    /// is a prefill paid for twice.
    int64_t preemptions = 0;
    /// Blocks the §6.3 prefix cache is holding. Reclaimable, so this being
    /// large is the cache working rather than memory pressure.
    int64_t cached_blocks = 0;
    /// Prompt tokens served from the cache, and tokens that had to be
    /// computed. Their ratio is the hit rate, and it is the number that says
    /// whether prefix caching is earning its complexity on this workload.
    int64_t prefix_hit_tokens = 0;
    int64_t prefix_miss_tokens = 0;
    /// Blocks reclaimed from the cache to satisfy an allocation. Read against
    /// the hit count: eviction rising while hits do not is the pool being too
    /// small for the working set, not the cache being the wrong idea.
    int64_t evicted_blocks = 0;
  };

  Stats stats() const;

  /// Prometheus text exposition for process-lifetime serving metrics and the
  /// current scheduler/cache snapshot. Safe to call from HTTP scrape threads.
  std::string metrics() const;

 private:
  struct Impl;

  explicit Engine(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::server
