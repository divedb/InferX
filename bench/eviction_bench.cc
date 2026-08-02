/// What the prefix cache costs when the pool is too small to hold it (§6.3).
///
/// `prefix_cache_bench` measures the cache with room to breathe and finds it
/// worth 5.9x on warm time-to-first-token. That is the easy half of the
/// question. The hard half is what happens when cached blocks and running
/// sequences want the same memory, because then every hit is memory taken from
/// somebody who needed it, and the cache stops being free.
///
/// Three ways this could go badly, and the sweep below is built to tell them
/// apart:
///
///   1. **Churn.** Blocks are cached, evicted before anyone reuses them, and
///      recomputed. The hit rate collapses and the tree walk is pure overhead.
///      Visible as evictions climbing while hits do not.
///   2. **Displacement.** The cache holds memory a running sequence needs, so
///      the engine preempts where it otherwise would not have. Visible as
///      preemptions being *higher* with the cache on.
///   3. **A soft landing**, which is what the design intends: eviction gives
///      memory back on demand, so the cache degrades to roughly the uncached
///      case rather than falling below it.
///
/// The workload is the one with the most to gain -- a 1024-token preamble every
/// request shares -- run as a burst of concurrent requests so the pool is under
/// real contention rather than being handed one sequence at a time. The pool is
/// swept from comfortably larger than the working set down to smaller than it.
/// Everything else is held fixed, and `enable_prefix_cache` is the only thing
/// that differs between the two rows at each size.
///
/// Wall clock for the whole burst is the headline, because under pressure the
/// question is throughput rather than any one request's latency: TTFT here
/// includes queueing behind other requests in the burst and is reported only as
/// a secondary signal.
///
/// Graphs are off, and the KV pool is re-attached per sweep point -- safe only
/// because nothing here captures one. A graph holds device pointers into the
/// pool it was captured against, and swapping the pool under it is R9's exact
/// shape.
///
/// For numbers worth comparing across runs, lock the clocks first:
///     sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 2400

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/model/qwen2.h"
#include "inferx/scheduler/scheduler.h"

namespace inferx::bench {
namespace {

using model::ForwardBatch;
using scheduler::RequestId;
using scheduler::SamplingParams;
using scheduler::Scheduler;
using scheduler::SchedulerConfig;
using scheduler::TokenDelta;

constexpr int64_t kBlockSize = 16;

/// The shared preamble, and the part that differs per request.
constexpr int64_t kSystemPrompt = 1024;
constexpr int64_t kQuestion = 48;
constexpr int32_t kGenerate = 64;

/// Requests in the burst, and how many may be resident at once.
constexpr int kRequests = 12;
constexpr int64_t kMaxRunning = 4;

/// Pool sizes, in blocks of 16.
///
/// The two configurations do not have the same working set, and that turns out
/// to be most of the story. Uncached, each resident sequence needs its whole
/// 1136 tokens to itself -- 71 blocks, times 4 resident is **284**. Cached, all
/// four point at the *same* 64 blocks of preamble and own only their own tails,
/// so the four of them fit in **about 96**.
///
/// The sweep therefore has to run much lower than the uncached working set to
/// put the cache itself under pressure, which is the thing being measured. 320
/// is roomy for both; 128 is below what the uncached case needs to run four at
/// once; and at 80 even the cached case cannot hold the preamble and four tails
/// together, which is where eviction has to start costing something.
constexpr int64_t kPoolBlocks[] = {320, 192, 128, 96, 88, 80};

/// For the no-sharing sweep, where both configurations have the same 284-block
/// working set: from roomy down to below it.
constexpr int64_t kDistinctPoolBlocks[] = {512, 320, 256, 192, 128};

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

std::vector<int32_t> Tokens(int64_t n, uint64_t seed) {
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(n));

  uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;

  for (int64_t i = 0; i < n; ++i) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29;
    out.push_back(static_cast<int32_t>(1000 + (x % 100000)));
  }

  return out;
}

struct Result {
  double wall_ms = 0.0;
  double mean_ttft_ms = 0.0;
  int64_t hit_tokens = 0;
  int64_t miss_tokens = 0;
  int64_t preemptions = 0;
  int64_t evicted = 0;
  int completed = 0;
  bool ok = false;

  double hit_rate() const {
    const int64_t total = hit_tokens + miss_tokens;
    return total > 0 ? 100.0 * static_cast<double>(hit_tokens) /
                           static_cast<double>(total)
                     : 0.0;
  }
};

/// Submits the whole burst at once and runs it to completion.
///
/// \param shared  Whether the requests share a preamble. False gives every
///                request its own unrelated prompt of the same length, which is
///                the case where the cache can only cost and never earn.
Result RunBurst(model::Qwen2Model* model, bool cache, bool shared) {
  Result result;

  SchedulerConfig config;
  config.max_running = kMaxRunning;
  config.max_batch_tokens = 2048;
  config.max_seq_len = 2048;
  config.enable_prefix_cache = cache;

  auto created = Scheduler::Create(config, model->kv_pool());
  if (!created.ok()) {
    std::fprintf(stderr, "scheduler: %s\n", created.status().ToString().c_str());
    return result;
  }
  Scheduler sched = *std::move(created);

  const std::vector<int32_t> preamble = Tokens(kSystemPrompt, 1);

  SamplingParams params;
  params.max_tokens = kGenerate;

  for (int i = 0; i < kRequests; ++i) {
    std::vector<int32_t> prompt =
        shared ? preamble
               : Tokens(kSystemPrompt, static_cast<uint64_t>(7000 + i));

    const std::vector<int32_t> question =
        Tokens(kQuestion, static_cast<uint64_t>(100 + i));
    prompt.insert(prompt.end(), question.begin(), question.end());

    if (const Status s = sched.AddRequest(static_cast<RequestId>(i + 1),
                                          std::move(prompt), params);
        !s.ok()) {
      std::fprintf(stderr, "add: %s\n", s.ToString().c_str());
      return result;
    }
  }

  std::vector<int32_t> sampled;
  std::vector<TokenDelta> deltas;
  absl::flat_hash_map<RequestId, double> first_token_at;

  const auto t0 = std::chrono::steady_clock::now();

  int steps = 0;

  while (sched.HasWork() && steps < 20000) {
    ++steps;

    ForwardBatch batch;
    if (const Status s = sched.PrepareStep(&batch); !s.ok()) {
      std::fprintf(stderr, "prepare: %s\n", s.ToString().c_str());
      return result;
    }

    if (batch.num_tokens() == 0) continue;

    Status s = model->StepAsync(batch);
    if (s.ok()) s = model->AwaitStep(&sampled);

    if (!s.ok()) {
      std::fprintf(stderr, "step: %s\n", s.ToString().c_str());
      return result;
    }

    if (const Status c = sched.CommitStep(sampled, &deltas); !c.ok()) {
      std::fprintf(stderr, "commit: %s\n", c.ToString().c_str());
      return result;
    }

    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double, std::milli>(now - t0).count();

    for (const TokenDelta& delta : deltas) {
      // Time to *first* token. A preempted sequence comes back and produces its
      // first token again; the first one it ever produced is the one a client
      // saw, so the earliest wins.
      if (first_token_at.find(delta.id) == first_token_at.end()) {
        first_token_at[delta.id] = elapsed;
      }
    }

    result.completed += static_cast<int>(sched.TakeCompleted().size());
  }

  const auto t1 = std::chrono::steady_clock::now();

  result.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  result.completed += static_cast<int>(sched.TakeCompleted().size());

  double total_ttft = 0.0;
  for (const auto& [id, at] : first_token_at) total_ttft += at;

  result.mean_ttft_ms =
      first_token_at.empty()
          ? 0.0
          : total_ttft / static_cast<double>(first_token_at.size());

  result.hit_tokens = sched.prefix_hit_tokens();
  result.miss_tokens = sched.prefix_miss_tokens();
  result.preemptions = sched.preemptions();
  result.evicted = sched.evicted_blocks();
  result.ok = result.completed == kRequests;

  return result;
}

int Main(int argc, char** argv) {
  if (argc > 1) {
    std::fprintf(stderr, "usage: %s\n", argv[0]);
    return 2;
  }

  if (!CudaAvailable()) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }

  auto loaded = model::Qwen2Model::LoadFromDirectory(CheckpointDir());
  if (!loaded.ok()) {
    std::fprintf(stderr, "cannot load model: %s\n",
                 loaded.status().ToString().c_str());
    return 1;
  }

  model::Qwen2Model model = *std::move(loaded);

  if (const Status s = model.ReserveActivations(2048); !s.ok()) {
    std::fprintf(stderr, "reserve activations: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.EnableDeviceSampling(kMaxRunning); !s.ok()) {
    std::fprintf(stderr, "device sampling: %s\n", s.ToString().c_str());
    return 1;
  }

  std::printf("%s\n", model.config().ToString().c_str());
  std::printf("weights: bf16, %.2f GB\n\n", model.WeightBytes() / 1e9);
  std::printf(
      "%d requests sharing a %ld-token preamble, %ld resident at once,\n"
      "%d tokens generated each.\n\n"
      "Working set uncached: 4 x 71 = 284 blocks. Cached: ~96, because the\n"
      "four resident sequences share one copy of the preamble.\n\n",
      kRequests, static_cast<long>(kSystemPrompt), static_cast<long>(kMaxRunning),
      kGenerate);

  // One throwaway burst so that allocator growth and cuBLASLt's per-shape
  // heuristic are paid for before the first measured row.
  if (model.AttachKvCache(320, kBlockSize).ok()) {
    (void)RunBurst(&model, /*cache=*/false, /*shared=*/true);
  }

  int failures = 0;

  const auto sweep = [&](const char* title, bool shared,
                         absl::Span<const int64_t> sizes) {
    std::printf("%s\n", title);
    std::printf("%7s %6s %10s %11s %8s %8s %9s\n", "blocks", "cache",
                "wall_ms", "mean_ttft", "hit_%", "preempt", "evicted");
    std::printf("%s\n", std::string(70, '-').c_str());

    for (const int64_t blocks : sizes) {
      Result off;
      Result on;

      for (int pass = 0; pass < 2; ++pass) {
        const bool cache = pass == 1;

        // A fresh pool per row, so no row inherits another's fragmentation or
        // leftover cache contents.
        if (const Status s = model.AttachKvCache(blocks, kBlockSize); !s.ok()) {
          std::fprintf(stderr, "attach %ld: %s\n", static_cast<long>(blocks),
                       s.ToString().c_str());
          ++failures;
          continue;
        }

        (cache ? on : off) = RunBurst(&model, cache, shared);
      }

      if (!off.ok || !on.ok) {
        std::fprintf(stderr, "%ld blocks: did not complete\n",
                     static_cast<long>(blocks));
        ++failures;
        continue;
      }

      std::printf("%7ld %6s %10.0f %11.0f %8.0f %8ld %9ld\n",
                  static_cast<long>(blocks), "off", off.wall_ms,
                  off.mean_ttft_ms, off.hit_rate(),
                  static_cast<long>(off.preemptions),
                  static_cast<long>(off.evicted));

      std::printf("%7s %6s %10.0f %11.0f %8.0f %8ld %9ld   %.2fx\n", "", "on",
                  on.wall_ms, on.mean_ttft_ms, on.hit_rate(),
                  static_cast<long>(on.preemptions),
                  static_cast<long>(on.evicted),
                  on.wall_ms > 0.0 ? off.wall_ms / on.wall_ms : 0.0);
    }

    std::printf("\n");
  };

  sweep("shared 1024-token preamble", /*shared=*/true, kPoolBlocks);

  // Nothing to share, so every cached block is dead weight the moment it is
  // stored. This is the row that says whether the cache can make things worse.
  sweep("distinct prompts of the same length", /*shared=*/false,
        kDistinctPoolBlocks);

  std::printf(
      "The last column is the speedup on total wall time from turning the\n"
      "cache on. hit_%% is prompt tokens served from the tree; preempt and\n"
      "evicted are cumulative over the burst.\n");

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
