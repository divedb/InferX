/// What the radix prefix cache (§6.3) is worth, on the workloads it exists for.
///
/// §6.3 names them: a system prompt in front of every request, few-shot
/// examples, multi-turn chat, an agent rebuilding its scratchpad. All of them
/// send a prompt whose head someone has already computed, and all of them pay
/// for it again on every request without a cache. Prefill is the expensive half
/// of serving -- `prefill_bench` prices 2k tokens at 176 ms against 11.6 ms for
/// a decode step -- so what is being measured here is time-to-first-token, and
/// specifically how much of it goes away when the head is already resident.
///
/// Three workloads, each run twice, with `enable_prefix_cache` the only thing
/// that differs between the runs:
///
///   1. **Shared system prompt.** A long preamble common to every request, each
///      with its own short question after it. The canonical case, and the one
///      with the highest possible hit rate.
///   2. **Multi-turn chat.** Each turn resends the whole conversation and adds
///      to it, so turn k's prompt contains every token of turns 1..k-1 plus
///      what the model said back. The hit grows with the conversation, which is
///      the case where recomputing is most obviously absurd.
///   3. **Distinct prompts.** Nothing shared. This one is here to show the
///      cache costs nothing when it cannot help -- a lookup that misses is a
///      tree descent against a prompt's first tokens, and if that showed up
///      against a 176 ms prefill it would be the finding.
///
/// TTFT is wall clock, because that is what the client waits. The requests run
/// one at a time: this measures the prefill path, and interleaving decodes from
/// other sequences would put someone else's step time in the denominator.
/// `tbt_bench` is the one that measures a mixed load.
///
/// Graphs are off, as in `tbt_bench` -- prefill shapes vary per request here by
/// construction, so a graph would never match.
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

/// Workload 1: a preamble every request shares, and the question after it.
constexpr int64_t kSystemPrompt = 1024;
constexpr int64_t kQuestion = 48;
constexpr int kRequests = 8;

/// Workload 2: how a conversation grows.
constexpr int64_t kChatOpening = 256;
constexpr int64_t kChatUserTurn = 48;
constexpr int32_t kChatReply = 48;
constexpr int kChatTurns = 6;

/// Workload 3: unrelated prompts of a comparable size.
constexpr int64_t kDistinctPrompt = 1024;

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

/// Tokens that look like text to the model without being any particular text.
/// Their content does not matter -- what is being timed is how many of them the
/// engine has to push through the model -- but they must vary, since a run of
/// one repeated id is not a representative attention pattern.
std::vector<int32_t> Tokens(int64_t n, uint64_t seed) {
  std::vector<int32_t> out;
  out.reserve(static_cast<size_t>(n));

  uint64_t x = seed * 0x9e3779b97f4a7c15ULL + 1;

  for (int64_t i = 0; i < n; ++i) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 29;
    // Well inside Qwen2.5's 151936-entry vocabulary, and away from the special
    // ids at the top of it.
    out.push_back(static_cast<int32_t>(1000 + (x % 100000)));
  }

  return out;
}

struct Served {
  double ttft_ms = 0.0;
  /// Tokens the model actually forwarded before the first one came back. This
  /// is the quantity the cache is trying to shrink; TTFT is what that buys.
  int64_t forwarded = 0;
  /// Prompt tokens this request took from the cache.
  int64_t hit = 0;
  std::vector<int32_t> output;
  bool ok = false;
};

/// Runs one request to completion, timing its first token.
///
/// Nothing else is in flight, so every step between submission and the first
/// token belongs to this request's prefill.
Served Serve(model::Qwen2Model* model, Scheduler* sched, RequestId id,
             const std::vector<int32_t>& prompt, int32_t max_tokens) {
  Served out;

  SamplingParams params;
  params.max_tokens = max_tokens;

  const int64_t hits_before = sched->prefix_hit_tokens();

  if (const Status s = sched->AddRequest(id, prompt, params); !s.ok()) {
    std::fprintf(stderr, "add: %s\n", s.ToString().c_str());
    return out;
  }

  std::vector<int32_t> sampled;
  std::vector<TokenDelta> deltas;

  const auto t0 = std::chrono::steady_clock::now();
  bool seen_first = false;

  for (int step = 0; step < 4000 && sched->HasWork(); ++step) {
    ForwardBatch batch;
    if (const Status s = sched->PrepareStep(&batch); !s.ok()) {
      std::fprintf(stderr, "prepare: %s\n", s.ToString().c_str());
      return out;
    }

    if (batch.num_tokens() == 0) continue;

    if (!seen_first) out.forwarded += batch.num_tokens();

    Status s = model->StepAsync(batch);
    if (s.ok()) s = model->AwaitStep(&sampled);

    if (!s.ok()) {
      std::fprintf(stderr, "step: %s\n", s.ToString().c_str());
      return out;
    }

    if (const Status c = sched->CommitStep(sampled, &deltas); !c.ok()) {
      std::fprintf(stderr, "commit: %s\n", c.ToString().c_str());
      return out;
    }

    for (const TokenDelta& delta : deltas) {
      if (delta.id != id) continue;

      if (!seen_first) {
        const auto t1 = std::chrono::steady_clock::now();
        out.ttft_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        seen_first = true;
      }

      out.output.push_back(delta.token);
    }
  }

  (void)sched->TakeCompleted();

  out.hit = sched->prefix_hit_tokens() - hits_before;
  out.ok = seen_first;

  return out;
}

/// One workload's worth of numbers, for one setting of the cache.
struct Result {
  double cold_ttft_ms = 0.0;
  /// Mean over every request after the first, which is where a cache can help.
  double warm_ttft_ms = 0.0;
  int64_t prompt_tokens = 0;
  int64_t forwarded = 0;
  int64_t hit = 0;
  bool ok = false;
};

Result Summarize(const std::vector<Served>& served) {
  Result r;
  if (served.empty()) return r;

  r.cold_ttft_ms = served.front().ttft_ms;

  double warm_total = 0.0;
  int warm_count = 0;

  for (size_t i = 0; i < served.size(); ++i) {
    if (!served[i].ok) return r;

    r.forwarded += served[i].forwarded;
    r.hit += served[i].hit;

    if (i > 0) {
      warm_total += served[i].ttft_ms;
      ++warm_count;
    }
  }

  r.warm_ttft_ms = warm_count > 0 ? warm_total / warm_count : r.cold_ttft_ms;
  r.ok = true;

  return r;
}

StatusOr<Scheduler> MakeScheduler(model::Qwen2Model* model, bool cache) {
  SchedulerConfig config;
  config.max_running = 4;
  config.max_batch_tokens = 2048;
  config.max_seq_len = 2048;
  config.enable_prefix_cache = cache;

  return Scheduler::Create(config, model->kv_pool());
}

Result SharedSystemPrompt(model::Qwen2Model* model, bool cache) {
  auto created = MakeScheduler(model, cache);
  if (!created.ok()) return {};
  Scheduler sched = *std::move(created);

  const std::vector<int32_t> preamble = Tokens(kSystemPrompt, 1);

  std::vector<Served> served;

  for (int i = 0; i < kRequests; ++i) {
    std::vector<int32_t> prompt = preamble;
    const std::vector<int32_t> question =
        Tokens(kQuestion, static_cast<uint64_t>(100 + i));
    prompt.insert(prompt.end(), question.begin(), question.end());

    served.push_back(Serve(model, &sched, static_cast<RequestId>(i + 1), prompt,
                           /*max_tokens=*/8));
  }

  Result r = Summarize(served);
  r.prompt_tokens = kSystemPrompt + kQuestion;
  return r;
}

Result MultiTurnChat(model::Qwen2Model* model, bool cache) {
  auto created = MakeScheduler(model, cache);
  if (!created.ok()) return {};
  Scheduler sched = *std::move(created);

  // The conversation so far: everything both sides have said. Each turn resends
  // it and appends, which is what a chat client does.
  std::vector<int32_t> conversation = Tokens(kChatOpening, 7);

  std::vector<Served> served;
  int64_t last_prompt = 0;

  for (int turn = 0; turn < kChatTurns; ++turn) {
    if (turn > 0) {
      // What the model said, then what the user says next.
      const Served& previous = served.back();
      conversation.insert(conversation.end(), previous.output.begin(),
                          previous.output.end());

      const std::vector<int32_t> user =
          Tokens(kChatUserTurn, static_cast<uint64_t>(500 + turn));
      conversation.insert(conversation.end(), user.begin(), user.end());
    }

    last_prompt = static_cast<int64_t>(conversation.size());

    served.push_back(Serve(model, &sched, static_cast<RequestId>(turn + 1),
                           conversation, kChatReply));

    if (!served.back().ok) break;
  }

  Result r = Summarize(served);
  r.prompt_tokens = last_prompt;  // the last and largest turn
  return r;
}

Result DistinctPrompts(model::Qwen2Model* model, bool cache) {
  auto created = MakeScheduler(model, cache);
  if (!created.ok()) return {};
  Scheduler sched = *std::move(created);

  std::vector<Served> served;

  for (int i = 0; i < kRequests; ++i) {
    served.push_back(Serve(model, &sched, static_cast<RequestId>(i + 1),
                           Tokens(kDistinctPrompt, static_cast<uint64_t>(9000 + i)),
                           /*max_tokens=*/8));
  }

  Result r = Summarize(served);
  r.prompt_tokens = kDistinctPrompt;
  return r;
}

void Report(const char* name, int64_t prompt_tokens, const Result& off,
            const Result& on) {
  if (!off.ok || !on.ok) {
    std::fprintf(stderr, "%s: failed\n", name);
    return;
  }

  std::printf("%s (prompt %ld tokens)\n", name, static_cast<long>(prompt_tokens));
  std::printf("  %-14s %11s %11s %13s %10s\n", "", "cold TTFT", "warm TTFT",
              "prefill tok", "reused");

  std::printf("  %-14s %9.1f ms %9.1f ms %13ld %10ld\n", "cache off",
              off.cold_ttft_ms, off.warm_ttft_ms,
              static_cast<long>(off.forwarded), static_cast<long>(off.hit));

  std::printf("  %-14s %9.1f ms %9.1f ms %13ld %10ld\n", "cache on",
              on.cold_ttft_ms, on.warm_ttft_ms,
              static_cast<long>(on.forwarded), static_cast<long>(on.hit));

  const double speedup =
      on.warm_ttft_ms > 0.0 ? off.warm_ttft_ms / on.warm_ttft_ms : 0.0;

  std::printf("  warm TTFT %.2fx, %.0f%% of prompt tokens reused\n\n", speedup,
              on.forwarded > 0
                  ? 100.0 * static_cast<double>(on.hit) /
                        static_cast<double>(on.hit + on.forwarded)
                  : 0.0);
}

int Main(int argc, char** argv) {
  bool fp8 = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--fp8") {
      fp8 = true;
    } else {
      std::fprintf(stderr, "usage: %s [--fp8]\n", argv[0]);
      return 2;
    }
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

  // Generous, so that nothing here is measuring eviction. What eviction costs
  // under pressure is a different question and belongs with a different bench.
  if (const Status s = model.AttachKvCache(4096, kBlockSize); !s.ok()) {
    std::fprintf(stderr, "cannot attach KV cache: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.ReserveActivations(2048); !s.ok()) {
    std::fprintf(stderr, "reserve activations: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.EnableDeviceSampling(4); !s.ok()) {
    std::fprintf(stderr, "device sampling: %s\n", s.ToString().c_str());
    return 1;
  }

  if (fp8) {
    if (const Status s = model.QuantizeWeightsToF8(); !s.ok()) {
      std::fprintf(stderr, "cannot quantize: %s\n", s.ToString().c_str());
      return 1;
    }
  }

  std::printf("%s\n", model.config().ToString().c_str());
  std::printf("weights: %s, %.2f GB\n\n", fp8 ? "fp8 e4m3" : "bf16",
              model.WeightBytes() / 1e9);

  // A throwaway request first: the first prefill of the process pays for
  // allocator growth and cuBLASLt's per-shape heuristic, and charging that to
  // whichever workload happens to run first would make the cache look worse or
  // better than it is depending on the order.
  {
    auto warm = MakeScheduler(&model, /*cache=*/false);
    if (warm.ok()) {
      Scheduler s = *std::move(warm);
      (void)Serve(&model, &s, 1, Tokens(512, 42), 4);
    }
  }

  {
    const Result off = SharedSystemPrompt(&model, false);
    const Result on = SharedSystemPrompt(&model, true);
    Report("shared system prompt", kSystemPrompt + kQuestion, off, on);
  }

  {
    const Result off = MultiTurnChat(&model, false);
    const Result on = MultiTurnChat(&model, true);
    Report("multi-turn chat, 6 turns", on.prompt_tokens, off, on);
  }

  {
    const Result off = DistinctPrompts(&model, false);
    const Result on = DistinctPrompts(&model, true);
    Report("distinct prompts (nothing to share)", kDistinctPrompt, off, on);
  }

  std::printf(
      "cold TTFT is the first request, which can never hit. warm TTFT is the\n"
      "mean of the rest. prefill tok is what the model actually forwarded\n"
      "across the whole workload before each first token.\n");

  return 0;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
