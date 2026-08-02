/// Inter-token latency under a mixed workload -- what chunked prefill is for.
///
/// Every number measured so far is one shape in isolation: a decode step at a
/// fixed batch, a prefill of a fixed length, a GEMM. §8.1 is not about any of
/// those. It is about what happens to sequences that are *already decoding*
/// when a prompt arrives beside them, and that only shows up in a workload
/// where both are present at once.
///
/// The claim being tested is the one §8.1 is written on: a prefill is expensive
/// enough to dominate a step -- 176 ms for 2k tokens against 11.6 ms for a
/// decode -- so running one whole inside a step makes every decoding sequence
/// in that step wait the full 176 ms for its next token. Chunking spreads the
/// same work over more steps, which cannot make the prefill cheaper and is not
/// meant to: it bounds the worst gap any decoder sees. The cost is real and
/// this measures it too, because chunks re-read prior KV and shorter GEMMs are
/// less efficient, so the prompt itself finishes later.
///
/// So: one workload, swept over the chunk size, reporting both sides.
///
///   - **TBT** is the gap between one sequence's consecutive tokens. It is a
///     wall-clock property of what a client experiences, so it is measured on
///     the wall clock and not with CUDA events. The p99 and the max are the
///     interesting columns; the median mostly measures the decode step and is
///     printed as a control.
///   - **TTFT** is the arriving prompt's own latency, from the step it is
///     submitted on to the step that produces its first token. This is the
///     column that should get *worse* as chunks get smaller.
///
/// There is no unchunked configuration to compare against, because there is no
/// longer an unchunked code path -- a budget larger than the prompt is what
/// that was, and it is the last row of the sweep.
///
/// Graphs are off. A captured graph fixes the step's shape, and the whole point
/// of this workload is that the shape changes every step; leaving capture on
/// would measure how often the sweep happened to hit a cached shape. Decode
/// steps are correspondingly ~0.4 ms slower here than the M4 table reports,
/// uniformly across every row, which does not disturb a comparison between
/// rows.
///
/// For numbers worth comparing across runs, lock the clocks first:
///     sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 2400

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
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

/// The chunk sizes swept. The top of the range exceeds the arriving prompt, so
/// the last row runs it whole in one step and is the "no chunking" case.
constexpr int64_t kBudgets[] = {128, 256, 512, 1024, 2048, 4096};

/// Sequences decoding throughout. These are the ones whose TBT is measured.
constexpr int64_t kDecoders = 8;

/// Short, because their prefill is not what is being measured -- they only have
/// to be resident and generating before the timed window opens.
constexpr int64_t kDecoderPrompt = 16;

/// Large enough that no decoder finishes inside the measurement and starts
/// perturbing the batch size.
constexpr int32_t kGenerate = 2000;

/// The prompt that arrives. 2048 is the length the prefill table prices at
/// 176 ms, which is ~16 decode steps -- big enough for the effect to be the
/// dominant term rather than noise.
constexpr int64_t kPromptLen = 2048;

/// Timed steps per row. Enough that the whole prefill fits even at the smallest
/// chunk size (2048/128 = 16 steps) with the rest of the window spent back in
/// steady-state decode, so every row is summarized over a comparable number of
/// samples.
constexpr int kTimedSteps = 56;

/// Timed steps before the prompt is submitted.
///
/// Not zero, and the reason is not realism. A decoder's gap can only be charged
/// once it has a previous token recorded, so if the prompt arrives on the
/// window's first step, the step carrying the prefill is the one step whose
/// gaps get discarded -- which is precisely the measurement. This benchmark
/// reported a 12.67 ms p99 beside a 167 ms step before that was fixed.
constexpr int kPreArrival = 8;

constexpr int64_t kBlockSize = 16;

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

/// Linear-interpolated percentile over a sorted copy. Reported rather than a
/// mean because the whole question is about the tail: a workload where one step
/// in twenty is 15x the others has a perfectly reasonable mean.
double Percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());

  const double rank = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(rank);
  const size_t hi = std::min(lo + 1, v.size() - 1);

  return v[lo] + (rank - static_cast<double>(lo)) * (v[hi] - v[lo]);
}

struct Run {
  /// Every inter-token gap of every decoding sequence, in the timed window.
  std::vector<double> gaps_ms;
  /// The arriving prompt's time to its first token. Zero when none arrived.
  double ttft_ms = 0.0;
  /// Steps the arriving prompt was spread over, which is what TTFT buys.
  int prefill_steps = 0;
  double longest_step_ms = 0.0;
  double window_ms = 0.0;
  bool ok = false;
};

/// Runs the workload once at one chunk size.
///
/// \param with_prompt  False for the control row: the same decoders, the same
///                     number of steps, nothing arriving. That row is the TBT
///                     floor, and without it a p99 of 12 ms would look like an
///                     achievement rather than just the decode step.
Run Measure(model::Qwen2Model* model, int64_t budget, bool with_prompt) {
  Run run;

  SchedulerConfig config;
  config.max_running = kDecoders + 1;
  config.max_batch_tokens = budget;
  config.max_seq_len = kPromptLen + 64;

  auto created = Scheduler::Create(config, model->kv_pool());
  if (!created.ok()) {
    std::fprintf(stderr, "scheduler: %s\n",
                 created.status().ToString().c_str());
    return run;
  }
  Scheduler sched = *std::move(created);

  SamplingParams params;
  params.max_tokens = kGenerate;

  const std::vector<int32_t> filler(static_cast<size_t>(kDecoderPrompt), 785);

  for (int64_t i = 0; i < kDecoders; ++i) {
    if (const Status s =
            sched.AddRequest(static_cast<RequestId>(i + 1), filler, params);
        !s.ok()) {
      std::fprintf(stderr, "add: %s\n", s.ToString().c_str());
      return run;
    }
  }

  std::vector<int32_t> sampled;
  std::vector<TokenDelta> deltas;

  // One step of the loop. Returns its wall time, or a negative number on
  // failure. Wall clock and not CUDA events: AwaitStep synchronizes, so this
  // brackets exactly the interval a client is waiting through, host work
  // included -- which is what TBT means.
  const auto step = [&](double* out_ms) {
    ForwardBatch batch;
    if (const Status s = sched.PrepareStep(&batch); !s.ok()) {
      std::fprintf(stderr, "prepare: %s\n", s.ToString().c_str());
      return false;
    }

    deltas.clear();
    if (batch.num_tokens() == 0) {
      *out_ms = 0.0;
      return true;
    }

    const auto t0 = std::chrono::steady_clock::now();

    Status s = model->StepAsync(batch);
    if (s.ok()) s = model->AwaitStep(&sampled);

    const auto t1 = std::chrono::steady_clock::now();

    if (!s.ok()) {
      std::fprintf(stderr, "step: %s\n", s.ToString().c_str());
      return false;
    }

    if (const Status c = sched.CommitStep(sampled, &deltas); !c.ok()) {
      std::fprintf(stderr, "commit: %s\n", c.ToString().c_str());
      return false;
    }

    *out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
  };

  // Warm-up, untimed: get every decoder past its prefill and into steady-state
  // generation, and let the allocator and the clocks settle. Timing from step
  // zero would put eight prefills in the first sample of every row.
  for (int i = 0; i < 12; ++i) {
    double ms = 0.0;
    if (!step(&ms)) return run;
  }

  constexpr RequestId kArriving = 999;

  const std::vector<int32_t> prompt(static_cast<size_t>(kPromptLen), 785);

  // Timed window. Per step: how long it took, and which requests it produced a
  // token for. A decoder's TBT is the elapsed time between the steps that
  // produced its consecutive tokens, which is the sum of everything in between
  // -- including the steps it was not in.
  absl::flat_hash_map<RequestId, double> last_token_at;
  double elapsed = 0.0;
  double arrival_ms = 0.0;
  bool submitted = false;

  for (int i = 0; i < kTimedSteps; ++i) {
    if (with_prompt && i == kPreArrival) {
      if (const Status s = sched.AddRequest(kArriving, prompt, params);
          !s.ok()) {
        std::fprintf(stderr, "add prompt: %s\n", s.ToString().c_str());
        return run;
      }
      arrival_ms = elapsed;
      submitted = true;
    }

    double ms = 0.0;
    if (!step(&ms)) return run;

    elapsed += ms;
    run.longest_step_ms = std::max(run.longest_step_ms, ms);

    if (submitted && run.ttft_ms == 0.0) ++run.prefill_steps;

    for (const TokenDelta& delta : deltas) {
      if (delta.id == kArriving) {
        // Its first token is the end of its prefill, however many steps that
        // took. Measured from submission, not from the window, so it is the
        // latency the client submitting it would see.
        if (run.ttft_ms == 0.0) run.ttft_ms = elapsed - arrival_ms;
        continue;
      }

      const auto it = last_token_at.find(delta.id);
      if (it != last_token_at.end()) run.gaps_ms.push_back(elapsed - it->second);

      last_token_at[delta.id] = elapsed;
    }
  }

  run.window_ms = elapsed;
  run.ok = true;
  return run;
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

  // Room for nine sequences of up to kPromptLen plus what they generate, with
  // headroom -- a row that ran out of blocks would report a scheduling stall as
  // a latency result.
  if (const Status s = model.AttachKvCache(2048, kBlockSize); !s.ok()) {
    std::fprintf(stderr, "cannot attach KV cache: %s\n", s.ToString().c_str());
    return 1;
  }

  constexpr int64_t kMaxBudget =
      *std::max_element(std::begin(kBudgets), std::end(kBudgets));

  if (const Status s = model.ReserveActivations(kMaxBudget); !s.ok()) {
    std::fprintf(stderr, "reserve activations: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.EnableDeviceSampling(kDecoders + 1); !s.ok()) {
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
  std::printf("weights: %s, %.2f GB\n", fp8 ? "fp8 e4m3" : "bf16",
              model.WeightBytes() / 1e9);
  std::printf("\n%ld sequences decoding; one %ld-token prompt arrives.\n",
              static_cast<long>(kDecoders), static_cast<long>(kPromptLen));
  std::printf("%d timed steps per row, CUDA graphs off.\n\n", kTimedSteps);

  // The control first. Every TBT column below has to be read against it: the
  // floor is the decode step itself, and chunking cannot go below it.
  const Run control = Measure(&model, /*budget=*/2048, /*with_prompt=*/false);
  if (!control.ok) return 1;

  std::printf("decoders alone, no prompt arriving (the floor):\n");
  std::printf("  p50 %.2f ms   p99 %.2f ms   max %.2f ms   over %zu gaps\n\n",
              Percentile(control.gaps_ms, 0.50),
              Percentile(control.gaps_ms, 0.99),
              Percentile(control.gaps_ms, 1.00), control.gaps_ms.size());

  std::printf("%8s %9s %9s %10s %9s %8s %11s\n", "chunk", "p50_tbt", "p99_tbt",
              "worst_step", "ttft_ms", "chunks", "prefill_t/s");
  std::printf("%s\n", std::string(72, '-').c_str());

  int failures = 0;
  Run smallest;
  Run largest;

  for (const int64_t budget : kBudgets) {
    const Run run = Measure(&model, budget, /*with_prompt=*/true);

    if (!run.ok) {
      std::fprintf(stderr, "chunk %ld failed\n", static_cast<long>(budget));
      ++failures;
      continue;
    }

    if (!smallest.ok) smallest = run;
    largest = run;

    // Tokens of prompt per second of the client's wait. Not the kernel's
    // prefill throughput -- the decodes interleaved into those steps are in the
    // denominator, which is the point: this is what the prompt actually gets on
    // a busy engine.
    const double prefill_tps =
        run.ttft_ms > 0.0
            ? static_cast<double>(kPromptLen) / (run.ttft_ms * 1e-3)
            : 0.0;

    std::printf("%8ld %9.2f %9.2f %10.2f %9.1f %8d %11.0f\n",
                static_cast<long>(budget), Percentile(run.gaps_ms, 0.50),
                Percentile(run.gaps_ms, 0.99), run.longest_step_ms,
                run.ttft_ms, run.prefill_steps, prefill_tps);
  }

  std::printf(
      "\nchunk is max_batch_tokens. The last row exceeds the prompt, so it\n"
      "runs whole in one step -- that row is what chunking replaced. The\n"
      "2048 row still takes two chunks: the decodes are served first and\n"
      "spend 8 of the budget before the prompt sees any of it.\n");

  if (smallest.ok && largest.ok) {
    const double tbt_before = Percentile(largest.gaps_ms, 0.99);
    const double tbt_after = Percentile(smallest.gaps_ms, 0.99);
    const double floor = Percentile(control.gaps_ms, 0.99);

    std::printf(
        "\nThe trade, across the sweep: p99 TBT %.1f -> %.1f ms (%.1fx, against\n"
        "a %.1f ms floor), TTFT %.0f -> %.0f ms (%.2fx). Chunking does not make\n"
        "the prefill cheaper and is not meant to -- it decides who waits for it.\n",
        tbt_before, tbt_after, tbt_before / tbt_after, floor, largest.ttft_ms,
        smallest.ttft_ms, smallest.ttft_ms / largest.ttft_ms);
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
