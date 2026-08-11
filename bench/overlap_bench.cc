/// How much of a decode step the host is responsible for, and so what §5.2's
/// overlap pipeline could possibly be worth.
///
/// M6's goal is G4: the CPU is never the bottleneck. The way §5.2 gets there is
/// to run the scheduler one step ahead, so that planning step N+1 happens while
/// step N is on the GPU rather than after it. The most that can buy is whatever
/// the host is currently doing *serially* with the device -- so before building
/// the pipeline it is worth knowing how large that is, because a 400
/// microsecond saving against an 11 millisecond step is a different project
/// from a 4 millisecond one.
///
/// One engine step is four host-visible phases:
///
///   - **prepare** — `Scheduler::PrepareStep`. Pure host, no CUDA. Admission,
///     the budget plan, block-table growth, the slot arithmetic.
///   - **issue** — `Qwen2Model::StepAsync`. Host work (the FlashInfer planner,
///     the index uploads, and either ~400 kernel launches or one graph replay)
///     that returns without waiting for any of it.
///   - **await** — `Qwen2Model::AwaitStep`. Blocks. Whatever is left of the
///     GPU's work after `issue` handed it over.
///   - **commit** — `Scheduler::CommitStep`. Pure host. Appending tokens,
///     checking stop conditions, retiring.
///
/// `prepare + issue + commit` is host time on the critical path. Overlapping
/// perfectly would hide `min(host, await)` of it per step, which is the ceiling
/// reported below. It is a ceiling and not a forecast: a real depth-1 pipeline
/// still has to launch, and the launch cannot overlap with itself.
///
/// Both dispatch modes are measured, because they are different problems.
/// Launch-by-launch, `issue` is ~400 launches and dominates the host side.
/// Graphed, it is one replay, and what is left is the scheduler and the
/// uploads -- which is the number that says whether M6 has anything to do.
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

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/model/qwen2.h"
#include "inferx/scheduler/scheduler.h"

namespace inferx::bench {
namespace {

using model::ForwardBatch;
using scheduler::RequestId;
using scheduler::SamplingParams;
using scheduler::Scheduler;
using scheduler::SchedulerConfig;

constexpr int64_t kBlockSize = 16;

/// Batch sizes swept. G4 states its target at 256, which does not fit here;
/// what these show is the trend, and the trend is what says whether the host
/// share grows with the batch.
constexpr int64_t kBatches[] = {1, 4, 8, 16, 32};

/// Long enough that every sequence is past its prefill and the steps being
/// timed are all decode.
constexpr int64_t kPrompt = 64;
constexpr int32_t kGenerate = 2000;

constexpr int kWarmupSteps = 20;
constexpr int kMeasuredSteps = 60;

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

/// Median of a copy, because a single slow step from a clock transition drags a
/// mean and the question here is what a typical step costs.
double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

struct Phases {
  double prepare_ms = 0.0;
  double issue_ms = 0.0;
  double await_ms = 0.0;
  double commit_ms = 0.0;
  bool ok = false;

  double host_ms() const { return prepare_ms + issue_ms + commit_ms; }
  double step_ms() const { return host_ms() + await_ms; }

  /// The most a perfect depth-1 pipeline could remove from a step: host work
  /// hides behind device work only up to whichever is smaller.
  double hideable_ms() const { return std::min(host_ms(), await_ms); }
};

Phases Measure(model::Qwen2Model* model, int64_t batch, bool graphs) {
  Phases out;

  SchedulerConfig config;
  config.max_running = batch;
  config.max_batch_tokens = 4096;
  config.max_seq_len = 4096;
  // Off: a hit would change how much prefill runs before the steady state and
  // has nothing to do with per-step host cost.
  config.enable_prefix_cache = false;

  auto created = Scheduler::Create(config, model->kv_pool());
  if (!created.ok()) {
    std::fprintf(stderr, "scheduler: %s\n",
                 created.status().ToString().c_str());
    return out;
  }
  Scheduler sched = *std::move(created);

  SamplingParams params;
  params.max_tokens = kGenerate;

  for (int64_t i = 0; i < batch; ++i) {
    std::vector<int32_t> prompt(static_cast<size_t>(kPrompt));
    for (size_t k = 0; k < prompt.size(); ++k) {
      prompt[k] = static_cast<int32_t>(1000 + (i * 131 + k) % 50000);
    }

    if (const Status s = sched.AddRequest(static_cast<RequestId>(i + 1),
                                          std::move(prompt), params);
        !s.ok()) {
      std::fprintf(stderr, "add: %s\n", s.ToString().c_str());
      return out;
    }
  }

  std::vector<int32_t> sampled;

  const auto one_step = [&](double* prepare, double* issue, double* await,
                            double* commit) {
    ForwardBatch b;

    const auto t0 = std::chrono::steady_clock::now();
    const Status prepared = sched.PrepareStep(&b);
    const auto t1 = std::chrono::steady_clock::now();

    if (!prepared.ok() || b.num_tokens() == 0) return false;

    const Status issued = model->StepAsync(b);
    const auto t2 = std::chrono::steady_clock::now();

    if (!issued.ok()) {
      std::fprintf(stderr, "issue: %s\n", issued.ToString().c_str());
      return false;
    }

    const Status awaited = model->AwaitStep(&sampled);
    const auto t3 = std::chrono::steady_clock::now();

    if (!awaited.ok()) {
      std::fprintf(stderr, "await: %s\n", awaited.ToString().c_str());
      return false;
    }

    const Status committed = sched.CommitStep(sampled, nullptr);
    const auto t4 = std::chrono::steady_clock::now();

    if (!committed.ok()) return false;

    (void)sched.TakeCompleted();

    *prepare = std::chrono::duration<double, std::milli>(t1 - t0).count();
    *issue = std::chrono::duration<double, std::milli>(t2 - t1).count();
    *await = std::chrono::duration<double, std::milli>(t3 - t2).count();
    *commit = std::chrono::duration<double, std::milli>(t4 - t3).count();

    return true;
  };

  // Warm-up gets every sequence through its prefill and into decode, and pays
  // for the allocator and cuBLASLt's per-shape heuristic.
  for (int i = 0; i < kWarmupSteps; ++i) {
    double a = 0, b = 0, c = 0, d = 0;
    if (!one_step(&a, &b, &c, &d)) return out;
  }

  // Captured after the shape is live: a graph can only be recorded for a shape
  // that has already been executed.
  if (graphs) {
    const int64_t max_blocks =
        (config.max_seq_len + kBlockSize - 1) / kBlockSize;
    if (const Status s = model->CaptureDecodeGraph(batch, max_blocks);
        !s.ok()) {
      std::fprintf(stderr, "capture: %s\n", s.ToString().c_str());
      return out;
    }

    for (int i = 0; i < 5; ++i) {
      double a = 0, b = 0, c = 0, d = 0;
      if (!one_step(&a, &b, &c, &d)) return out;
    }
  }

  std::vector<double> prepare, issue, await, commit;

  for (int i = 0; i < kMeasuredSteps; ++i) {
    double a = 0, b = 0, c = 0, d = 0;
    if (!one_step(&a, &b, &c, &d)) return out;

    prepare.push_back(a);
    issue.push_back(b);
    await.push_back(c);
    commit.push_back(d);
  }

  out.prepare_ms = Median(std::move(prepare));
  out.issue_ms = Median(std::move(issue));
  out.await_ms = Median(std::move(await));
  out.commit_ms = Median(std::move(commit));
  out.ok = true;

  return out;
}

/// How often a prompt arrives in the mixed workload.
constexpr int kArrivalPeriod = 24;
constexpr int64_t kMixedDecoders = 8;
constexpr int64_t kMixedPrompt = 1024;
constexpr int kMixedSteps = 240;

struct Mixed {
  int decode_steps = 0;
  int prefill_steps = 0;

  double decode_prepare = 0.0, decode_issue = 0.0, decode_await = 0.0;
  double prefill_prepare = 0.0, prefill_issue = 0.0, prefill_await = 0.0;

  double total_host_ms = 0.0;
  double total_wall_ms = 0.0;
  bool ok = false;
};

/// A server-shaped stream: sequences decoding throughout, with prompts arriving
/// into the same steps. Graphs on, so decode-only steps replay and steps
/// carrying a prefill chunk do not.
Mixed MeasureMixed(model::Qwen2Model* model) {
  Mixed out;

  SchedulerConfig config;
  config.max_running = kMixedDecoders + 2;
  config.max_batch_tokens = 2048;
  config.max_seq_len = 4096;
  config.enable_prefix_cache = false;

  auto created = Scheduler::Create(config, model->kv_pool());
  if (!created.ok()) return out;
  Scheduler sched = *std::move(created);

  SamplingParams params;
  params.max_tokens = kGenerate;

  const auto make = [](int64_t n, uint64_t seed) {
    std::vector<int32_t> t(static_cast<size_t>(n));
    for (size_t i = 0; i < t.size(); ++i) {
      t[i] = static_cast<int32_t>(1000 + (seed * 131 + i * 17) % 50000);
    }
    return t;
  };

  RequestId next = 1;

  for (int64_t i = 0; i < kMixedDecoders; ++i) {
    if (!sched
             .AddRequest(next++, make(kPrompt, static_cast<uint64_t>(i)),
                         params)
             .ok()) {
      return out;
    }
  }

  std::vector<int32_t> sampled;

  // Warm up and capture the decode-only shape, so graphed steps really are
  // graphed.
  for (int i = 0; i < kWarmupSteps; ++i) {
    ForwardBatch b;
    if (!sched.PrepareStep(&b).ok() || b.num_tokens() == 0) return out;
    if (!model->StepAsync(b).ok()) return out;
    if (!model->AwaitStep(&sampled).ok()) return out;
    if (!sched.CommitStep(sampled, nullptr).ok()) return out;
    (void)sched.TakeCompleted();
  }

  // Every resident count, largest first -- exactly what the server does at
  // startup. A mixed workload's sequence count changes whenever a request
  // arrives or finishes, so capturing only one shape would measure graphs
  // missing rather than graphs working. Descending because capture records
  // device addresses and preparing a bigger batch grows buffers that are sized
  // on demand, which would strand every smaller graph already recorded.
  const int64_t max_blocks = (config.max_seq_len + kBlockSize - 1) / kBlockSize;

  for (int64_t seqs = config.max_running; seqs >= 1; --seqs) {
    (void)model->CaptureDecodeGraph(seqs, max_blocks);
  }

  std::vector<double> d_prep, d_issue, d_await, p_prep, p_issue, p_await;

  for (int step = 0; step < kMixedSteps; ++step) {
    if (step % kArrivalPeriod == 0) {
      (void)sched.AddRequest(
          next++, make(kMixedPrompt, static_cast<uint64_t>(500 + step)),
          params);
    }

    ForwardBatch b;

    const auto t0 = std::chrono::steady_clock::now();
    if (!sched.PrepareStep(&b).ok()) return out;
    const auto t1 = std::chrono::steady_clock::now();

    if (b.num_tokens() == 0) continue;

    // A step is "decode-only" when every sequence contributes exactly one
    // token, which is the shape a captured graph matches.
    const bool decode_only = b.num_tokens() == b.num_seqs;

    if (!model->StepAsync(b).ok()) return out;
    const auto t2 = std::chrono::steady_clock::now();

    if (!model->AwaitStep(&sampled).ok()) return out;
    const auto t3 = std::chrono::steady_clock::now();

    if (!sched.CommitStep(sampled, nullptr).ok()) return out;
    const auto t4 = std::chrono::steady_clock::now();

    (void)sched.TakeCompleted();

    const double prep =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double issue =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    const double await =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double commit =
        std::chrono::duration<double, std::milli>(t4 - t3).count();

    out.total_host_ms += prep + issue + commit;
    out.total_wall_ms += prep + issue + await + commit;

    if (decode_only) {
      ++out.decode_steps;
      d_prep.push_back(prep);
      d_issue.push_back(issue);
      d_await.push_back(await);
    } else {
      ++out.prefill_steps;
      p_prep.push_back(prep);
      p_issue.push_back(issue);
      p_await.push_back(await);
    }
  }

  out.decode_prepare = Median(std::move(d_prep));
  out.decode_issue = Median(std::move(d_issue));
  out.decode_await = Median(std::move(d_await));
  out.prefill_prepare = Median(std::move(p_prep));
  out.prefill_issue = Median(std::move(p_issue));
  out.prefill_await = Median(std::move(p_await));
  out.ok = out.decode_steps > 0 && out.prefill_steps > 0;

  return out;
}

void ReportRow(int64_t batch, const Phases& p) {
  std::printf("%6ld %9.3f %9.3f %9.3f %9.3f %9.3f %8.1f%% %9.2f%%\n",
              static_cast<long>(batch), p.prepare_ms, p.issue_ms, p.await_ms,
              p.commit_ms, p.step_ms(), 100.0 * p.host_ms() / p.step_ms(),
              100.0 * p.hideable_ms() / p.step_ms());
}

int Main(int argc, char** argv) {
  if (argc > 1) {
    std::fprintf(stderr, "usage: %s\n", argv[0]);
    return 2;
  }

  if (!cuda::Available()) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }

  auto loaded =
      model::Qwen2Model::LoadFromDirectory(CheckpointDir(), DeviceId::Cuda(0));
  if (!loaded.ok()) {
    std::fprintf(stderr, "cannot load model: %s\n",
                 loaded.status().ToString().c_str());
    return 1;
  }

  model::Qwen2Model model = *std::move(loaded);

  constexpr int64_t kMaxBatch =
      *std::max_element(std::begin(kBatches), std::end(kBatches));

  if (const Status s = model.AttachKvCache(4096, kBlockSize); !s.ok()) {
    std::fprintf(stderr, "attach: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.ReserveActivations(4096); !s.ok()) {
    std::fprintf(stderr, "reserve: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.EnableDeviceSampling(kMaxBatch); !s.ok()) {
    std::fprintf(stderr, "sampling: %s\n", s.ToString().c_str());
    return 1;
  }

  std::printf("%s\n", model.config().ToString().c_str());
  std::printf("weights: bf16, %.2f GB\n\n", model.WeightBytes() / 1e9);
  std::printf(
      "One decode step, split by phase. prepare and commit are the scheduler;\n"
      "issue is StepAsync, which returns without waiting; await is what is\n"
      "left of the device's work. host%% is (prepare+issue+commit)/step, and\n"
      "ceiling is the most a perfect depth-1 pipeline could hide.\n\n");

  std::printf("launch-by-launch\n");
  std::printf("%6s %9s %9s %9s %9s %9s %9s %10s\n", "batch", "prepare", "issue",
              "await", "commit", "step", "host_%", "ceiling");
  std::printf("%s\n", std::string(78, '-').c_str());

  int failures = 0;

  for (const int64_t batch : kBatches) {
    const Phases p = Measure(&model, batch, /*graphs=*/false);
    if (!p.ok) {
      ++failures;
      continue;
    }
    ReportRow(batch, p);
  }

  std::printf("\nwith CUDA graphs\n");
  std::printf("%6s %9s %9s %9s %9s %9s %9s %10s\n", "batch", "prepare", "issue",
              "await", "commit", "step", "host_%", "ceiling");
  std::printf("%s\n", std::string(78, '-').c_str());

  for (const int64_t batch : kBatches) {
    const Phases p = Measure(&model, batch, /*graphs=*/true);
    if (!p.ok) {
      ++failures;
      continue;
    }
    ReportRow(batch, p);
  }

  std::printf(
      "\nThe ceiling is a bound, not a forecast: a depth-1 pipeline still has\n"
      "to issue each step, and issuing cannot overlap with itself.\n\n");

  // Pure decode is the friendliest case for graphs, because every step has a
  // shape that has been captured. A server does not look like that: chunked
  // prefill puts a differently-shaped step in the stream whenever a prompt
  // arrives, and those steps fall back to launch-by-launch, where the host
  // share was 22%. If the mixed number is much worse than the decode one, the
  // overlap pipeline has a case that the table above does not show.
  {
    const Mixed m = MeasureMixed(&model);

    if (!m.ok) {
      ++failures;
    } else {
      std::printf(
          "mixed workload: 8 decoding, a 1024-token prompt every %d steps\n",
          kArrivalPeriod);
      std::printf("%14s %7s %9s %9s %9s %9s\n", "steps", "count", "prepare",
                  "issue", "await", "host_%");
      std::printf("%s\n", std::string(62, '-').c_str());

      std::printf("%14s %7d %9.3f %9.3f %9.3f %8.1f%%\n", "decode-only",
                  m.decode_steps, m.decode_prepare, m.decode_issue,
                  m.decode_await,
                  100.0 * (m.decode_prepare + m.decode_issue) /
                      (m.decode_prepare + m.decode_issue + m.decode_await));

      std::printf("%14s %7d %9.3f %9.3f %9.3f %8.1f%%\n", "with prefill",
                  m.prefill_steps, m.prefill_prepare, m.prefill_issue,
                  m.prefill_await,
                  100.0 * (m.prefill_prepare + m.prefill_issue) /
                      (m.prefill_prepare + m.prefill_issue + m.prefill_await));

      std::printf("%14s %7d %9s %9s %9s %8.1f%%\n", "all",
                  m.decode_steps + m.prefill_steps, "", "", "",
                  100.0 * m.total_host_ms / m.total_wall_ms);
    }
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
