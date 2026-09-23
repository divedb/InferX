// `inferx bench workload` (InferX extension: fixed-suite benchmark through
// the real scheduler and runner; not part of vLLM's command tree). Capacity
// knobs come from EngineArgs; the suite contributes workload data.
#include <CLI/CLI.hpp>
#include <memory>
#include <string>

#include "cli/args/bench_output_args.h"
#include "cli/args/engine_args.h"
#include "cli/args/model_config.h"
#include "cli/commands.h"
#include "cli/error.h"
#include "inferx/bench/workload.h"

namespace inferx::cli {
namespace {

struct WorkloadArgs {
  ModelConfigArgs model;
  EngineArgs engine;
  std::string suite = "benchmarks/qwen3/workload.json";
  int repeats = 3;
  int profile_step = -1;
  bool step_timings = false;
  BenchOutputArgs output;

  bench::WorkloadParams Build() const {
    bench::WorkloadParams p;
    p.model = model.Build();
    p.cache = engine.BuildCacheConfig();
    p.scheduler = engine.BuildSchedulerConfig();
    p.execution = engine.BuildExecutionConfig();
    p.scheduler = engine.BuildSchedulerConfig();
    p.suite = suite;
    p.repeats = repeats;
    p.profile_step = profile_step;
    p.step_timings = step_timings;
    p.output_json = output.output_json;
    p.disable_log_stats = output.disable_log_stats;
    return p;
  }
};

}  // namespace

void RegisterBenchWorkload(CLI::App& parent) {
  auto args = std::make_shared<WorkloadArgs>();
  CLI::App* sub = parent.add_subcommand(
      "workload", "Benchmark the fixed workload through the real scheduler and model runner");
  args->model.AddOptions(*sub);
  args->engine.AddOptions(*sub);
  sub->add_option("--suite", args->suite, "Workload JSON")->capture_default_str();
  sub->add_option("--repeats", args->repeats, "Timed trials per case after one warmup")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  // Diagnostic-only profiling range after a full warmup; normal timing is
  // unchanged.
  sub->add_option("--profile-step", args->profile_step,
                  "CUDA profiler range during repeat 0; -1 disables")
      ->capture_default_str()
      ->check(CLI::Range(-1, 1 << 20));
  sub->add_flag("--step-timings", args->step_timings,
                "Emit per-step timing CSV to stderr");
  args->output.AddOptions(*sub);
  sub->callback([args] { ThrowIfError(bench::RunWorkload(args->Build())); });
}

}  // namespace inferx::cli
