// `inferx bench throughput`. This file is the complete CLI-side story for the
// command: surface struct, options, validation, conversion, dispatch. The
// benchmark itself lives in src/bench/throughput.cc.
#include <CLI/CLI.hpp>
#include <memory>

#include "cli/args/bench_output_args.h"
#include "cli/args/dataset_args.h"
#include "cli/args/engine_args.h"
#include "cli/args/model_config.h"
#include "cli/commands.h"
#include "cli/error.h"
#include "inferx/bench/throughput.h"

namespace inferx::cli {
namespace {

/// `inferx bench throughput` surface: vLLM's throughput groups (model +
/// engine + dataset) plus this command's own knobs. Plain data; parsing
/// fills it, Build() converts it, the callback dispatches with it.
struct ThroughputArgs {
  ModelConfigArgs model;
  EngineArgs engine;
  DatasetArgs dataset;
  int num_prompts = 64;
  int num_iters = 3;         // Timed trials after warmup; workload convention
  int num_iters_warmup = 1;  // (vLLM's throughput bench has no such flags)
  BenchOutputArgs output;

  bench::ThroughputParams Build() const {
    bench::ThroughputParams p;
    p.model = model.Build();
    p.cache = engine.BuildCacheConfig();
    p.scheduler = engine.BuildSchedulerConfig();
    p.execution = engine.BuildExecutionConfig();
    p.dataset = dataset.Build();
    p.dataset.seed = model.seed;  // vLLM folds --seed into the model args
    p.num_prompts = num_prompts;
    p.num_iters = num_iters;
    p.num_iters_warmup = num_iters_warmup;
    p.output_json = output.output_json;
    p.disable_log_stats = output.disable_log_stats;
    return p;
  }
};

}  // namespace

void RegisterBenchThroughput(CLI::App& parent) {
  auto args = std::make_shared<ThroughputArgs>();
  CLI::App* sub =
      parent.add_subcommand("throughput", "Benchmark offline inference throughput.");
  args->model.AddOptions(*sub);
  args->engine.AddOptions(*sub);
  args->dataset.AddOptions(*sub);
  sub->add_option("--num-prompts", args->num_prompts, "Number of prompts to run")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--num-iters", args->num_iters,
                  "Number of timed trials to run after warmup")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--num-iters-warmup", args->num_iters_warmup,
                  "Number of trials to run for warmup")
      ->capture_default_str()
      ->check(CLI::Range(0, 1 << 20));
  args->output.AddOptions(*sub);
  sub->callback([args] {
    // Cross-option validation runs inside parse(), so throwing
    // ValidationError lands in main's CLI::ParseError handler with usage
    // output and a proper exit code.
    if (args->dataset.name != "random" && !args->dataset.path)
      throw CLI::ValidationError("inferx bench throughput",
                                 "--dataset-path is required unless --dataset-name random");
    ThrowIfError(bench::RunThroughput(args->Build()));
  });
}

}  // namespace inferx::cli
