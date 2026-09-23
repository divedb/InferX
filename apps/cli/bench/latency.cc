// `inferx bench latency` — batch-latency benchmark; execution pending.
#include <CLI/CLI.hpp>
#include <memory>

#include "cli/args/bench_output_args.h"
#include "cli/args/engine_args.h"
#include "cli/args/model_config.h"
#include "cli/commands.h"
#include "cli/error.h"
#include "inferx/bench/latency.h"

namespace inferx::cli {
namespace {

struct LatencyArgs {
  ModelConfigArgs model;
  EngineArgs engine;
  int batch_size = 8;    // vLLM bench latency: --batch-size
  int output_len = 256;  // vLLM bench latency: --output-len
  int input_len = 32;         // vLLM bench latency: --input-len
  int num_iters = 30;         // vLLM bench latency: --num-iters
  int num_iters_warmup = 10;  // vLLM bench latency: --num-iters-warmup
  BenchOutputArgs output;

  bench::LatencyParams Build() const {
    bench::LatencyParams p;
    p.model = model.Build();
    p.cache = engine.BuildCacheConfig();
    p.scheduler = engine.BuildSchedulerConfig();
    p.execution = engine.BuildExecutionConfig();
    p.scheduler = engine.BuildSchedulerConfig();
    p.batch_size = batch_size;
    p.output_len = output_len;
    p.input_len = input_len;
    p.num_iters = num_iters;
    p.num_iters_warmup = num_iters_warmup;
    p.output_json = output.output_json;
    p.disable_log_stats = output.disable_log_stats;
    return p;
  }
};

}  // namespace

void RegisterBenchLatency(CLI::App& parent) {
  auto args = std::make_shared<LatencyArgs>();
  CLI::App* sub =
      parent.add_subcommand("latency", "Benchmark the latency of a single batch of requests.");
  args->model.AddOptions(*sub);
  args->engine.AddOptions(*sub);
  sub->add_option("--batch-size", args->batch_size, "Number of requests in the batch")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--output-len", args->output_len, "Tokens generated per request")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--input-len", args->input_len, "Input length of the synthetic prompts")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--num-iters", args->num_iters, "Number of iterations to run")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--num-iters-warmup", args->num_iters_warmup,
                  "Number of iterations to run for warmup")
      ->capture_default_str()
      ->check(CLI::Range(0, 1 << 20));
  args->output.AddOptions(*sub);
  sub->callback([args] { ThrowIfError(bench::RunLatency(args->Build())); });
}

}  // namespace inferx::cli
