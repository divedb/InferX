// `inferx worker` -- one tensor-parallel rank. Spawned by the controller
// (WorkerPool) with rank, world, device, and channel names on the command
// line; the model/engine flag groups are the same ones every other command
// uses, so a worker is configured exactly like the in-process engine.
#include <CLI/CLI.hpp>
#include <string>

#include "cli/args/engine_args.h"
#include "cli/args/model_config_args.h"
#include "cli/commands.h"
#include "cli/error.h"
#include "inferx/config/parallel_config.h"
#include "inferx/dist/worker_ipc.h"
#include "inferx/dist/worker_main.h"

namespace inferx::cli {
namespace {

struct WorkerArgs {
  ModelConfigArgs model;
  EngineArgs engine;
  int rank = 0;
  int world = 1;
  std::string ipc_commands;
  std::string ipc_events;

  void AddOptions(CLI::App& sub) {
    model.AddOptions(sub);
    engine.AddOptions(sub);
    sub.add_option("--rank", rank, "This worker's tensor-parallel rank")
        ->capture_default_str()
        ->check(CLI::Range(0, 255));
    sub.add_option("--world", world, "Tensor-parallel world size")
        ->capture_default_str()
        ->check(CLI::Range(1, 255));
    sub.add_option("--ipc-commands", ipc_commands,
                   "Message-queue name for controller-to-worker commands")
        ->required();
    sub.add_option("--ipc-events", ipc_events,
                   "Message-queue name for worker-to-controller events")
        ->required();
  }

  dist::WorkerOptions Build() const {
    dist::WorkerOptions options;
    options.model = model.Build();
    options.cache = engine.BuildCacheConfig();
    options.scheduler = engine.BuildSchedulerConfig();
    options.execution = engine.BuildExecutionConfig();
    options.parallel = engine.BuildParallelConfig();
    options.parallel.tensor_parallel_rank = rank;
    options.parallel.tensor_parallel_size = world;
    return options;
  }
};

}  // namespace

void RegisterWorker(CLI::App& root) {
  auto args = std::make_shared<WorkerArgs>();
  CLI::App* sub = root.add_subcommand(
      "worker",
      "Internal: one tensor-parallel worker process (spawned by the engine; "
      "not for direct use).");
  args->AddOptions(*sub);
  sub->callback([args] {
    const dist::IpcChannelNames channels{args->ipc_commands, args->ipc_events};
    ThrowIfError(dist::RunWorker(args->Build(), channels));
  });
}

}  // namespace inferx::cli
