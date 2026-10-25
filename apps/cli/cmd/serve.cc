#include "inferx/server/serve.h"

#include <CLI/CLI.hpp>
#include <memory>
#include <string>
#include <vector>

#include "cli/args/engine_args.h"
#include "cli/args/model_config_args.h"
#include "cli/args/sampling_args.h"
#include "cli/commands.h"
#include "cli/error.h"
#include "inferx/config/generation_config.h"
#include "inferx/config/merge_config.h"
#include "inferx/server/serve.h"

namespace inferx::cli {
namespace {

struct ServeArgs {
  ModelConfigArgs model;
  EngineArgs engine;
  SamplingArgs sampling;
  std::string host = "127.0.0.1";  // vLLM: --host
  int port = 8000;                 // vLLM: --port
  std::vector<std::string> api_keys;  // vLLM: --api-key (repeatable)
  int tokenizer_workers = 2;
  int tokenizer_queue_limit = 64;

  /// The serving configuration assembled from every args group. The
  /// server-side flag surface grows with the config documentation step;
  /// engine members come straight from the engine flags.
  ServeConfig Build() const {
    ServeConfig config;
    config.server.api.host = host;
    config.server.api.port = port;
    config.server.api.api_keys = api_keys;
    config.server.tokenizer_workers = static_cast<std::size_t>(tokenizer_workers);
    config.server.tokenizer_queue_limit = static_cast<std::size_t>(tokenizer_queue_limit);
    config.model = model.Build();
    config.cache = engine.BuildCacheConfig();
    config.scheduler = engine.BuildSchedulerConfig();
    config.execution = engine.BuildExecutionConfig();
    config.parallel = engine.BuildParallelConfig();
    ApplyGenerationConfig(&config.default_sampling);
    return config;
  }

  /// \brief vLLM --generation-config semantics for the served defaults:
  ///        the checkpoint's generation_config.json sets defaults,
  ///        --override-generation-config merges over it at the JSON level
  ///        (both inside LoadGenerationConfig), and fields the user set
  ///        explicitly win over everything (skip set here). The whole chain
  ///        is validated at startup so bad configs fail the command.
  void ApplyGenerationConfig(sampling::SamplingParams* defaults) const {
    auto loaded = LoadGenerationConfig(model.config);
    if (!loaded.ok()) throw CommandError(loaded.status());
    MergeGenerationConfig(*loaded, defaults, sampling.ExplicitFields());
    ThrowIfError(defaults->Validate());
  }
};

}  // namespace

void RegisterServe(CLI::App& root) {
  // Options bind to *args fields by reference and the App (plus its callback)
  // outlives this function, so the state must too. CLI11 callbacks are
  // std::function (copyable), which rules out unique_ptr: shared_ptr is the
  // idiom.
  auto args = std::make_shared<ServeArgs>();
  CLI::App* sub = root.add_subcommand(
      "serve",
      "Launch a local OpenAI-compatible API server to serve LLM completions via HTTP.");
  args->model.AddOptions(*sub);
  args->engine.AddOptions(*sub);
  args->sampling.AddOptions(*sub);
  sub->add_option("--host", args->host, "Host address")
      ->type_name("HOST")
      ->capture_default_str();
  sub->add_option("--port", args->port, "Port number")
      ->type_name("PORT")
      ->capture_default_str()
      ->check(CLI::Range(1, 65535));
  sub->add_option("--api-key", args->api_keys,
                  "Bearer API key guarding /v1 (repeatable; empty disables auth)")
      ->type_name("KEY");
  sub->add_option("--tokenizer-workers", args->tokenizer_workers,
                  "Prompt-preparation worker threads")
      ->type_name("N")
      ->capture_default_str()
      ->check(CLI::Range(1, 1024));
  sub->add_option("--tokenizer-queue-limit", args->tokenizer_queue_limit,
                  "Max queued prompt-preparation jobs")
      ->type_name("N")
      ->group("Advanced")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->callback([args] {
    ServeConfig config = args->Build();
    MergeEnvironmentOverrides(&config.server);
    ThrowIfError(RunServe(config));
  });
}

}  // namespace inferx::cli
