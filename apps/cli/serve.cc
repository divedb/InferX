#include "inferx/server/serve.h"

#include <CLI/CLI.hpp>
#include <memory>
#include <string>

#include "cli/args/engine_args.h"
#include "cli/args/model_config.h"
#include "cli/args/sampling_args.h"
#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {
namespace {

struct ServeArgs {
  ModelConfigArgs model;
  EngineArgs engine;
  SamplingArgs sampling;
  std::string host = "127.0.0.1";  // vLLM: --host
  int port = 8000;                 // vLLM: --port

  server::ServeParams Build() const {
    server::ServeParams p;
    p.model = model.Build();
    p.cache = engine.BuildCacheConfig();
    p.scheduler = engine.BuildSchedulerConfig();
    p.execution = engine.BuildExecutionConfig();
    p.default_sampling = sampling.Build();
    ApplyGenerationConfig(&p.default_sampling);
    p.host = host;
    p.port = port;
    p.served_model_name = model.ServedName();
    return p;
  }

 private:
  /// \brief vLLM --generation-config semantics: the checkpoint's
  ///        generation_config.json sets server defaults, fields the user set
  ///        explicitly win over it, and --override-generation-config merges
  ///        last regardless.
  void ApplyGenerationConfig(sampling::SamplingParams* defaults) const {
    auto resolved = model.ResolveGenerationConfig();
    if (!resolved.ok()) throw CommandError(resolved.status());
    if (resolved->has_value()) {
      MergeGenerationConfig(**resolved, defaults, sampling.ExplicitFields());
    }
    if (!model.override_generation_config.empty()) {
      auto overrides = GenerationConfig::FromJson(model.override_generation_config);
      if (!overrides.ok()) throw CommandError(overrides.status());
      MergeGenerationConfig(*overrides, defaults);
    }
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
  sub->add_option("--host", args->host, "Host address")->capture_default_str();
  sub->add_option("--port", args->port, "Port number")
      ->capture_default_str()
      ->check(CLI::Range(1, 65535));
  sub->callback([args] { ThrowIfError(server::RunServe(args->Build())); });
}

}  // namespace inferx::cli
