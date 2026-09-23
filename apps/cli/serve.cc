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
  int tokenizer_workers = 2;
  int tokenizer_queue_max_requests = 128;
  int tokenizer_queue_max_mb = 32;
  int tokenizer_request_timeout_ms = 10000;
  int tokenizer_startup_timeout_ms = 30000;

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
    p.tokenizer.workers = tokenizer_workers;
    p.tokenizer.queue_max_requests =
        static_cast<std::size_t>(tokenizer_queue_max_requests);
    p.tokenizer.queue_max_bytes =
        static_cast<std::size_t>(tokenizer_queue_max_mb) << 20;
    p.tokenizer.request_timeout =
        std::chrono::milliseconds(tokenizer_request_timeout_ms);
    p.tokenizer.startup_timeout =
        std::chrono::milliseconds(tokenizer_startup_timeout_ms);
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
  // Prompt-preparation worker pool (docs/tokenizer_process_pool.md): the
  // exposed surface is worker count, queue bounds, and deadlines.
  sub->add_option("--tokenizer-workers", args->tokenizer_workers,
                  "CPU tokenizer worker processes (no inline fallback)")
      ->capture_default_str()
      ->check(CLI::Range(1, 1024));
  sub->add_option("--tokenizer-queue-max-requests",
                  args->tokenizer_queue_max_requests,
                  "Max pending+running tokenizer requests")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--tokenizer-queue-max-mb", args->tokenizer_queue_max_mb,
                  "Max pending tokenizer input in MiB")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--tokenizer-request-timeout-ms",
                  args->tokenizer_request_timeout_ms,
                  "Per-request prompt preprocessing budget incl. queueing")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--tokenizer-startup-timeout-ms",
                  args->tokenizer_startup_timeout_ms,
                  "Worker startup handshake timeout")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->callback([args] { ThrowIfError(server::RunServe(args->Build())); });
}

}  // namespace inferx::cli
