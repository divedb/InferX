#include <memory>
#include <optional>

#include "inferx/engine/engine.h"
#include "inferx/server/http_server.h"
#include "inferx/support/crash.h"
#include "inferx/support/log.h"
#include "inferx/support/options.h"
#include "inferx/support/shutdown.h"

int main(int argc, char** argv) {
  inferx::InstallCrashHandler(argv[0]);

  int exit_code = 0;
  auto options = inferx::ParseServeOptions(argc, argv, &exit_code);

  if (!options.has_value()) return exit_code;

  inferx::InitLogging(options->log);

  // Must precede Engine::Create so every worker thread inherits the mask and
  // shutdown signals are delivered only to the waiter thread below.
  inferx::BlockShutdownSignals();

  LOG(INFO) << "loading model from " << options->engine.model_dir;

  inferx::StatusOr<std::unique_ptr<inferx::engine::Engine>> engine =
      inferx::engine::Engine::Create(options->engine);
  if (!engine.ok()) {
    LOG(ERROR) << "model load failed: " << engine.status();
    return 1;
  }

  inferx::StatusOr<std::unique_ptr<inferx::server::HttpServer>> server =
      inferx::server::HttpServer::Create(engine->get(), options->http);
  if (!server.ok()) {
    LOG(ERROR) << "server creation failed: " << server.status();
    return 1;
  }

  const inferx::engine::EngineConfig& engine_config = options->engine;
  LOG(INFO) << "inferx-serve listening on http://" << options->http.host << ':'
            << options->http.port << " model=" << (*engine)->model_name()
            << " weights="
            << (engine_config.fp8_weights    ? "fp8-e4m3"
                : engine_config.int4_weights ? "w4a16"
                                             : "bf16")
            << " cuda_graphs=" << (engine_config.capture_graphs ? "on" : "off")
            << " max_running=" << engine_config.scheduler.max_running
            << " max_seq_len=" << engine_config.scheduler.max_seq_len
            << " kv_blocks=" << engine_config.kv_blocks
            << " block_size=" << engine_config.block_size;

  inferx::server::HttpServer* listener = server->get();
  inferx::ShutdownSignalWaiter waiter([listener](int signum) {
    LOG(INFO) << "caught signal " << signum << ", stopping";
    listener->Stop();
  });

  const inferx::Status listened = listener->Listen();

  if (!listened.ok()) {
    LOG(ERROR) << "listen failed: " << listened;
    return 1;
  }

  LOG(INFO) << "shutting down";

  return 0;
}
