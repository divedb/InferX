/// inferx-gateway -- the process-separated OpenAI-compatible gateway.

#include <memory>
#include <optional>

#include "inferx/server/gateway/gateway_server.h"
#include "inferx/support/crash.h"
#include "inferx/support/log.h"
#include "inferx/support/shutdown.h"
#include "options.h"

int main(int argc, char** argv) {
  inferx::InstallCrashHandler(argv[0]);

  int exit_code = 0;
  std::optional<inferx::server::gateway::GatewayOptions> options =
      inferx::server::gateway::ParseGatewayOptions(argc, argv, &exit_code);
  if (!options.has_value()) return exit_code;

  inferx::InitLogging(options->log);

  // Must precede GatewayServer::Create so every worker thread inherits the
  // mask and shutdown signals reach only the waiter thread below.
  inferx::BlockShutdownSignals();

  inferx::StatusOr<
      std::unique_ptr<inferx::server::gateway::GatewayServer>>
      gateway =
          inferx::server::gateway::GatewayServer::Create(options->config);
  if (!gateway.ok()) {
    LOG(ERROR) << "gateway creation failed: " << gateway.status();
    return 1;
  }

  LOG(INFO) << "inferx-gateway listening on http://" << options->config.host
            << ':' << options->config.port;

  inferx::server::gateway::GatewayServer* listener = gateway->get();
  inferx::ShutdownSignalWaiter waiter([listener](int signum) {
    LOG(INFO) << "caught signal " << signum << ", stopping";
    listener->Stop();
  });

  const inferx::Status status = listener->Listen();

  if (!status.ok()) {
    LOG(ERROR) << "gateway listen failed: " << status;
    return 1;
  }

  LOG(INFO) << "shutting down";
  return 0;
}
