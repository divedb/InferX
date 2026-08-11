#pragma once

#include <optional>

#include "inferx/server/gateway/gateway_server.h"
#include "inferx/support/log.h"

namespace inferx::server::gateway {

struct GatewayOptions {
  GatewayServerConfig config;
  LogOptions log;
};

// Parses the command line into GatewayOptions. Returns nullopt after printing
// help or a parse error, with *exit_code set to what main should return
// (0 for --help, 2 for a bad command line).
std::optional<GatewayOptions> ParseGatewayOptions(int argc, char** argv,
                                                  int* exit_code);

}  // namespace inferx::server::gateway
