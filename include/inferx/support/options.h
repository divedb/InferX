#pragma once

#include <optional>

#include "inferx/engine/engine.h"
#include "inferx/server/http_server.h"
#include "inferx/support/log.h"

namespace inferx {

struct ServeOptions {
  engine::EngineConfig engine;
  server::HttpServerConfig http;
  LogOptions log;
};

// Parses the command line into ServeOptions, applying validation and the
// derived defaults (nccl for TP=2, the batch-token budget). Returns nullopt
// after printing help or a parse error, with *exit_code set to what main
// should return (0 for --help, 2 for a bad command line).
std::optional<ServeOptions> ParseServeOptions(int argc, char** argv,
                                              int* exit_code);

}  // namespace inferx
