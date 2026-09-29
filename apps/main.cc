#include <cstdio>

#include "cli/app.h"
#include "inferx/core/logging.h"

int main(int argc, char** argv) {
  // Thresholds from INFERX_LOG_LEVEL / INFERX_VLOG; the env-derived config
  // is always valid, so this only fails on direct misuse.
  if (!inferx::core::InitializeLogging(inferx::core::LoggingConfigFromEnv()).ok()) {
    std::fprintf(stderr, "inferx: logging left at Abseil defaults\n");
  }

  inferx::cli::InferxCli cli{"InferX — High-performance LLM inference engine", "inferx"};

  return cli.Run(argc, argv);
}
