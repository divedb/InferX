// `inferx collect-env` — environment report; not implemented yet.
#include <CLI/CLI.hpp>

#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {

void RegisterCollectEnv(CLI::App& root) {
  CLI::App* sub = root.add_subcommand("collect-env", "Start collecting environment information.");
  sub->callback(
      [] { ThrowIfError(UnimplementedError("collect-env is not implemented yet")); });
}

}  // namespace inferx::cli
