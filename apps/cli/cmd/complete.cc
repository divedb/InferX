// `inferx complete` — client for the running API server; not implemented yet.
#include <CLI/CLI.hpp>

#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {

void RegisterComplete(CLI::App& root) {
  CLI::App* sub = root.add_subcommand(
      "complete", "Generate text completions based on the given prompt via the running API server.");
  sub->callback([] { ThrowIfError(UnimplementedError("complete is not implemented yet")); });
}

}  // namespace inferx::cli
