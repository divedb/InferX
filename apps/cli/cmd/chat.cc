// `inferx chat` — client for the running API server; not implemented yet.
#include <CLI/CLI.hpp>

#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {

void RegisterChat(CLI::App& root) {
  CLI::App* sub =
      root.add_subcommand("chat", "Generate chat completions via the running API server.");
  sub->callback([] { ThrowIfError(UnimplementedError("chat is not implemented yet")); });
}

}  // namespace inferx::cli
