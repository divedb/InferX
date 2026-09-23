// `inferx run-batch` — offline batch runner; not implemented yet.
#include <CLI/CLI.hpp>

#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {

void RegisterRunBatch(CLI::App& root) {
  CLI::App* sub = root.add_subcommand("run-batch", "Run batch prompts and write results to file.");
  sub->callback([] { ThrowIfError(UnimplementedError("run-batch is not implemented yet")); });
}

}  // namespace inferx::cli
