#include <CLI/CLI.hpp>

#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {

void RegisterRunBatch(CLI::App& root) {
  std::string subcmd_name = "run-batch";
  std::string subcmd_description = "Run batch prompts and write results to file.";
  CLI::App* sub = root.add_subcommand(std::move(subcmd_name), std::move(subcmd_description));

  sub->callback([] { ThrowIfError(UnimplementedError("run-batch is not implemented yet")); });
}

}  // namespace inferx::cli
