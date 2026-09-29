// `inferx version` — prints the build version and exits.
#include <CLI/CLI.hpp>
#include <iostream>

#include "cli/app.h"
#include "cli/commands.h"

namespace inferx::cli {

void RegisterVersion(CLI::App& root) {
  CLI::App* sub = root.add_subcommand("version", "Display program version information and exit.");
  sub->callback([] { std::cout << INFERX_VERSION << "\n"; });
}

}  // namespace inferx::cli
