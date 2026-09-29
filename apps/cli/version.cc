#include <fmt/format.h>

#include <CLI/CLI.hpp>

#include "cli/app.h"
#include "cli/commands.h"

namespace inferx::cli {

void RegisterVersion(CLI::App& root) {
  std::string subcmd_name = "version";
  std::string subcmd_desc = "Display program version information and exit.";
  CLI::App* sub = root.add_subcommand(std::move(subcmd_name), std::move(subcmd_desc));
  sub->callback([] { fmt::print("{}\n", INFERX_VERSION); });
}

}  // namespace inferx::cli
