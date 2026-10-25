#include <CLI/CLI.hpp>
#include <string>

#include "cli/commands.h"

namespace inferx::cli {

void RegisterHelp(CLI::App& root) {
  std::string subcmd_name = "help";
  std::string subcmd_desc = "Show help for a command.";
  CLI::App* sub = root.add_subcommand(std::move(subcmd_name), std::move(subcmd_desc));
  // The positional words exist so CLI11 accepts any command path; the path
  // is resolved by Console's scan, so the values are discarded.
  sub->add_option_function<std::string>(
         "command", [](const std::string&) {}, "Command path to describe")
      ->expected(0, -1)
      ->type_name("");
  sub->callback([] { throw CLI::CallForHelp(); });
}

}  // namespace inferx::cli
