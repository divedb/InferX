#pragma once

#include <CLI/CLI.hpp>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "cli/terminal.h"

namespace inferx::cli {

/// Full command path of `app`: the program name plus the subcommand chain,
/// e.g. "inferx bench latency".
std::string CommandPath(const CLI::App& app);

class Presentation {
 public:
  /// `out_is_terminal` / `err_is_terminal` drive per-stream auto color.
  Presentation(CLI::App& app, std::ostream& out, std::ostream& err, bool out_is_terminal,
               bool err_is_terminal);

  /// Parses `argv` (argv[0] is the program name). Translates CLI11 parse
  /// errors into the project's message style while preserving CLI11's exit
  /// codes; returns the exit code. Business errors (CommandError) propagate
  /// to the caller.
  int Parse(int argc, const char* const argv[]);

  void Error(std::string_view message) const;
  void Warning(std::string_view message) const;
  void Success(std::string_view message) const;

 private:
  /// Installs the shared help formatter on the whole command tree.
  void InstallFormatter();

  /// Indexes every option name in the tree and whether it takes a value,
  /// so the parse prepass can reject `--opt --other-flag` as a missing value.
  void CollectValueOptions(CLI::App* app);

  /// Renders `app`'s help into the out stream; returns the exit code (0).
  int PrintHelp(const CLI::App* app);

  /// Reports a translated parse error; returns `code` as the exit code.
  int ReportParseError(const CLI::ParseError& error, const CLI::App* target,
                       const std::string& unmatched, int code);

  CLI::App& app_;
  std::ostream& out_;
  std::ostream& err_;
  bool out_is_terminal_;
  bool err_is_terminal_;
  std::map<std::string, bool> value_options_;  // name -> takes a value
  // Color is decided once from the environment and each stream's tty state.
  cli::TextStyler out_style_;
  cli::TextStyler err_style_;
};

}  // namespace inferx::cli
