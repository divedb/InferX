#pragma once

#include <CLI/CLI.hpp>
#include <string>
#include <string_view>

#ifndef INFERX_VERSION
#define INFERX_VERSION "0.1.0-dev"
#endif

namespace inferx::cli {

/// Default checkpoint directory pre-filled into model options.
inline constexpr std::string_view kDefaultModelDir = "models/Qwen3-0.6B";

/// \brief Command-line interface for InferX.
class InferxCli {
 public:
  /// \brief Constructs a CLI application with the given description and name.
  ///
  /// \param description The description of the CLI application.
  /// \param name        The name of the CLI application.
  InferxCli(std::string description, std::string name)
      : app_(std::move(description), std::move(name)) {
    // Set the default behavior for subcommands: allow 0 or 1 subcommand to be
    // specified.
    app_.require_subcommand(0, 1);
    // Registered before the commands so subcommands inherit the flag; the
    // console normalizes the friendlier "--help=all" spelling onto it.
    app_.set_help_all_flag("--help-all", "Show all options, including advanced ones");
    RegisterCommands();
  }

  /// \brief Parses argv and runs the selected command. Returns the process
  ///        exit code (parse errors preserve CLI11's codes).
  ///
  /// \param argc The number of arguments in argv.
  /// \param argv The command-line arguments.
  /// \return     The process exit code.
  int Run(int argc, char** argv);

  /// \brief The command tree, for in-process rendering (tests, tooling).
  CLI::App& app() { return app_; }

 private:
  void RegisterCommands();
  int ParseCommandLine(int argc, const char* const argv[]);

  CLI::App app_;
};

}  // namespace inferx::cli
