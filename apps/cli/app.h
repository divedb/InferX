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

class InferxCli {
 public:
  /// \brief
  ///
  /// \param description The description of the CLI application.
  /// \param name        The name of the CLI application.
  InferxCli(std::string description, std::string name)
      : app_(std::move(description), std::move(name)) {
    // Set the default behavior for subcommands: allow 0 or 1 subcommand to be
    // specified.
    app_.require_subcommand(0, 1);
    app_.set_version_flag("-v,--version", INFERX_VERSION);
    RegisterCommands();
  }

  /// \brief Parses argv and runs the selected command. Returns the process
  /// exit code (parse errors preserve CLI11's codes).
  int Run(int argc, char** argv);

 private:
  /// \brief Register all subcommands and their options into the CLI application.
  void RegisterCommands();

  CLI::App app_;
};

}  // namespace inferx::cli
