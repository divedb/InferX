// Results-and-logging options shared by every `inferx bench` command
// (vLLM bench: --output-json, --disable-log-stats). Plain data; commands
// copy it into their params structs alongside their own knobs.
#pragma once
#include <CLI/CLI.hpp>
#include <string>

namespace inferx::cli {

struct BenchOutputArgs {
  std::string output_json;         // vLLM bench: --output-json
  bool disable_log_stats = false;  // vLLM bench: --disable-log-stats

  void AddOptions(CLI::App& sub) {
    CLI::Option_group* g = sub.add_option_group("Output", "results and logging");
    g->add_option("--output-json", output_json,
                  "Path to save benchmark results in JSON format");
    g->add_flag("--disable-log-stats", disable_log_stats,
                "Disable logging statistics");
  }
};

}  // namespace inferx::cli
