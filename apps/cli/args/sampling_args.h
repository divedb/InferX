#pragma once

#include <CLI/CLI.hpp>
#include <map>
#include <set>
#include <string>

#include "inferx/sampling/sampling_params.h"

namespace inferx::cli {

struct SamplingArgs {
  inferx::sampling::SamplingParams params;

  /// \brief Names of options passed on the command line.
  ///
  /// The generation-config merge must not clobber values the user asked for
  /// explicitly; vLLM never faces this (its serve command has no sampling
  /// flags, only --override-generation-config).
  std::set<std::string> ExplicitFields() const {
    std::set<std::string> explicit_fields;
    for (const auto& [name, option] : options_) {
      if (option->count() > 0) explicit_fields.insert(name);
    }
    return explicit_fields;
  }

  void AddOptions(CLI::App& sub) {
    CLI::Option_group* g = sub.add_option_group("Sampling", "generation defaults");

    options_["temperature"] =
        g->add_option("--temperature", params.temperature, "Sampling temperature")
            ->capture_default_str()
            ->check(CLI::Range(0.0f, 10.0f));
    options_["top-p"] = g->add_option("--top-p", params.top_p, "Top-p sampling")
                            ->capture_default_str()
                            ->check(CLI::Range(0.0f, 1.0f));
    options_["max-tokens"] =
        g->add_option("--max-tokens", params.max_tokens, "Maximum tokens per request")
            ->capture_default_str()
            ->check(CLI::PositiveNumber);
    g->add_flag("--ignore-eos", params.ignore_eos, "Never stop on EOS");
  }

  const inferx::sampling::SamplingParams& Build() const { return params; }

 private:
  /// Option pointers are owned by the App, which outlives the args struct
  /// (commands keep both alive until the callback completes).
  std::map<std::string, CLI::Option*> options_;
};

}  // namespace inferx::cli
