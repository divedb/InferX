// Prompt-source options shared by `bench latency` / `bench throughput`
// (vLLM: --dataset-name, --dataset-path, and the random dataset's lens).
// Conversion turns the stringly CLI surface into the typed DatasetParams.
#pragma once
#include <CLI/CLI.hpp>
#include <optional>
#include <string>

#include "inferx/bench/dataset.h"

namespace inferx::cli {

struct DatasetArgs {
  std::string name = "random";      // vLLM: --dataset-name
  std::optional<std::string> path;  // vLLM: --dataset-path
  int input_len = 32;               // vLLM random dataset: --input-len
  int output_len = 128;             // vLLM random dataset: --output-len
  // vLLM registers one --seed on the model config and feeds the dataset
  // sampler from it; commands carrying both groups forward ModelConfigArgs'
  // seed into Build()'s output.
  int seed = 0;

  void AddOptions(CLI::App& sub) {
    CLI::Option_group* g = sub.add_option_group("Dataset", "prompt source");
    g->add_option("--dataset-name", name, "Dataset name")
        ->capture_default_str()
        ->check(CLI::IsMember(bench::DatasetNameValues()));
    g->add_option("--dataset-path", path, "Path to the dataset");
    g->add_option("--input-len", input_len,
                  "Input length of prompts for the random dataset")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
    g->add_option("--output-len", output_len,
                  "Output length of responses for the random dataset")
        ->capture_default_str()
        ->check(CLI::PositiveNumber);
  }

  /// Converts to the runtime form. --dataset-name was validated by
  /// IsMember, so the lookup cannot fail.
  bench::DatasetParams Build() const {
    bench::DatasetParams p;
    StatusOr<bench::DatasetName> parsed = bench::ParseDatasetName(name);
    if (!parsed.ok())
      throw CLI::ValidationError("--dataset-name", std::string(parsed.status().message()));
    p.name = *parsed;
    p.path = path.value_or("");
    p.input_len = input_len;
    p.output_len = output_len;
    p.seed = seed;
    return p;
  }
};

}  // namespace inferx::cli
