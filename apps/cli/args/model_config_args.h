#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "cli/app.h"
#include "cli/error.h"
#include "inferx/config/model_config.h"
#include "inferx/core/device.h"

namespace inferx::cli {

/// \brief CLI binding over ModelConfig (vLLM ModelConfig analogue).
///
/// Options bind straight into the embedded engine config -- adding a
/// ModelConfig field means adding one option binding, not a second storage
/// declaration. Two deliberate deviations: the --device choice set is
/// compile-time (a build without the CUDA runtime does not offer "cuda"),
/// and --device-ids stays a CLI string because comma-list parsing with
/// CLI-shaped errors belongs here. The constructor applies the CLI's
/// convenience defaults so ModelConfig{} stays neutral for engine-side
/// constructors.
struct ModelConfigArgs {
  ModelConfig config;       ///< The engine values; options bind into it.
  std::string device_ids;   ///< vLLM/CUDA ordinals, comma-separated.

  ModelConfigArgs() { config.model_dir = std::string(kDefaultModelDir); }

  /// \brief Binds this group's options to `sub`. Deep-tuning and rarely used
  /// identity overrides sit in the "Advanced" group (shown by --help=all).
  void AddOptions(CLI::App& sub) {
    CLI::Option_group* g =
        sub.add_option_group("Model", "checkpoint identity and presentation");
    g->add_option("--model", config.model_dir, "Directory of the checkpoint to load")
        ->type_name("PATH")
        ->capture_default_str();
    g->add_option("--tokenizer", config.tokenizer_dir,
                  "Directory of the tokenizer to load; defaults to the model directory")
        ->type_name("PATH")
        ->capture_default_str();
    // The choice set is the compile-time capability: a build without the
    // CUDA runtime cannot honor "cuda", so it is not offered.
    std::set<std::string> device_types = {"cpu"};
    if (kCudaBuilt) device_types.insert("cuda");
    g->add_option("--device", config.device.device_type, "Device type")
        ->type_name("TYPE")
        ->capture_default_str()
        ->check(CLI::IsMember(device_types));
    g->add_option("--device-ids", device_ids,
                  "Device ordinals to execute on, comma-separated (logical ids "
                  "after CUDA_VISIBLE_DEVICES); defaults to device 0")
        ->type_name("IDS")
        ->capture_default_str();
    g->add_option("--dtype", config.dtype, "Data type for weights and activations")
        ->type_name("TYPE")
        ->capture_default_str()
        ->check(CLI::IsMember(std::set<std::string>{"auto", "bfloat16", "float16", "float32"}));
    g->add_option("--seed", config.seed, "Random seed for reproducible sampling")
        ->type_name("N")
        ->group("Advanced")
        ->capture_default_str();
    g->add_option("--max-model-len", config.max_model_len,
                  "Maximum sequence length, prompt plus output; 0 keeps the "
                  "checkpoint limit")
        ->type_name("N")
        ->group("Advanced")
        ->capture_default_str()
        ->check(CLI::Range(std::int64_t{0}, std::int64_t{1} << 30));
    g->add_option("--served-model-name", config.served_model_name,
                  "The model name used in the API; if not specified, the model "
                  "argument is used")
        ->type_name("NAME")
        ->group("Advanced")
        ->capture_default_str();
    g->add_option("--generation-config", config.generation_config,
                  "\"auto\" loads generation_config.json from the model "
                  "directory, \"vllm\" keeps engine defaults, any other value "
                  "is the directory to load it from")
        ->type_name("TEXT")
        ->group("Advanced")
        ->capture_default_str();
    g->add_option("--override-generation-config", config.override_generation_config,
                  "JSON merged over the resolved generation config, e.g. "
                  "'{\"temperature\": 0.5}'")
        ->type_name("JSON")
        ->group("Advanced")
        ->capture_default_str();
  }

  /// \brief CLI surface -> core ModelConfig: checkpoint identity for
  ///        ModelRunner::Create and the server.
  ModelConfig Build() const {
    ModelConfig built = config;
    built.device.device_ids = ParseDeviceIds(device_ids);
    if (auto valid = built.device.Validate(); !valid.ok()) {
      throw CommandError(valid);
    }
    return built;
  }

 private:
  /// \brief Parses "0,1" into {0, 1}; empty input means "default".
  static std::vector<int> ParseDeviceIds(const std::string& text) {
    std::vector<int> ids;
    if (text.empty()) return ids;
    std::string token;
    const auto flush = [&] {
      ids.push_back(ParseDeviceId(token));
      token.clear();
    };
    for (const char c : text) {
      if (c == ',') {
        flush();
      } else {
        token.push_back(c);
      }
    }
    if (!token.empty()) flush();
    return ids;
  }

  static int ParseDeviceId(const std::string& token) {
    if (token.empty() || token.find_first_not_of("0123456789") != std::string::npos) {
      throw CommandError(InvalidArgumentError(
          "--device-ids expects comma-separated non-negative integers, got '", token, "'"));
    }
    return std::stoi(token);
  }
};

}  // namespace inferx::cli
