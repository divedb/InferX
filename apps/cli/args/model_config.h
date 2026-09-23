#pragma once

#include <CLI/CLI.hpp>
#include <cstdint>
#include <optional>
#include <set>
#include <string>

#include "cli/app.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "inferx/core/device.h"
#include "inferx/models/checkpoint_config.h"
#include "inferx/models/model_config.h"
#include "inferx/sampling/sampling_params.h"

namespace inferx::cli {

/// \brief vLLM ModelConfig analogue: which checkpoint to load and how the
///        API presents it.
///
/// The essential vLLM options only — InferX loads local directories (no Hub
/// revision/token/remote-code surface), supports no quantization or
/// multimodal inputs yet, and executes fixed C++ architectures (nothing is
/// ever run from the checkpoint). Capacity knobs (KV blocks, batch limits,
/// CUDA graphs) stay in EngineArgs, vLLM's CacheConfig/SchedulerConfig
/// analogue.
struct ModelConfigArgs {
  std::string model = std::string(kDefaultModelDir);  // vLLM: --model
  /// vLLM: --tokenizer; empty uses the model directory.
  std::string tokenizer;
  std::string device = "cuda";  // vLLM: --device
  std::string dtype = "auto";   // vLLM: --dtype
  /// vLLM: --seed (also seeds dataset samplers); 0 stays the engine default.
  std::int64_t seed = 0;
  /// vLLM: --max-model-len; 0 keeps the checkpoint's context limit.
  std::int64_t max_model_len = 0;
  /// vLLM: --served-model-name (single name; vLLM also accepts a list).
  std::string served_model_name;
  /// vLLM: --generation-config: "auto" (default) loads
  /// generation_config.json from the model directory, "vllm" keeps engine
  /// defaults, any other value is a directory to load it from.
  std::string generation_config = "auto";
  /// vLLM: --override-generation-config: JSON merged over the resolved
  /// generation config, e.g. `{"temperature": 0.5}`.
  std::string override_generation_config;

  /// \brief Binds this group's options to `sub`.
  void AddOptions(CLI::App& sub) {
    CLI::Option_group* g =
        sub.add_option_group("Model", "checkpoint identity and presentation");
    g->add_option("--model", model, "Directory of the checkpoint to load")
        ->capture_default_str();
    g->add_option("--tokenizer", tokenizer,
                  "Directory of the tokenizer to load; defaults to the model "
                  "directory")
        ->capture_default_str();
    g->add_option("--device", device, "Device type")
        ->capture_default_str()
        ->check(CLI::IsMember(std::set<std::string>{"cuda"}));
    g->add_option("--dtype", dtype, "Data type for weights and activations")
        ->capture_default_str()
        ->check(CLI::IsMember(std::set<std::string>{"auto", "bfloat16", "float16", "float32"}));
    g->add_option("--seed", seed, "Random seed for reproducible sampling")
        ->capture_default_str();
    g->add_option("--max-model-len", max_model_len,
                  "Maximum sequence length, prompt plus output; 0 keeps the "
                  "checkpoint limit")
        ->capture_default_str()
        ->check(CLI::Range(std::int64_t{0}, std::int64_t{1} << 30));
    g->add_option("--served-model-name", served_model_name,
                  "The model name used in the API; if not specified, the model "
                  "argument is used")
        ->capture_default_str();
    g->add_option("--generation-config", generation_config,
                  "\"auto\" loads generation_config.json from the model "
                  "directory, \"vllm\" keeps engine defaults, any other value "
                  "is the directory to load it from")
        ->capture_default_str();
    g->add_option("--override-generation-config", override_generation_config,
                  "JSON merged over the resolved generation config, e.g. "
                  "'{\"temperature\": 0.5}'")
        ->capture_default_str();
  }

  /// \brief CLI surface -> core ModelConfig: checkpoint identity for
  ///        ModelRunner::Create and the server.
  ModelConfig Build() const {
    ModelConfig config;
    config.model_dir = model;
    config.tokenizer_dir = TokenizerDir();
    config.device = ParseDevice(device);
    config.dtype = dtype;
    config.seed = seed;
    config.max_model_len = max_model_len;
    return config;
  }

  /// \brief The name the API reports; vLLM falls back to the model argument.
  std::string ServedName() const {
    return served_model_name.empty() ? model : served_model_name;
  }

  /// \brief The directory holding tokenizer.json; defaults to the model
  ///        directory (vLLM: --tokenizer).
  std::string TokenizerDir() const { return tokenizer.empty() ? model : tokenizer; }

  /// \brief Resolves --generation-config into a checkpoint config, or
  ///        nullopt when engine defaults should stand ("vllm", or "auto"
  ///        with no generation_config.json — minimal checkpoints ship
  ///        none). An explicit directory that cannot be read is an error.
  absl::StatusOr<std::optional<GenerationConfig>> ResolveGenerationConfig() const {
    if (generation_config == "vllm") return std::nullopt;
    const std::string& dir = generation_config == "auto" ? model : generation_config;
    auto loaded = GenerationConfig::FromFile(dir + "/generation_config.json");
    if (loaded.ok()) return std::optional<GenerationConfig>(*loaded);
    if (generation_config == "auto" && absl::IsNotFound(loaded.status())) {
      return std::nullopt;
    }
    return loaded.status();
  }

 private:
  static DeviceId ParseDevice(const std::string& device) {
    if (device == "cuda") return DeviceId::Cuda(0);
    throw CLI::ValidationError("--device", "unsupported device: " + device);
  }
};

/// \brief Merges checkpoint generation defaults into `sampling`.
///
/// A field named in `skip` keeps the front end's value — used for fields the
/// user set explicitly on the command line, which must win over the
/// checkpoint file (a choice vLLM never faces: its serve command has no
/// sampling flags). HF top_k <= 0 means "disabled" and maps to 0.
inline void MergeGenerationConfig(const GenerationConfig& gen,
                                  sampling::SamplingParams* sampling,
                                  const std::set<std::string>& skip = {}) {
  if (gen.temperature && !skip.count("temperature")) {
    sampling->temperature = *gen.temperature;
  }
  if (gen.top_p && !skip.count("top-p")) sampling->top_p = *gen.top_p;
  if (gen.min_p) sampling->min_p = *gen.min_p;
  if (gen.top_k) sampling->top_k = *gen.top_k > 0 ? static_cast<std::uint32_t>(*gen.top_k) : 0;
  if (gen.repetition_penalty) {
    sampling->repetition_penalty = *gen.repetition_penalty;
  }
  if (gen.max_new_tokens && !skip.count("max-tokens") && *gen.max_new_tokens > 0) {
    sampling->max_tokens = static_cast<std::uint32_t>(*gen.max_new_tokens);
  }
}

}  // namespace inferx::cli
