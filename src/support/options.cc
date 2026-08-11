#include "inferx/support/options.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "CLI/CLI.hpp"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "inferx/support/vllm_compat.h"

namespace inferx {

namespace {

// Deprecated InferX spellings stay parseable but out of --help; the canonical
// names are vLLM's, so a vLLM launch script ports without translation.
constexpr std::string_view kHiddenGroup = "";
constexpr std::string_view kExtensionsGroup = "InferX extensions";

struct ServeParseState {
  bool enforce_eager = false;
  bool enable_chunked_prefill = true;
  std::string log_level = "info";
  std::string model_positional;
  std::string dtype = "auto";
  std::string kv_cache_dtype = "auto";
  std::string quantization;
  std::vector<std::string> served_model_names;
  std::vector<std::string> api_keys;

  CLI::Option* model = nullptr;
  CLI::Option* comm_backend = nullptr;
  CLI::Option* max_batch_tokens = nullptr;
  CLI::Option* kv_cache_dtype_opt = nullptr;
  CLI::Option* fp8_kv = nullptr;
  CLI::Option* chunked_prefill = nullptr;
  std::vector<std::pair<std::string, CLI::Option*>> deprecated;

  CompatState compat;
};

void AddEngineOptions(CLI::App& app, engine::EngineConfig& config,
                      ServeParseState& state) {
  app.add_option("model_tag", state.model_positional,
                 "checkpoint directory (vLLM positional form)")
      ->check(CLI::ExistingDirectory);
  state.model = app.add_option("--model", config.model_dir,
                               "checkpoint directory")
                    ->check(CLI::ExistingDirectory);
  app.add_option("--served-model-name", state.served_model_names,
                 "name reported in responses");
  app.add_option("--dtype", state.dtype,
                 "model dtype (InferX runs bfloat16 only)")
      ->check(CLI::IsMember(
          {"auto", "half", "float16", "bfloat16", "float", "float32"}));

  CLI::Option* max_num_seqs =
      app.add_option("--max-num-seqs", config.scheduler.max_running,
                     "concurrent sequences")
          ->check(CLI::PositiveNumber);
  state.deprecated.emplace_back(
      "--max-running",
      app.add_option("--max-running", config.scheduler.max_running,
                     "deprecated spelling of --max-num-seqs")
          ->check(CLI::PositiveNumber)
          ->group(std::string(kHiddenGroup))
          ->excludes(max_num_seqs));

  CLI::Option* max_model_len =
      app.add_option("--max-model-len", config.scheduler.max_seq_len,
                     "prompt + generation cap")
          ->check(CLI::PositiveNumber);
  state.deprecated.emplace_back(
      "--max-seq-len",
      app.add_option("--max-seq-len", config.scheduler.max_seq_len,
                     "deprecated spelling of --max-model-len")
          ->check(CLI::PositiveNumber)
          ->group(std::string(kHiddenGroup))
          ->excludes(max_model_len));

  state.max_batch_tokens =
      app.add_option("--max-num-batched-tokens",
                     config.scheduler.max_batch_tokens,
                     "token budget per scheduler step")
          ->check(CLI::PositiveNumber);

  app.add_flag("--enable-prefix-caching,!--no-enable-prefix-caching",
               config.scheduler.enable_prefix_cache,
               "reuse prompt-prefix KV across requests");
  state.chunked_prefill =
      app.add_flag("--enable-chunked-prefill,!--no-enable-chunked-prefill",
                   state.enable_chunked_prefill,
                   "chunked prefill (always on; cannot be disabled)");

  CLI::Option* blocks_override =
      app.add_option("--num-gpu-blocks-override", config.kv_blocks,
                     "KV cache blocks (overrides --gpu-memory-utilization)")
          ->check(CLI::PositiveNumber);
  state.deprecated.emplace_back(
      "--kv-blocks",
      app.add_option("--kv-blocks", config.kv_blocks,
                     "deprecated spelling of --num-gpu-blocks-override")
          ->check(CLI::PositiveNumber)
          ->group(std::string(kHiddenGroup))
          ->excludes(blocks_override));

  app.add_option("--block-size", config.block_size, "tokens per block")
      ->check(CLI::PositiveNumber);
  app.add_option("--gpu-memory-utilization", config.gpu_memory_utilization,
                 "fraction of GPU memory the engine may use")
      ->check(CLI::Range(0.0, 1.0));
  app.add_option("--kv-cache-memory-bytes", config.kv_cache_memory_bytes,
                 "explicit KV cache budget in bytes")
      ->check(CLI::PositiveNumber);

  app.add_option("--tensor-parallel-size", config.tensor_parallel_size,
                 "tensor-parallel ranks")
      ->check(CLI::Range(1, 2));
  CLI::Option* device_ids =
      app.add_option("--device-ids", config.devices,
                     "comma-separated CUDA devices")
          ->delimiter(',')
          ->check(CLI::NonNegativeNumber);
  state.deprecated.emplace_back(
      "--devices", app.add_option("--devices", config.devices,
                                  "deprecated spelling of --device-ids")
                       ->delimiter(',')
                       ->check(CLI::NonNegativeNumber)
                       ->group(std::string(kHiddenGroup))
                       ->excludes(device_ids));

  state.comm_backend =
      app.add_option("--comm-backend", config.comm_backend, "single or nccl")
          ->check(CLI::IsMember({"single", "nccl"}))
          ->group(std::string(kExtensionsGroup));
  app.add_option("--collective-timing-sample-rate",
                 config.collective_timing_sample_every,
                 "sample every Nth collective (0 disables)")
      ->group(std::string(kExtensionsGroup));

  app.add_option("--quantization,-q", state.quantization,
                 "weight quantization: fp8, or the InferX extension w4a16");
  CLI::Option* fp8 = app.add_flag("--fp8", config.fp8_weights,
                                  "deprecated spelling of --quantization fp8")
                         ->group(std::string(kHiddenGroup));
  state.deprecated.emplace_back("--fp8", fp8);
  state.deprecated.emplace_back(
      "--w4a16", app.add_flag("--w4a16", config.int4_weights,
                              "deprecated spelling of --quantization w4a16")
                     ->group(std::string(kHiddenGroup))
                     ->excludes(fp8));

  state.kv_cache_dtype_opt =
      app.add_option("--kv-cache-dtype", state.kv_cache_dtype,
                     "auto, bfloat16, fp8, or fp8_e4m3");
  state.fp8_kv = app.add_flag("--fp8-kv", config.fp8_kv_cache,
                              "deprecated spelling of --kv-cache-dtype fp8")
                     ->group(std::string(kHiddenGroup));
  state.deprecated.emplace_back("--fp8-kv", state.fp8_kv);

  app.add_flag("--enforce-eager,!--no-enforce-eager", state.enforce_eager,
               "skip decode graph capture at startup");
  state.deprecated.emplace_back(
      "--no-cuda-graphs",
      app.add_flag("--no-cuda-graphs", state.enforce_eager,
                   "deprecated spelling of --enforce-eager")
          ->group(std::string(kHiddenGroup)));
}

void AddHttpOptions(CLI::App& app, server::HttpServerConfig& config,
                    ServeParseState& state) {
  app.add_option("--host", config.host, "bind address");
  app.add_option("--port", config.port, "bind port")
      ->check(CLI::Range(1, 65535));
  app.add_option("--api-key", state.api_keys,
                 "accepted bearer token (repeatable; stored hashed)");
  app.add_option("--api-key-sha256", config.api_key_sha256,
                 "accepted bearer-token SHA-256 hash (repeatable)")
      ->group(std::string(kExtensionsGroup));
  app.add_option("--scheduler-endpoint", config.scheduler_endpoint,
                 "use a process-separated gRPC scheduler")
      ->group(std::string(kExtensionsGroup));

  app.add_option("--max-active-requests", config.max_active_requests,
                 "admission cap across all in-flight requests")
      ->check(CLI::PositiveNumber)
      ->group(std::string(kExtensionsGroup));
  app.add_option("--max-request-bytes", config.max_request_bytes,
                 "largest accepted request body")
      ->check(CLI::PositiveNumber)
      ->group(std::string(kExtensionsGroup));
  app.add_option("--read-timeout", config.read_timeout_seconds,
                 "socket read timeout, seconds")
      ->check(CLI::PositiveNumber)
      ->group(std::string(kExtensionsGroup));
  app.add_option("--write-timeout", config.write_timeout_seconds,
                 "socket write timeout, seconds")
      ->check(CLI::PositiveNumber)
      ->group(std::string(kExtensionsGroup));
  app.add_option("--request-timeout", config.request_timeout_seconds,
                 "generation deadline after submission, seconds")
      ->check(CLI::PositiveNumber)
      ->group(std::string(kExtensionsGroup));
  app.add_option("--io-threads", config.io_threads,
                 "socket event-loop threads (0 = auto)")
      ->group(std::string(kExtensionsGroup));
  app.add_option("--application-threads", config.application_threads,
                 "application executor threads (0 = auto)")
      ->group(std::string(kExtensionsGroup));
}

void AddLogOptions(CLI::App& app, LogOptions& options, ServeParseState& state) {
  app.add_option("--log-level", state.log_level,
                 "debug, info, warning, or error")
      ->check(CLI::IsMember({"debug", "info", "warning", "error"}));
  app.add_option("--v", options.verbosity,
                 "enable VLOG messages through level n")
      ->check(CLI::NonNegativeNumber);
  app.add_flag("--log-json", options.json, "write JSON-lines logs to stderr");
  app.add_option("--log-file", options.file, "append logs to a file");
}

Status FinalizeOptions(ServeOptions& options, const ServeParseState& state) {
  for (const auto& [name, option] : state.deprecated) {
    if (option->count() > 0) {
      LOG(WARNING) << name << " is deprecated; see --help for the vLLM name";
    }
  }

  // vLLM's positional MODEL and --model are the same option; InferX requires
  // exactly one of them.
  if (!state.model_positional.empty()) {
    if (state.model->count() > 0 &&
        options.engine.model_dir != state.model_positional) {
      return InvalidArgumentError(
          "the MODEL positional and --model disagree; pass one of them");
    }
    options.engine.model_dir = state.model_positional;
  }
  if (options.engine.model_dir.empty()) {
    return InvalidArgumentError(
        "a checkpoint directory is required (positional MODEL or --model)");
  }

  if (!state.served_model_names.empty()) {
    if (state.served_model_names.size() > 1) {
      return InvalidArgumentError(
          "multiple --served-model-name aliases are not supported by InferX");
    }
    options.engine.served_model_name = state.served_model_names.front();
  }

  if (state.dtype != "auto" && state.dtype != "bfloat16") {
    return InvalidArgumentError(
        absl::StrCat("--dtype ", state.dtype,
                     " is not supported: InferX runs bfloat16 only (use "
                     "'auto' or 'bfloat16')"));
  }

  if (!state.quantization.empty()) {
    if (state.quantization == "fp8") {
      options.engine.fp8_weights = true;
    } else if (state.quantization == "w4a16") {
      options.engine.int4_weights = true;
    } else {
      return InvalidArgumentError(absl::StrCat(
          "--quantization ", state.quantization,
          " is not supported by InferX (supported: fp8, w4a16)"));
    }
  }
  if (options.engine.fp8_weights && options.engine.int4_weights) {
    return InvalidArgumentError(
        "fp8 and w4a16 weight quantization are mutually exclusive");
  }

  if (state.kv_cache_dtype_opt->count() > 0) {
    const std::string& dtype = state.kv_cache_dtype;
    const bool wants_fp8 = dtype == "fp8" || dtype == "fp8_e4m3";
    if (!wants_fp8 && dtype != "auto" && dtype != "bfloat16") {
      return InvalidArgumentError(absl::StrCat(
          "--kv-cache-dtype ", dtype,
          " is not supported by InferX (supported: auto, bfloat16, fp8, "
          "fp8_e4m3)"));
    }
    if (state.fp8_kv->count() > 0 && !wants_fp8) {
      return InvalidArgumentError(
          "--kv-cache-dtype and --fp8-kv disagree; pass one of them");
    }
    options.engine.fp8_kv_cache = wants_fp8;
  }

  if (state.chunked_prefill->count() > 0 && !state.enable_chunked_prefill) {
    return InvalidArgumentError(
        "chunked prefill cannot be disabled in InferX");
  }

  if (options.engine.gpu_memory_utilization <= 0.0) {
    return InvalidArgumentError("--gpu-memory-utilization must be positive");
  }

  options.engine.capture_graphs = !state.enforce_eager;

  if (state.log_level == "debug") {
    options.log.min_level = "info";
    options.log.verbosity = std::max(options.log.verbosity, 1);
  } else {
    options.log.min_level = state.log_level;
  }

  if (state.comm_backend->count() == 0 &&
      options.engine.tensor_parallel_size == 2) {
    options.engine.comm_backend = "nccl";
  }

  // Unset, the batch is bounded by how many sequences can be resident, so the
  // token budget follows from the sequence cap. Set, it is honored verbatim:
  // a budget below --max-model-len is legal because chunked prefill splits
  // prompts across steps.
  if (state.max_batch_tokens->count() == 0) {
    options.engine.scheduler.max_batch_tokens =
        std::max<int64_t>(options.engine.scheduler.max_batch_tokens,
                          options.engine.scheduler.max_seq_len);
  }

  for (const std::string& key : state.api_keys) {
    std::string hash = Sha256Hex(key);
    if (hash.empty()) return InternalError("failed to hash --api-key");
    options.http.api_key_sha256.push_back(std::move(hash));
  }

  return OkStatus();
}

}  // namespace

std::optional<ServeOptions> ParseServeOptions(int argc, char** argv,
                                              int* exit_code) {
  ServeOptions options;
  ServeParseState state;

  CLI::App app{
      "inferx-serve -- the OpenAI-compatible inference server.\n"
      "\n"
      "Flags follow vLLM's names. Flags for features InferX does not\n"
      "implement parse, but any value other than the vLLM default is a\n"
      "startup error naming the missing feature."};
  app.option_defaults()->always_capture_default();

  AddEngineOptions(app, options.engine, state);
  AddHttpOptions(app, options.http, state);
  AddLogOptions(app, options.log, state);
  AddCompatStubs(app, ServeCompatStubs(), state.compat);

  app.set_config("--config", "",
                 "TOML/INI config file (vLLM YAML configs are not supported)")
      ->check([](const std::string& path) -> std::string {
        if (path.ends_with(".yaml") || path.ends_with(".yml")) {
          return "YAML configs are not supported by InferX; provide TOML/INI";
        }
        return {};
      });

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    *exit_code = app.exit(e) == 0 ? 0 : 2;
    return std::nullopt;
  }

  Status status = FinalizeOptions(options, state);
  if (status.ok()) status = CheckCompatStubs(ServeCompatStubs(), state.compat);
  if (!status.ok()) {
    std::cerr << "inferx-serve: " << status.message() << std::endl;
    *exit_code = 2;
    return std::nullopt;
  }

  return options;
}

}  // namespace inferx
