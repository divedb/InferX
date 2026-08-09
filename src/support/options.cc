#include "inferx/support/options.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include "CLI/CLI.hpp"

namespace inferx {

std::optional<ServeOptions> ParseServeOptions(int argc, char** argv,
                                              int* exit_code) {
  ServeOptions options;
  bool no_cuda_graphs = false;
  std::string log_level = "info";

  CLI::App app{
      "inferx-serve -- the OpenAI-compatible inference server.\n"
      "\n"
      "Only greedy decoding is implemented: temperature and top_p are\n"
      "accepted and ignored, and responses report system_fingerprint "
      "\"greedy\"."};
  app.option_defaults()->always_capture_default();

  app.add_option("--model", options.engine.model_dir,
                 "checkpoint directory (required)")
      ->required()
      ->check(CLI::ExistingDirectory);
  app.add_option("--host", options.http.host, "bind address");
  app.add_option("--port", options.http.port, "bind port")
      ->check(CLI::Range(1, 65535));
  app.add_option("--scheduler-endpoint", options.http.scheduler_endpoint,
                 "use a process-separated gRPC scheduler");
  app.add_option("--served-model-name", options.engine.served_model_name,
                 "name reported in responses");
  app.add_option("--api-key-sha256", options.http.api_key_sha256,
                 "accepted bearer-token hash (repeatable)");
  app.add_option("--max-running", options.engine.scheduler.max_running,
                 "concurrent sequences")
      ->check(CLI::PositiveNumber);
  app.add_option("--max-seq-len", options.engine.scheduler.max_seq_len,
                 "prompt + generation cap")
      ->check(CLI::PositiveNumber);
  app.add_option("--kv-blocks", options.engine.kv_blocks, "KV cache blocks")
      ->check(CLI::PositiveNumber);
  app.add_option("--block-size", options.engine.block_size, "tokens per block")
      ->check(CLI::PositiveNumber);
  app.add_option("--tensor-parallel-size",
                 options.engine.tensor_parallel_size, "tensor-parallel ranks")
      ->check(CLI::Range(1, 2));
  app.add_option("--devices", options.engine.devices,
                 "comma-separated CUDA devices")
      ->delimiter(',')
      ->check(CLI::NonNegativeNumber);
  CLI::Option* comm_backend =
      app.add_option("--comm-backend", options.engine.comm_backend,
                     "single or nccl")
          ->check(CLI::IsMember({"single", "nccl"}));
  app.add_option("--collective-timing-sample-rate",
                 options.engine.collective_timing_sample_every,
                 "sample every Nth collective (0 disables)");
  CLI::Option* fp8 = app.add_flag("--fp8", options.engine.fp8_weights,
                                  "quantize weights to FP8 e4m3");
  app.add_flag("--w4a16", options.engine.int4_weights,
               "quantize projection weights to grouped int4")
      ->excludes(fp8);
  app.add_flag("--fp8-kv", options.engine.fp8_kv_cache,
               "store the KV cache as FP8 e4m3");
  app.add_flag("--no-cuda-graphs", no_cuda_graphs,
               "skip decode graph capture at startup");
  app.add_option("--log-level", log_level, "debug, info, warning, or error")
      ->check(CLI::IsMember({"debug", "info", "warning", "error"}));
  app.add_option("--v", options.log.verbosity,
                 "enable VLOG messages through level n")
      ->check(CLI::NonNegativeNumber);
  app.add_flag("--log-json", options.log.json,
               "write JSON-lines logs to stderr");
  app.add_option("--log-file", options.log.file, "append logs to a file");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    *exit_code = app.exit(e) == 0 ? 0 : 2;
    return std::nullopt;
  }

  options.engine.capture_graphs = !no_cuda_graphs;

  if (log_level == "debug") {
    options.log.min_level = "info";
    options.log.verbosity = std::max(options.log.verbosity, 1);
  } else {
    options.log.min_level = log_level;
  }

  if (comm_backend->count() == 0 &&
      options.engine.tensor_parallel_size == 2) {
    options.engine.comm_backend = "nccl";
  }

  // The batch is bounded by how many sequences can be resident, so the token
  // budget follows from it rather than being set independently.
  options.engine.scheduler.max_batch_tokens =
      std::max<int64_t>(options.engine.scheduler.max_batch_tokens,
                        options.engine.scheduler.max_seq_len);

  return options;
}

}  // namespace inferx
