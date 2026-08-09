/// inferx-serve -- the OpenAI-compatible server.
///
/// M4's deliverable, and the point at which the engine becomes usable by
/// something other than a test: a checkpoint directory in, an HTTP endpoint
/// out.

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "inferx/server/engine.h"
#include "inferx/server/http_server.h"
#include "inferx/support/log.h"

namespace {

inferx::server::HttpServer* g_server = nullptr;

void HandleSignal(int /*signum*/) {
  // Only async-signal-safe work here: flip the listener off and let the main
  // thread unwind normally, so in-flight generations are cancelled through the
  // ordinary path rather than by exiting underneath them.
  if (g_server != nullptr) g_server->Stop();
}

void PrintUsage(const char* argv0) {
  std::fprintf(
      stderr,
      "usage: %s --model <dir> [options]\n"
      "\n"
      "  --model <dir>          checkpoint directory (required)\n"
      "  --host <addr>          bind address (default 127.0.0.1)\n"
      "  --port <n>             bind port (default 8000)\n"
      "  --scheduler-endpoint <target>  use a process-separated gRPC "
      "scheduler\n"
      "  --served-model-name <s>  name reported in responses\n"
      "  --api-key-sha256 <hex>  accepted bearer-token hash (repeatable)\n"
      "  --max-running <n>      concurrent sequences (default 8)\n"
      "  --max-seq-len <n>      prompt + generation cap (default 2048)\n"
      "  --kv-blocks <n>        KV cache blocks (default 4096)\n"
      "  --block-size <n>       tokens per block (default 16)\n"
      "  --tensor-parallel-size <n>  tensor-parallel ranks (1 or 2)\n"
      "  --devices <ids>        comma-separated CUDA devices (default 0)\n"
      "  --comm-backend <name>  single or nccl\n"
      "  --collective-timing-sample-rate <n>\n"
      "                         sample every Nth collective (default off)\n"
      "  --fp8                  quantize weights to FP8 e4m3\n"
      "  --w4a16                quantize projection weights to grouped int4\n"
      "  --fp8-kv               store the KV cache as FP8 e4m3\n"
      "  --no-cuda-graphs       skip decode graph capture at startup\n"
      "  --log-level <level>    debug, info, warning, or error\n"
      "  --v <n>                enable VLOG messages through level n\n"
      "  --log-json             write JSON-lines logs to stderr\n"
      "  --log-file <path>      append logs to a file\n"
      "\n"
      "Only greedy decoding is implemented: temperature and top_p are\n"
      "accepted and ignored, and responses report system_fingerprint\n"
      "\"greedy\".\n",
      argv0);
}

// A tiny hand-rolled parser rather than a flags library, because the binary has
// a dozen options and gflags would be a dependency bought for nothing.
bool NextValue(int argc, char** argv, int* i, const char* name,
               std::string* out) {
  if (*i + 1 >= argc) {
    std::fprintf(stderr, "error: %s needs a value\n", name);
    return false;
  }

  *out = argv[++(*i)];
  return true;
}

bool ParseDevices(const std::string& value, std::vector<int>* devices) {
  devices->clear();
  size_t begin = 0;
  while (begin < value.size()) {
    const size_t end = value.find(',', begin);
    const std::string item = value.substr(begin, end - begin);
    if (item.empty()) return false;
    char* tail = nullptr;
    const long parsed = std::strtol(item.c_str(), &tail, 10);
    if (tail == nullptr || *tail != '\0' || parsed < 0) return false;
    devices->push_back(static_cast<int>(parsed));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return !devices->empty();
}

}  // namespace

int main(int argc, char** argv) {
  inferx::server::EngineConfig engine_config;
  inferx::server::HttpServerConfig http_config;
  inferx::LogOptions log_options;
  bool comm_backend_explicit = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    std::string value;

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else if (arg == "--model") {
      if (!NextValue(argc, argv, &i, "--model", &engine_config.model_dir)) {
        return 2;
      }
    } else if (arg == "--host") {
      if (!NextValue(argc, argv, &i, "--host", &http_config.host)) return 2;
    } else if (arg == "--port") {
      if (!NextValue(argc, argv, &i, "--port", &value)) return 2;
      http_config.port = std::atoi(value.c_str());
    } else if (arg == "--scheduler-endpoint") {
      if (!NextValue(argc, argv, &i, "--scheduler-endpoint",
                     &http_config.scheduler_endpoint)) {
        return 2;
      }
    } else if (arg == "--served-model-name") {
      if (!NextValue(argc, argv, &i, "--served-model-name",
                     &engine_config.served_model_name)) {
        return 2;
      }
    } else if (arg == "--api-key-sha256") {
      if (!NextValue(argc, argv, &i, "--api-key-sha256", &value)) return 2;
      http_config.api_key_sha256.push_back(value);
    } else if (arg == "--max-running") {
      if (!NextValue(argc, argv, &i, "--max-running", &value)) return 2;
      engine_config.scheduler.max_running = std::atoll(value.c_str());
    } else if (arg == "--max-seq-len") {
      if (!NextValue(argc, argv, &i, "--max-seq-len", &value)) return 2;
      engine_config.scheduler.max_seq_len = std::atoll(value.c_str());
    } else if (arg == "--kv-blocks") {
      if (!NextValue(argc, argv, &i, "--kv-blocks", &value)) return 2;
      engine_config.kv_blocks = std::atoll(value.c_str());
    } else if (arg == "--block-size") {
      if (!NextValue(argc, argv, &i, "--block-size", &value)) return 2;
      engine_config.block_size = std::atoll(value.c_str());
    } else if (arg == "--tensor-parallel-size") {
      if (!NextValue(argc, argv, &i, "--tensor-parallel-size", &value))
        return 2;
      engine_config.tensor_parallel_size = std::atoi(value.c_str());
    } else if (arg == "--devices") {
      if (!NextValue(argc, argv, &i, "--devices", &value)) return 2;
      if (!ParseDevices(value, &engine_config.devices)) {
        std::fprintf(stderr, "error: invalid --devices list %s\n",
                     value.c_str());
        return 2;
      }
    } else if (arg == "--comm-backend") {
      if (!NextValue(argc, argv, &i, "--comm-backend", &value)) return 2;
      engine_config.comm_backend = value;
      comm_backend_explicit = true;
    } else if (arg == "--collective-timing-sample-rate") {
      if (!NextValue(argc, argv, &i, "--collective-timing-sample-rate",
                     &value)) {
        return 2;
      }
      engine_config.collective_timing_sample_every =
          std::strtoull(value.c_str(), nullptr, 10);
    } else if (arg == "--fp8") {
      engine_config.fp8_weights = true;
    } else if (arg == "--w4a16") {
      engine_config.int4_weights = true;
    } else if (arg == "--fp8-kv") {
      engine_config.fp8_kv_cache = true;
    } else if (arg == "--no-cuda-graphs") {
      engine_config.capture_graphs = false;
    } else if (arg == "--log-level") {
      if (!NextValue(argc, argv, &i, "--log-level", &log_options.min_level)) {
        return 2;
      }
      if (log_options.min_level == "debug") {
        log_options.min_level = "info";
        log_options.verbosity = std::max(log_options.verbosity, 1);
      } else if (log_options.min_level != "info" &&
                 log_options.min_level != "warning" &&
                 log_options.min_level != "error") {
        std::fprintf(stderr, "error: invalid --log-level %s\n",
                     log_options.min_level.c_str());
        return 2;
      }
    } else if (arg == "--v") {
      if (!NextValue(argc, argv, &i, "--v", &value)) return 2;
      log_options.verbosity = std::max(0, std::atoi(value.c_str()));
    } else if (arg == "--log-json") {
      log_options.json = true;
    } else if (arg == "--log-file") {
      if (!NextValue(argc, argv, &i, "--log-file", &value)) return 2;
      log_options.file = value;
    } else {
      std::fprintf(stderr, "error: unknown option %s\n\n", arg.c_str());
      PrintUsage(argv[0]);
      return 2;
    }
  }

  if (engine_config.model_dir.empty()) {
    std::fprintf(stderr, "error: --model is required\n\n");
    PrintUsage(argv[0]);
    return 2;
  }

  if (engine_config.fp8_weights && engine_config.int4_weights) {
    std::fprintf(stderr, "error: --fp8 and --w4a16 are mutually exclusive\n");
    return 2;
  }
  if (!comm_backend_explicit && engine_config.tensor_parallel_size == 2) {
    engine_config.comm_backend = "nccl";
  }

  // The batch is bounded by how many sequences can be resident, so the token
  // budget follows from it rather than being set independently.
  engine_config.scheduler.max_batch_tokens =
      std::max<int64_t>(engine_config.scheduler.max_batch_tokens,
                        engine_config.scheduler.max_seq_len);

  inferx::InitLogging(log_options);
  LOG(INFO) << "loading model from " << engine_config.model_dir;

  inferx::StatusOr<std::unique_ptr<inferx::server::Engine>> engine =
      inferx::server::Engine::Create(engine_config);

  if (!engine.ok()) {
    LOG(ERROR) << "model load failed: " << engine.status();
    return 1;
  }

  inferx::StatusOr<std::unique_ptr<inferx::server::HttpServer>> server =
      inferx::server::HttpServer::Create(engine->get(), http_config);

  if (!server.ok()) {
    LOG(ERROR) << "server creation failed: " << server.status();
    return 1;
  }

  g_server = server->get();
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);

  LOG(INFO) << "inferx-serve listening on http://" << http_config.host << ':'
            << http_config.port << " model=" << (*engine)->model_name()
            << " weights="
            << (engine_config.fp8_weights ? "fp8-e4m3" :
                engine_config.int4_weights ? "w4a16" : "bf16")
            << " cuda_graphs=" << (engine_config.capture_graphs ? "on" : "off")
            << " max_running=" << engine_config.scheduler.max_running
            << " max_seq_len=" << engine_config.scheduler.max_seq_len
            << " kv_blocks=" << engine_config.kv_blocks
            << " block_size=" << engine_config.block_size;

  const inferx::Status listened = (*server)->Listen();

  g_server = nullptr;

  if (!listened.ok()) {
    LOG(ERROR) << "listen failed: " << listened;
    return 1;
  }

  LOG(INFO) << "shutting down";
  return 0;
}
