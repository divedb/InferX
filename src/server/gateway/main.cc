#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "inferx/server/gateway/gateway_server.h"
#include "inferx/support/log.h"

namespace {

inferx::server::gateway::GatewayServer* g_gateway = nullptr;

void HandleSignal(int) {
  if (g_gateway != nullptr) g_gateway->Stop();
}

void Usage(const char* program) {
  std::fprintf(
      stderr,
      "usage: %s --scheduler-endpoint <target> --tokenizer <dir> "
      "--model <id> [options]\n"
      "  --model-version <version>       default: loaded\n"
      "  --tokenizer-revision <revision> scheduler contract revision\n"
      "  --chat-template <kind>          qwen2 (default) or deepseek-v2\n"
      "  --host <address>                default: 127.0.0.1\n"
      "  --port <number>                 default: 8000\n"
      "  --api-key-sha256 <hex>          repeatable bearer-token hash\n"
      "  --log-level <level>              debug, info, warning, or error\n"
      "  --v <n>                          VLOG level\n"
      "  --log-json                       JSON-lines stderr output\n"
      "  --log-file <path>                append logs to a file\n",
      program);
}

bool Value(int argc, char** argv, int* index, const char* option,
           std::string* output) {
  if (*index + 1 >= argc) {
    std::fprintf(stderr, "error: %s needs a value\n", option);
    return false;
  }
  *output = argv[++(*index)];
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  inferx::server::gateway::GatewayServerConfig config;
  inferx::LogOptions log_options;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    std::string value;
    if (option == "--help" || option == "-h") {
      Usage(argv[0]);
      return 0;
    }
    if (option == "--scheduler-endpoint") {
      if (!Value(argc, argv, &i, option.c_str(), &config.scheduler_endpoint)) {
        return 2;
      }
    } else if (option == "--tokenizer") {
      if (!Value(argc, argv, &i, option.c_str(), &config.tokenizer_directory)) {
        return 2;
      }
    } else if (option == "--model") {
      if (!Value(argc, argv, &i, option.c_str(), &config.model_id)) return 2;
    } else if (option == "--model-version") {
      if (!Value(argc, argv, &i, option.c_str(), &config.model_version)) {
        return 2;
      }
    } else if (option == "--tokenizer-revision") {
      if (!Value(argc, argv, &i, option.c_str(), &config.tokenizer_revision)) {
        return 2;
      }
    } else if (option == "--chat-template") {
      if (!Value(argc, argv, &i, option.c_str(), &config.chat_template)) {
        return 2;
      }
    } else if (option == "--host") {
      if (!Value(argc, argv, &i, option.c_str(), &config.host)) return 2;
    } else if (option == "--port") {
      if (!Value(argc, argv, &i, option.c_str(), &value)) return 2;
      config.port = std::atoi(value.c_str());
    } else if (option == "--api-key-sha256") {
      if (!Value(argc, argv, &i, option.c_str(), &value)) return 2;
      config.api_key_sha256.push_back(value);
    } else if (option == "--log-level") {
      if (!Value(argc, argv, &i, option.c_str(), &log_options.min_level)) {
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
    } else if (option == "--v") {
      if (!Value(argc, argv, &i, option.c_str(), &value)) return 2;
      log_options.verbosity = std::max(0, std::atoi(value.c_str()));
    } else if (option == "--log-json") {
      log_options.json = true;
    } else if (option == "--log-file") {
      if (!Value(argc, argv, &i, option.c_str(), &value)) return 2;
      log_options.file = value;
    } else {
      std::fprintf(stderr, "error: unknown option %s\n", option.c_str());
      Usage(argv[0]);
      return 2;
    }
  }

  inferx::InitLogging(log_options);
  auto gateway = inferx::server::gateway::GatewayServer::Create(config);
  if (!gateway.ok()) {
    LOG(ERROR) << "gateway creation failed: " << gateway.status();
    return 1;
  }
  g_gateway = gateway->get();
  std::signal(SIGINT, HandleSignal);
  std::signal(SIGTERM, HandleSignal);
  LOG(INFO) << "inferx-gateway listening on http://" << config.host << ':'
            << config.port;
  const auto status = (*gateway)->Listen();
  g_gateway = nullptr;
  if (!status.ok()) {
    LOG(ERROR) << "gateway listen failed: " << status;
    return 1;
  }
  return 0;
}
