#include "options.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "CLI/CLI.hpp"
#include "absl/log/log.h"
#include "inferx/support/vllm_compat.h"

namespace inferx::server::gateway {

std::optional<GatewayOptions> ParseGatewayOptions(int argc, char** argv,
                                                  int* exit_code) {
  GatewayOptions options;
  std::string log_level = "info";
  std::vector<std::string> api_keys;
  CompatState compat;

  CLI::App app{
      "inferx-gateway -- the process-separated OpenAI-compatible gateway.\n"
      "\n"
      "Frontend flags follow vLLM's names; flags for features InferX does\n"
      "not implement parse, but any value other than the vLLM default is a\n"
      "startup error naming the missing feature."};
  app.option_defaults()->always_capture_default();

  app.add_option("--scheduler-endpoint", options.config.scheduler_endpoint,
                 "gRPC scheduler target (required)")
      ->required();
  app.add_option("--tokenizer", options.config.tokenizer_directory,
                 "tokenizer directory (required)")
      ->required()
      ->check(CLI::ExistingDirectory);
  app.add_option("--model", options.config.model_id,
                 "served model id (required)")
      ->required();
  app.add_option("--model-version", options.config.model_version,
                 "scheduler model version");
  app.add_option("--tokenizer-revision", options.config.tokenizer_revision,
                 "scheduler contract revision");
  app.add_option("--chat-template", options.config.chat_template,
                 "chat template kind: qwen2 or deepseek-v2 (Jinja template "
                 "files are not supported)");
  app.add_option("--host", options.config.host, "bind address");
  app.add_option("--port", options.config.port, "bind port")
      ->check(CLI::Range(1, 65535));
  app.add_option("--api-key", api_keys,
                 "accepted bearer token (repeatable; stored hashed)");
  app.add_option("--api-key-sha256", options.config.api_key_sha256,
                 "accepted bearer-token SHA-256 hash (repeatable)")
      ->group("InferX extensions");
  app.add_option("--log-level", log_level, "debug, info, warning, or error")
      ->check(CLI::IsMember({"debug", "info", "warning", "error"}));
  app.add_option("--v", options.log.verbosity,
                 "enable VLOG messages through level n")
      ->check(CLI::NonNegativeNumber);
  app.add_flag("--log-json", options.log.json,
               "write JSON-lines logs to stderr");
  app.add_option("--log-file", options.log.file, "append logs to a file");

  AddCompatStubs(app, GatewayCompatStubs(), compat);

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    *exit_code = app.exit(e) == 0 ? 0 : 2;
    return std::nullopt;
  }

  Status status = OkStatus();
  const std::string& tmpl = options.config.chat_template;
  if (tmpl != "qwen2" && tmpl != "deepseek-v2") {
    // vLLM's --chat-template takes a Jinja file; InferX ships transcribed
    // templates selected by name. A path-looking value gets the clear error.
    const bool looks_like_file = tmpl.find('/') != std::string::npos ||
                                 tmpl.ends_with(".jinja") ||
                                 std::filesystem::exists(tmpl);
    status = InvalidArgumentError(
        looks_like_file
            ? "--chat-template files (Jinja) are not supported by InferX; "
              "use a built-in template name: qwen2, deepseek-v2"
            : "--chat-template must be one of: qwen2, deepseek-v2");
  }

  if (status.ok()) status = CheckCompatStubs(GatewayCompatStubs(), compat);
  if (status.ok()) {
    for (const std::string& key : api_keys) {
      std::string hash = Sha256Hex(key);
      if (hash.empty()) {
        status = InternalError("failed to hash --api-key");
        break;
      }
      options.config.api_key_sha256.push_back(std::move(hash));
    }
  }
  if (!status.ok()) {
    std::cerr << "inferx-gateway: " << status.message() << std::endl;
    *exit_code = 2;
    return std::nullopt;
  }

  if (log_level == "debug") {
    options.log.min_level = "info";
    options.log.verbosity = std::max(options.log.verbosity, 1);
  } else {
    options.log.min_level = log_level;
  }

  return options;
}

}  // namespace inferx::server::gateway
