#pragma once

#include <string>
#include <vector>

namespace inferx {

struct ApiServerConfig {
  /// The host address the API server will bind to. The default is "127.0.0.1".
  std::string host = "127.0.0.1";

  /// The port the API server will listen on. The default is 8000.
  int port = 8000;

  ///< Bearer API keys guarding /v1 and /metrics (vLLM --api-keys); empty disables auth.
  ///< INFERX_API_KEY is merged in at startup.
  std::vector<std::string> api_keys;
};

}  // namespace inferx