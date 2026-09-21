#include <iostream>

#include "kvc/cache.h"

int main(int argc, char** argv) {
  if (argc != 2) return 1;
  auto result = kvc::OpenProvider({argv[1], {}});
  if (!result) {
    std::cerr << result.error().diagnostic << '\n';
    return 1;
  }
  auto& provider = **result;
  if (provider.Id() != "example.scaffold" || !provider.GetCapabilities()) return 1;
  KvcConfig config{};
  config.struct_size = sizeof(config);
  if (provider.Configure(config).code != KVC_UNSUPPORTED) return 1;
  return provider.Shutdown().Ok() ? 0 : 1;
}
