// Loader behavior: missing libraries, missing entry symbols, ABI-version
// rejection, and a working open/capabilities/shutdown cycle.

#include <cstdio>

#include "kvc/cache.h"
#include "test_util.h"

namespace {

using namespace kvc_test;

void MissingLibrary() {
  auto provider = kvc::OpenProvider({"/nonexistent/libkvc_missing.so", {}});
  CHECK(!provider);
  CHECK_EQ(provider.error().code, KVC_UNAVAILABLE);
}

void EmptyPathRejected() {
  auto provider = kvc::OpenProvider({"", {}});
  CHECK(!provider);
  CHECK_EQ(provider.error().code, KVC_INVALID_ARGUMENT);
}

void BadAbiMajor(const char* path) {
  auto provider = kvc::OpenProvider({path, {}});
  CHECK(!provider);
  CHECK_EQ(provider.error().code, KVC_UNSUPPORTED);
}

void MissingEntry(const char* path) {
  auto provider = kvc::OpenProvider({path, {}});
  CHECK(!provider);
  CHECK_EQ(provider.error().code, KVC_UNSUPPORTED);
}

void OpenConfigureShutdown(const char* path) {
  auto opened = kvc::OpenProvider({path, {}});
  CHECK(opened);
  auto& provider = **opened;
  auto capabilities = provider.GetCapabilities();
  CHECK(capabilities);
  CHECK(capabilities->memory_types != 0);
  CHECK((capabilities->layouts & (1u << KVC_OPAQUE_BYTES)) != 0);
  ConfigBuilder builder;
  auto config = builder.Build();
  CHECK(provider.Configure(config.config).Ok());
  auto installed = provider.Configuration();
  CHECK(installed);
  CHECK_EQ((*installed)->group_count, 1u);
  CHECK(provider.Shutdown().Ok());
  // After a successful shutdown the session no longer admits new work.
  auto status = provider.Configure(config.config);
  CHECK(!status.Ok());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <bad-abi.so> <missing-entry.so> [provider.so]\n", argv[0]);
    return 2;
  }
  TEST(MissingLibrary);
  TEST(EmptyPathRejected);
  RunTest("BadAbiMajor", [path = argv[1]] { BadAbiMajor(path); });
  RunTest("MissingEntry", [path = argv[2]] { MissingEntry(path); });
  if (argc > 3)
    RunTest("OpenConfigureShutdown", [path = argv[3]] { OpenConfigureShutdown(path); });
  return ExitCode();
}
