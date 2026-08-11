#include "inferx/support/vllm_compat.h"

#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace inferx {
namespace {

using ::testing::HasSubstr;

Status ParseAndCheck(std::vector<std::string> args) {
  CLI::App app{"test"};
  CompatState state;
  AddCompatStubs(app, ServeCompatStubs(), state);
  args.insert(args.begin(), "test");
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (std::string& arg : args) argv.push_back(arg.data());
  try {
    app.parse(static_cast<int>(argv.size()), argv.data());
  } catch (const CLI::ParseError& e) {
    return InvalidArgumentError("parse error: ", e.what());
  }
  return CheckCompatStubs(ServeCompatStubs(), state);
}

TEST(VllmCompatTest, UnsetStubsPass) {
  EXPECT_TRUE(ParseAndCheck({}).ok());
}

TEST(VllmCompatTest, DefaultValuesPass) {
  EXPECT_TRUE(ParseAndCheck({"--pipeline-parallel-size", "1"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--data-parallel-size", "1"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--no-enable-lora"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--scheduling-policy", "fcfs"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--watermark", "0.0"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--master-port", "29501"}).ok());
}

TEST(VllmCompatTest, NonDefaultValuesErrorAndNameTheFeature) {
  Status status = ParseAndCheck({"--pipeline-parallel-size", "2"});
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("pipeline parallelism"));
  EXPECT_THAT(std::string(status.message()), HasSubstr("not supported"));

  status = ParseAndCheck({"--enable-lora"});
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()), HasSubstr("LoRA"));
}

TEST(VllmCompatTest, NoneDefaultedStubRejectsAnyValue) {
  Status status = ParseAndCheck({"--speculative-config", "{\"method\":1}"});
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              HasSubstr("speculative decoding"));
}

TEST(VllmCompatTest, AcceptAnyStubAlwaysPasses) {
  EXPECT_TRUE(ParseAndCheck({"--trust-remote-code"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--uvicorn-log-level", "debug"}).ok());
  EXPECT_TRUE(ParseAndCheck({"--no-use-tqdm-on-load"}).ok());
}

TEST(VllmCompatTest, BoolStubsAcceptBothSpellings) {
  // Default-true booleans: the positive spelling is the default.
  EXPECT_TRUE(ParseAndCheck({"--disable-cascade-attn"}).ok());
  Status status = ParseAndCheck({"--no-disable-cascade-attn"});
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()), HasSubstr("cascade attention"));
}

TEST(VllmCompatTest, GatewayStubsAreAFrontendSubset) {
  EXPECT_FALSE(GatewayCompatStubs().empty());
  for (const CompatStub& stub : GatewayCompatStubs()) {
    EXPECT_NE(stub.name, "--enable-lora");
    EXPECT_NE(stub.name, "--chat-template");  // real gateway flag
  }
}

TEST(VllmCompatTest, Sha256HexMatchesKnownVector) {
  EXPECT_EQ(Sha256Hex("secret"),
            "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b");
}

}  // namespace
}  // namespace inferx
