#include "inferx/support/options.h"

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace inferx {
namespace {

using ::testing::HasSubstr;

class ServeOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_dir_ = std::filesystem::path(::testing::TempDir()) / "checkpoint";
    std::filesystem::create_directories(model_dir_);
  }

  std::optional<ServeOptions> Parse(std::vector<std::string> args,
                                    int* exit_code) {
    args.insert(args.begin(), "inferx-serve");
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (std::string& arg : args) argv.push_back(arg.data());
    return ParseServeOptions(static_cast<int>(argv.size()), argv.data(),
                             exit_code);
  }

  std::optional<ServeOptions> ParseWithModel(std::vector<std::string> args,
                                             int* exit_code) {
    args.insert(args.begin(), {"--model", model_dir_.string()});
    return Parse(std::move(args), exit_code);
  }

  std::filesystem::path model_dir_;
};

TEST_F(ServeOptionsTest, VllmCanonicalNamesPopulateConfig) {
  int exit_code = 0;
  auto options = ParseWithModel({"--max-num-seqs",
                                 "4",
                                 "--max-model-len",
                                 "4096",
                                 "--num-gpu-blocks-override",
                                 "512",
                                 "--enforce-eager",
                                 "--dtype",
                                 "bfloat16",
                                 "--quantization",
                                 "fp8",
                                 "--kv-cache-dtype",
                                 "fp8",
                                 "--device-ids",
                                 "0",
                                 "--gpu-memory-utilization",
                                 "0.5",
                                 "--max-num-batched-tokens",
                                 "1024",
                                 "--no-enable-prefix-caching"},
                                &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->engine.model_dir, model_dir_.string());
  EXPECT_EQ(options->engine.scheduler.max_running, 4);
  EXPECT_EQ(options->engine.scheduler.max_seq_len, 4096);
  EXPECT_EQ(options->engine.kv_blocks, 512);
  EXPECT_FALSE(options->engine.capture_graphs);
  EXPECT_TRUE(options->engine.fp8_weights);
  EXPECT_FALSE(options->engine.int4_weights);
  EXPECT_TRUE(options->engine.fp8_kv_cache);
  EXPECT_DOUBLE_EQ(options->engine.gpu_memory_utilization, 0.5);
  // Explicit --max-num-batched-tokens is honored verbatim, below max-model-len.
  EXPECT_EQ(options->engine.scheduler.max_batch_tokens, 1024);
  EXPECT_FALSE(options->engine.scheduler.enable_prefix_cache);
}

TEST_F(ServeOptionsTest, PositionalModelMatchesVllmServe) {
  int exit_code = 0;
  auto options = Parse({model_dir_.string()}, &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->engine.model_dir, model_dir_.string());
}

TEST_F(ServeOptionsTest, MissingModelIsAnError) {
  int exit_code = 0;
  EXPECT_FALSE(Parse({"--max-num-seqs", "4"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, DeprecatedAliasesStillParse) {
  int exit_code = 0;
  auto options = ParseWithModel(
      {"--max-running", "5", "--max-seq-len", "1024", "--kv-blocks", "256",
       "--no-cuda-graphs", "--fp8-kv", "--devices", "0"},
      &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->engine.scheduler.max_running, 5);
  EXPECT_EQ(options->engine.scheduler.max_seq_len, 1024);
  EXPECT_EQ(options->engine.kv_blocks, 256);
  EXPECT_FALSE(options->engine.capture_graphs);
  EXPECT_TRUE(options->engine.fp8_kv_cache);
}

TEST_F(ServeOptionsTest, AliasAndCanonicalTogetherIsAnError) {
  int exit_code = 0;
  EXPECT_FALSE(
      ParseWithModel({"--max-running", "5", "--max-num-seqs", "6"}, &exit_code)
          .has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, DerivationsPreserved) {
  int exit_code = 0;
  auto options = ParseWithModel({"--tensor-parallel-size", "2", "--device-ids",
                                 "0,1", "--max-model-len", "8192"},
                                &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->engine.comm_backend, "nccl");
  // Unset, the token budget still derives from the sequence cap.
  EXPECT_EQ(options->engine.scheduler.max_batch_tokens, 8192);
}

TEST_F(ServeOptionsTest, ParsesExecutionBackend) {
  int exit_code = 0;
  auto options = ParseWithModel({"--device", "rocm"}, &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->engine.device_kind, DeviceKind::kRocm);
}

TEST_F(ServeOptionsTest, DtypeOtherThanBf16IsAnError) {
  int exit_code = 0;
  EXPECT_FALSE(ParseWithModel({"--dtype", "half"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, UnsupportedQuantizationIsAnError) {
  int exit_code = 0;
  EXPECT_FALSE(
      ParseWithModel({"--quantization", "awq"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, UnsupportedKvCacheDtypeIsAnError) {
  int exit_code = 0;
  EXPECT_FALSE(
      ParseWithModel({"--kv-cache-dtype", "fp8_e5m2"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, ChunkedPrefillCannotBeDisabled) {
  int exit_code = 0;
  EXPECT_FALSE(
      ParseWithModel({"--no-enable-chunked-prefill"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
  auto options = ParseWithModel({"--enable-chunked-prefill"}, &exit_code);
  EXPECT_TRUE(options.has_value());
}

TEST_F(ServeOptionsTest, ApiKeyIsStoredHashed) {
  int exit_code = 0;
  auto options = ParseWithModel({"--api-key", "secret"}, &exit_code);
  ASSERT_TRUE(options.has_value());
  ASSERT_EQ(options->http.api_key_sha256.size(), 1u);
  EXPECT_EQ(options->http.api_key_sha256.front(),
            "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b");
}

TEST_F(ServeOptionsTest, CompatStubAtDefaultParses) {
  int exit_code = 0;
  EXPECT_TRUE(ParseWithModel({"--pipeline-parallel-size", "1",
                              "--scheduling-policy", "fcfs"},
                             &exit_code)
                  .has_value());
}

TEST_F(ServeOptionsTest, CompatStubAtNonDefaultIsAnError) {
  int exit_code = 0;
  EXPECT_FALSE(ParseWithModel({"--pipeline-parallel-size", "2"}, &exit_code)
                   .has_value());
  EXPECT_EQ(exit_code, 2);
  EXPECT_FALSE(ParseWithModel({"--enable-lora"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
  EXPECT_FALSE(
      ParseWithModel({"--speculative-config", "{}"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, AcceptAnyStubParses) {
  int exit_code = 0;
  EXPECT_TRUE(
      ParseWithModel({"--trust-remote-code", "--disable-log-stats"}, &exit_code)
          .has_value());
}

TEST_F(ServeOptionsTest, HttpExtensionKnobs) {
  int exit_code = 0;
  auto options =
      ParseWithModel({"--max-active-requests", "64", "--request-timeout", "120",
                      "--io-threads", "2"},
                     &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->http.max_active_requests, 64u);
  EXPECT_EQ(options->http.request_timeout_seconds, 120);
  EXPECT_EQ(options->http.io_threads, 2u);
}

TEST_F(ServeOptionsTest, TomlConfigFileRoundTrips) {
  const std::filesystem::path config =
      std::filesystem::path(::testing::TempDir()) / "serve.toml";
  {
    std::ofstream out(config);
    out << "max-num-seqs = 6\nmax-model-len = 4096\n";
  }
  int exit_code = 0;
  auto options = ParseWithModel({"--config", config.string()}, &exit_code);
  ASSERT_TRUE(options.has_value());
  EXPECT_EQ(options->engine.scheduler.max_running, 6);
  EXPECT_EQ(options->engine.scheduler.max_seq_len, 4096);
}

TEST_F(ServeOptionsTest, YamlConfigIsRejected) {
  int exit_code = 0;
  EXPECT_FALSE(
      ParseWithModel({"--config", "serve.yaml"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 2);
}

TEST_F(ServeOptionsTest, HelpExitsZero) {
  int exit_code = 1;
  EXPECT_FALSE(Parse({"--help"}, &exit_code).has_value());
  EXPECT_EQ(exit_code, 0);
}

}  // namespace
}  // namespace inferx
