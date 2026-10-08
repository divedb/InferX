#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "absl/status/status.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace {

std::string Msg(const Status& s) { return s.ToString(); }

std::string WriteTemp(const std::string& name, const std::string& content) {
  const std::string path = std::string(testing::TempDir()) + "/" + name;
  { std::ofstream out(path, std::ios::binary); out << content; }
  return path;
}

TEST(HfConfigTest, FromJsonParsesIdentityAndRoot) {
  const auto cfg = HfConfig::FromJson(
      R"({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",)"
      R"( "torch_dtype": "bfloat16"})");
  ASSERT_TRUE(cfg.ok()) << cfg.status();
  EXPECT_EQ(cfg->Architecture(), "Qwen3ForCausalLM");
  EXPECT_EQ(cfg->ModelType(), "qwen3");

  const auto dtype = cfg->Root().RequireStr("torch_dtype");
  ASSERT_TRUE(dtype.ok()) << dtype.status();
  EXPECT_EQ(*dtype, "bfloat16");
}

TEST(HfConfigTest, ModelTypeIsOptional) {
  const auto cfg = HfConfig::FromJson(R"({"architectures": ["LlamaForCausalLM"]})");
  ASSERT_TRUE(cfg.ok()) << cfg.status();
  EXPECT_EQ(cfg->ModelType(), "");
}

TEST(HfConfigTest, CopyKeepsViewsValid) {
  auto cfg = HfConfig::FromJson(
      R"({"architectures": ["DeepseekV3ForCausalLM"], "hidden_size": 7168})");
  ASSERT_TRUE(cfg.ok()) << cfg.status();

  const HfConfig copy = *cfg;
  cfg = HfConfig();  // drop the original; the copy owns the document
  EXPECT_EQ(copy.Architecture(), "DeepseekV3ForCausalLM");
  const auto hidden = copy.Root().RequireInt("hidden_size");
  ASSERT_TRUE(hidden.ok()) << hidden.status();
  EXPECT_EQ(*hidden, 7168);
}

TEST(HfConfigTest, RejectsInvalidJson) {
  const auto cfg = HfConfig::FromJson("{not json");
  ASSERT_FALSE(cfg.ok());
  EXPECT_EQ(cfg.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(Msg(cfg.status()).find("not valid JSON"), std::string::npos);
}

TEST(HfConfigTest, RejectsNonObjectRoot) {
  const auto cfg = HfConfig::FromJson("[1, 2, 3]");
  ASSERT_FALSE(cfg.ok());
  EXPECT_EQ(cfg.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(Msg(cfg.status()).find("root must be an object"), std::string::npos);
}

TEST(HfConfigTest, ArchitecturesMustBeNonEmptyStringList) {
  const char* kBad[] = {
      "{}",
      R"({"architectures": []})",
      R"({"architectures": "LlamaForCausalLM"})",
      R"({"architectures": [42]})",
      R"({"architectures": null})",
  };
  for (const char* json : kBad) {
    const auto cfg = HfConfig::FromJson(json);
    ASSERT_FALSE(cfg.ok()) << json;
    EXPECT_EQ(cfg.status().code(), absl::StatusCode::kInvalidArgument) << json;
    EXPECT_NE(Msg(cfg.status()).find("'architectures'"), std::string::npos) << json;
  }
}

TEST(HfConfigTest, FromFileReadsConfig) {
  const std::string path = WriteTemp(
      "hf_config_ok.json",
      R"({"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe"})");
  const auto cfg = HfConfig::FromFile(path);
  ASSERT_TRUE(cfg.ok()) << cfg.status();
  EXPECT_EQ(cfg->Architecture(), "Qwen3MoeForCausalLM");
  EXPECT_EQ(cfg->ModelType(), "qwen3_moe");
}

TEST(HfConfigTest, FromFileReportsMissingFile) {
  const auto cfg = HfConfig::FromFile("/nonexistent/hf_config.json");
  ASSERT_FALSE(cfg.ok());
  EXPECT_EQ(cfg.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(Msg(cfg.status()).find("/nonexistent/hf_config.json"), std::string::npos);
}

TEST(HfConfigTest, FromFileWrapsParseErrorsWithPath) {
  const std::string path = WriteTemp("hf_config_bad.json", "{oops");
  const auto cfg = HfConfig::FromFile(path);
  ASSERT_FALSE(cfg.ok());
  EXPECT_NE(Msg(cfg.status()).find(path), std::string::npos);
  EXPECT_NE(Msg(cfg.status()).find("not valid JSON"), std::string::npos);
}

}  // namespace
}  // namespace inferx
