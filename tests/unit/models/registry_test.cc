#include "gtest/gtest.h"
#include "inferx/models/load_options.h"
#include "inferx/models/registry.h"

namespace inferx {
namespace models {
namespace {

TEST(RegistryTest, ResolvesByArchitecture) {
  const ArchitectureRegistry& registry = ArchitectureRegistry::Builtins();
  auto entry = registry.Resolve("", "Qwen3ForCausalLM");
  ASSERT_TRUE(entry.ok()) << entry.status();
  EXPECT_STREQ((*entry)->model_type, "qwen3");
}

TEST(RegistryTest, ResolvesByModelType) {
  const ArchitectureRegistry& registry = ArchitectureRegistry::Builtins();
  for (const char* type : {"qwen3", "qwen3_moe", "llama"}) {
    auto entry = registry.Resolve(type, "");
    ASSERT_TRUE(entry.ok()) << entry.status() << " for " << type;
  }
}

TEST(RegistryTest, ArchitectureWinsOverModelType) {
  const ArchitectureRegistry& registry = ArchitectureRegistry::Builtins();
  auto entry = registry.Resolve("llama", "Qwen3MoeForCausalLM");
  ASSERT_TRUE(entry.ok()) << entry.status();
  EXPECT_STREQ((*entry)->model_type, "qwen3_moe");
}

TEST(RegistryTest, UnknownIsNotFound) {
  const ArchitectureRegistry& registry = ArchitectureRegistry::Builtins();
  auto entry = registry.Resolve("gpt_oss", "GptOssForCausalLM");
  EXPECT_FALSE(entry.ok());
}

TEST(RegistryTest, DtypeResolution) {
  const char* qwen3 =
      R"({"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "torch_dtype": "bfloat16"})";
  auto config = HfConfig::FromJson(qwen3);
  ASSERT_TRUE(config.ok()) << config.status();

  auto dtype = ResolveDType(DType::kAuto, *config);
  ASSERT_TRUE(dtype.ok()) << dtype.status();
  EXPECT_EQ(*dtype, DataType::kBFloat16);

  ASSERT_TRUE(ResolveDType(DType::kBFloat16, *config).ok());
  EXPECT_EQ(ResolveDType(DType::kFloat16, *config).status().code(),
            absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace models
}  // namespace inferx
