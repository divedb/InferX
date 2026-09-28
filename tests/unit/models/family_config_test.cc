#include <string>

#include "gtest/gtest.h"
#include "inferx/core/status.h"
#include "inferx/models/causal/decoder_config.h"
#include "inferx/models/components/decoder_layer.h"
#include "inferx/models/model_registry.h"

namespace inferx {
namespace {

/// Minimal dense-attention dimensions every family config shares.
constexpr const char* kBaseDims =
    "\"hidden_size\":32,\"intermediate_size\":64,\"num_hidden_layers\":2,"
    "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":64,"
    "\"vocab_size\":100,\"max_position_embeddings\":128,"
    "\"rms_norm_eps\":0.00001,\"rope_theta\":10000.0,\"tie_word_embeddings\":false";

std::string Config(std::string_view model_type, std::string_view extra = "") {
  std::string json = std::string("{\"model_type\":\"") + model_type.data() + "\"";
  if (!extra.empty()) {
    json += "," + std::string(extra.data());
  }
  return json + "," + kBaseDims + "}";
}

CheckpointConfig Identity(std::string_view model_type) {
  CheckpointConfig identity;
  identity.model_type = std::string(model_type);
  return identity;
}

TEST(FamilyConfigTest, LlamaTranslatesDenseConfig) {
  const auto config = TranslateFamilyConfig(Identity("llama"), Config("llama"));
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->model.hidden_size, 32);
  EXPECT_EQ(config->blocks.size(), 2u);
  const auto& a = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  EXPECT_FALSE(a.qk_norm);
  EXPECT_EQ(a.query_heads, 4);
  EXPECT_EQ(a.kv_heads, 2);
  EXPECT_TRUE(std::holds_alternative<components::SwiGluConfig>(config->blocks[0].feed_forward));
  const auto executable = config->ValidateExecutable();
  EXPECT_TRUE(executable.ok()) << executable.ToString();
}

TEST(FamilyConfigTest, Qwen3TranslatesQkNorm) {
  const auto config = TranslateFamilyConfig(Identity("qwen3"), Config("qwen3"));
  ASSERT_TRUE(config.ok()) << config.status();
  const auto& a = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  EXPECT_TRUE(a.qk_norm);
  const auto executable = config->ValidateExecutable();
  EXPECT_TRUE(executable.ok()) << executable.ToString();
}

TEST(FamilyConfigTest, MoEIsRejectedAtBuildTimeNotForwardTime) {
  const auto config = TranslateFamilyConfig(
      Identity("qwen3_moe"),
      Config("qwen3_moe",
             "\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,"
             "\"shared_expert_intermediate_size\":32,\"norm_topk_prob\":true"));
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(std::holds_alternative<components::MoeConfig>(config->blocks[0].feed_forward));
  const auto executable = config->ValidateExecutable();
  ASSERT_FALSE(executable.ok());
  EXPECT_EQ(executable.code(), absl::StatusCode::kUnimplemented);
}

TEST(FamilyConfigTest, RecurrentLayersAreRejectedAsUnexecutable) {
  const auto config = TranslateFamilyConfig(
      Identity("qwen3_next"),
      Config("qwen3_next",
             "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
             "\"full_attention_interval\":2,"
             "\"linear_num_key_heads\":2,\"linear_num_value_heads\":2,"
             "\"linear_key_head_dim\":64,\"linear_value_head_dim\":64,"
             "\"linear_conv_kernel_dim\":4,"
             "\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,"
             "\"shared_expert_intermediate_size\":32"));
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(std::holds_alternative<components::GatedDeltaNetConfig>(config->blocks[0].mixer));
  const auto executable = config->ValidateExecutable();
  ASSERT_FALSE(executable.ok());
  EXPECT_EQ(executable.code(), absl::StatusCode::kUnimplemented);
}

TEST(FamilyConfigTest, UnknownModelTypeIsRejected) {
  const auto config = TranslateFamilyConfig(Identity("flux"), Config("flux"));
  EXPECT_FALSE(config.ok());
}

TEST(FamilyConfigTest, InvalidJsonIsRejected) {
  const auto config = TranslateFamilyConfig(Identity("llama"), "not json");
  EXPECT_FALSE(config.ok());
}

/// Factories must remain reachable through the registry in static builds.
TEST(FamilyConfigTest, BuiltinFamiliesResolve) {
  for (const char* type : {"llama", "qwen3", "qwen3_moe", "qwen3_next"}) {
    const auto family = ResolveFamily(Identity(type));
    EXPECT_TRUE(family.ok()) << type << ": " << family.status();
  }
}

TEST(FamilyConfigTest, ArchitectureAliasesSelectTheSameFactory) {
  for (const char* type : {"llama", "qwen3", "qwen3_moe", "qwen3_next"}) {
    const auto by_type = ResolveFamily(Identity(type));
    ASSERT_TRUE(by_type.ok());
    CheckpointConfig identity;
    identity.architectures = std::string((*by_type)->architecture);
    const auto by_arch = ResolveFamily(identity);
    ASSERT_TRUE(by_arch.ok()) << by_arch.status();
    EXPECT_EQ(*by_arch, *by_type);
    EXPECT_NE((*by_arch)->build, nullptr);
  }
}

TEST(FamilyConfigTest, ConflictingIdentitiesAreRejected) {
  auto identity = Identity("llama");
  identity.architectures = "Qwen3ForCausalLM";
  EXPECT_EQ(ResolveFamily(identity).status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(FamilyConfigTest, Qwen3MoeTranslatesThroughArchitectureAlias) {
  CheckpointConfig identity;
  identity.architectures = "Qwen3MoeForCausalLM";
  const auto config = TranslateFamilyConfig(
      identity,
      std::string("{") + kBaseDims +
          ",\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32}");
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(std::holds_alternative<components::MoeConfig>(config->blocks[0].feed_forward));
}

TEST(FamilyConfigTest, MalformedFieldsReturnStatusInsteadOfThrowing) {
  for (const std::string json :
       {std::string("[]"), Config("llama", "\"attention_bias\":\"invalid\""),
        Config("qwen3_moe")}) {
    EXPECT_EQ(TranslateFamilyConfig(Identity("qwen3_moe"), json).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(FamilyConfigTest, DenseQwen3RejectsUnknownLayerTypes) {
  const auto config = TranslateFamilyConfig(
      Identity("qwen3"),
      Config("qwen3", "\"layer_types\":[\"linear_attention\",\"full_attention\"]"));
  EXPECT_EQ(config.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(FamilyConfigTest, TypedBuilderRejectsUnsupportedBlocksBeforeReadingWeights) {
  for (const auto& json : {Config("qwen3_moe",
                                  "\"num_experts\":4,\"num_experts_per_tok\":2,"
                                  "\"moe_intermediate_size\":32")}) {
    models::LoadedCheckpoint checkpoint;
    auto identity = CheckpointConfig::FromJson(json);
    ASSERT_TRUE(identity.ok());
    checkpoint.config = *identity;
    checkpoint.config_json = json;
    // No weight files are opened. A failed weight lookup would report a
    // different status than the expected unsupported-execution rejection.
    const auto model = BuildModel(checkpoint, DeviceId::Cpu(), 8, 2);
    EXPECT_EQ(model.status().code(), absl::StatusCode::kUnimplemented);
  }
}

}  // namespace
}  // namespace inferx
