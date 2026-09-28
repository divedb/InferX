#include <string>

#include "gtest/gtest.h"
#include "inferx/core/status.h"
#include "inferx/models/causal/decoder_config.h"
#include "inferx/models/components/decoder_layer.h"
#include <cmath>
#include <fstream>

#include "inferx/models/model_registry.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/moe.h"

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

TEST(FamilyConfigTest, MoETranslatesAndValidatesAsExecutable) {
  const auto config = TranslateFamilyConfig(
      Identity("qwen3_moe"),
      Config("qwen3_moe",
             "\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,"
             "\"shared_expert_intermediate_size\":32,\"norm_topk_prob\":true"));
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_TRUE(std::holds_alternative<components::MoeConfig>(config->blocks[0].feed_forward));
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, RecurrentLayersValidateAsExecutable) {
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
  EXPECT_TRUE(config->ValidateExecutable().ok());
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

TEST(FamilyConfigTest, Qwen25TranslatesBiasedAttention) {
  const auto config = TranslateFamilyConfig(
      Identity("qwen2"), Config("qwen2", "\"attention_bias\":true,\"hidden_act\":\"silu\""));
  ASSERT_TRUE(config.ok()) << config.status();
  const auto& a = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  EXPECT_TRUE(a.qkv_bias);
  EXPECT_FALSE(a.output_bias);  // Qwen2 biases QKV only, not o_proj.
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, MistralTranslatesSlidingWindow) {
  const auto config =
      TranslateFamilyConfig(Identity("mistral"),
                            Config("mistral", "\"sliding_window\":4096,\"use_sliding_window\":true"));
  ASSERT_TRUE(config.ok()) << config.status();
  const auto& a = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  EXPECT_EQ(a.sliding_window, 4096);
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, Gemma3TranslatesSandwichNormsAndWindows) {
  const auto config = TranslateFamilyConfig(
      Identity("gemma3_text"),
      Config("gemma3_text",
             "\"hidden_act\":\"gelu_pytorch_tanh\",\"attention_bias\":true,"
             "\"query_pre_attn_scalar\":256,\"sliding_window\":1024,"
             "\"sliding_window_pattern\":2,\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":8.0},"
             "\"rope_local_base_freq\":10000"));
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_NEAR(config->embedding_scale, std::sqrt(32.0f), 1e-4f);
  const auto& first = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  const auto& second = std::get<components::AttentionConfig>(config->blocks[1].mixer);
  EXPECT_EQ(first.sliding_window, 1024);      // (0 + 1) % 2 != 0 -> local.
  EXPECT_EQ(second.sliding_window, 0);        // (1 + 1) % 2 == 0 -> global.
  EXPECT_NEAR(first.scale_override, 1.0f / std::sqrt(256.0f), 1e-6f);
  EXPECT_EQ(first.rotary.type, "default");    // Local layers use the local base.
  EXPECT_EQ(second.rotary.type, "linear");    // Global layers scale by 8.
  const auto& ffn = std::get<components::SwiGluConfig>(config->blocks[0].feed_forward);
  EXPECT_EQ(ffn.activation, ops::Activation::kGeluTanh);
  EXPECT_EQ(config->blocks[0].residual, components::ResidualStyle::kOutputNorm);
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, MixtralTranslatesSoftmaxRoutedExperts) {
  const auto config =
      TranslateFamilyConfig(Identity("mixtral"),
                            Config("mixtral", "\"num_local_experts\":8,"
                                              "\"num_experts_per_tok\":2,\"norm_topk_prob\":true,"
                                              "\"sliding_window\":4096,\"use_sliding_window\":true"));
  ASSERT_TRUE(config.ok()) << config.status();
  const auto& moe = std::get<components::MoeConfig>(config->blocks[0].feed_forward);
  EXPECT_EQ(moe.num_experts, 8);
  EXPECT_EQ(moe.experts_per_token, 2);
  EXPECT_EQ(moe.routing.scoring, ops::RouterScoring::kSoftmaxTopkRenorm);
  EXPECT_TRUE(moe.routing.normalize);
  EXPECT_EQ(moe.shared_intermediate_size, 0);
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, GptOssTranslatesSinksAndFusedExperts) {
  const auto config = TranslateFamilyConfig(
      Identity("gpt_oss"),
      Config("gpt_oss",
             "\"num_local_experts\":32,\"num_experts_per_tok\":4,\"swiglu_limit\":7.0,"
             "\"attention_bias\":true,\"sliding_window\":128,\"rms_norm_eps\":0.00001,"
             "\"layer_types\":[\"sliding_attention\",\"full_attention\"]"));
  ASSERT_TRUE(config.ok()) << config.status();
  const auto& first = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  EXPECT_TRUE(first.sinks);
  EXPECT_EQ(first.sliding_window, 128);
  const auto& moe = std::get<components::MoeConfig>(config->blocks[0].feed_forward);
  EXPECT_TRUE(moe.has_router_bias);
  EXPECT_TRUE(moe.fused_mxfp4_experts);
  EXPECT_EQ(moe.activation, ops::Activation::kSiluOai);
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, DeepseekV3TranslatesMlaAndGroupedRouting) {
  const auto config = TranslateFamilyConfig(
      Identity("deepseek_v3"),
      Config("deepseek_v3",
             "\"q_lora_rank\":64,\"kv_lora_rank\":32,\"qk_nope_head_dim\":32,"
             "\"qk_rope_head_dim\":16,\"v_head_dim\":24,\"first_k_dense_replace\":1,"
             "\"n_routed_experts\":16,\"num_experts_per_tok\":4,\"n_group\":4,"
             "\"topk_group\":2,\"norm_topk_prob\":true,\"routed_scaling_factor\":2.5,"
             "\"moe_intermediate_size\":48"));
  ASSERT_TRUE(config.ok()) << config.status();
  const auto& mla = std::get<components::MlaConfig>(config->blocks[0].mixer);
  EXPECT_EQ(mla.head_dim(), 48);
  EXPECT_EQ(mla.rotary.dim, 16);
  EXPECT_TRUE(std::holds_alternative<components::SwiGluConfig>(
      config->blocks[0].feed_forward));  // Dense prefix.
  const auto& moe = std::get<components::MoeConfig>(config->blocks[1].feed_forward);
  EXPECT_EQ(moe.routing.scoring, ops::RouterScoring::kSigmoidGroupTopk);
  EXPECT_EQ(moe.routing.group_count, 4);
  EXPECT_EQ(moe.routing.group_topk, 2);
  EXPECT_NEAR(moe.routing.routing_scale, 2.5f, 1e-6f);
  EXPECT_TRUE(moe.has_correction_bias);
  EXPECT_TRUE(config->ValidateExecutable().ok());
}

TEST(FamilyConfigTest, RealGptOssConfigTranslatesWhenPresent) {
  std::ifstream config_file("models/gpt-oss-20b/config.json");
  if (!config_file.good()) GTEST_SKIP() << "local gpt-oss checkpoint not present";
  std::string json((std::istreambuf_iterator<char>(config_file)),
                   std::istreambuf_iterator<char>());
  const auto config = TranslateFamilyConfig(Identity("gpt_oss"), json);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->model.num_hidden_layers, 24);
  const auto& moe = std::get<components::MoeConfig>(config->blocks[0].feed_forward);
  EXPECT_EQ(moe.num_experts, 32);
  EXPECT_EQ(moe.experts_per_token, 4);
  EXPECT_EQ(moe.oai_limit, 7.0f);
  const auto& a = std::get<components::AttentionConfig>(config->blocks[0].mixer);
  EXPECT_TRUE(a.sinks);
  EXPECT_EQ(a.sliding_window, 128);  // Layer 0 is sliding_attention.
  EXPECT_EQ(a.rotary.type, "yarn");
  EXPECT_NEAR(a.rotary.factor, 32.0, 1e-6);
  EXPECT_TRUE(config->ValidateExecutable().ok()) << config->ValidateExecutable().ToString();
}

TEST(FamilyConfigTest, TypedBuilderValidatesThenFailsAtMissingWeights) {
  for (const auto& json : {Config("qwen3_next",
                                  "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
                                  "\"full_attention_interval\":2,"
                                  "\"linear_num_key_heads\":2,\"linear_num_value_heads\":2,"
                                  "\"linear_key_head_dim\":64,\"linear_value_head_dim\":64,"
                                  "\"linear_conv_kernel_dim\":4,"
                                  "\"num_experts\":4,\"num_experts_per_tok\":2,"
                                  "\"moe_intermediate_size\":32,"
                                  "\"shared_expert_intermediate_size\":32")}) {
    models::LoadedCheckpoint checkpoint;
    auto identity = CheckpointConfig::FromJson(json);
    ASSERT_TRUE(identity.ok());
    checkpoint.config = *identity;
    checkpoint.config_json = json;
    // No weight files are opened: with every block now executable, the
    // build must pass validation and fail at the (absent) weights instead.
    const auto model = BuildModel(checkpoint, DeviceId::Cpu(), 8, 2);
    EXPECT_EQ(model.status().code(), absl::StatusCode::kNotFound);
  }
}

}  // namespace
}  // namespace inferx
