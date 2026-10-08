/// Architecture configuration and registry tests: each typed config parses
/// its checkpoint JSON, owns its defaults and validation, and composes an
/// executable stack; the registry maps checkpoint identity to factories
/// without ambiguity.

#include <cmath>
#include <fstream>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "inferx/core/status.h"
#include "inferx/models/architectures/deepseek_v3/model.h"
#include "inferx/models/architectures/gemma3/model.h"
#include "inferx/models/architectures/gpt_oss/model.h"
#include "inferx/models/architectures/llama/model.h"
#include "inferx/models/architectures/mistral/model.h"
#include "inferx/models/architectures/mixtral/model.h"
#include "inferx/models/architectures/qwen2/model.h"
#include "inferx/models/architectures/qwen3/model.h"
#include "inferx/models/architectures/qwen3_moe/model.h"
#include "inferx/models/architectures/qwen3_next/model.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/models/registry.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/moe.h"
#include "inferx/ops/rotary.h"
#include "inferx/util/config_view.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace {

namespace models = ::inferx::models;
namespace layers = ::inferx::models::layers;

/// Minimal dense-attention dimensions every architecture config shares.
constexpr const char* kBaseDims =
    "\"hidden_size\":32,\"intermediate_size\":64,\"num_hidden_layers\":2,"
    "\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":64,"
    "\"vocab_size\":100,\"max_position_embeddings\":128,"
    "\"rms_norm_eps\":0.00001,\"rope_theta\":10000.0,\"tie_word_embeddings\":false";

std::string Config(std::string_view model_type, std::string_view extra = "") {
  std::string json = std::string("{\"model_type\":\"") + model_type.data() + "\"";
  if (!extra.empty()) json += "," + std::string(extra.data());
  return json + "," + kBaseDims + "}";
}

/// Parses a JSON text into the ConfigView handed to architecture parsers.
struct ParsedConfig {
  nlohmann::json doc;
  ConfigView view;

  static StatusOr<ParsedConfig> Parse(const std::string& json) {
    nlohmann::json doc = nlohmann::json::parse(json, nullptr, /*allow_exceptions=*/false);
    if (doc.is_discarded() || !doc.is_object()) {
      return InvalidArgumentError("test config is not a JSON object");
    }
    return ParsedConfig{std::move(doc), ConfigView(nullptr, "")};
  }

  ConfigView operator()() const { return ConfigView(&doc, ""); }
};

TEST(ArchitectureConfigTest, LlamaComposesDenseStack) {
  auto parsed = ParsedConfig::Parse(Config("llama"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::llama::LlamaConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->hidden_size, 32);
  auto stack = models::llama::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  EXPECT_EQ(stack->blocks.size(), 2u);
  const auto& a = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  EXPECT_FALSE(a.qk_norm);
  EXPECT_EQ(a.query_heads, 4);
  EXPECT_EQ(a.kv_heads, 2);
  EXPECT_TRUE(std::holds_alternative<layers::GatedMlpConfig>(stack->blocks[0].feed_forward));
  const auto executable = stack->ValidateExecutable();
  EXPECT_TRUE(executable.ok()) << executable.ToString();
}

TEST(ArchitectureConfigTest, Qwen3ComposesQkNormStack) {
  auto parsed = ParsedConfig::Parse(Config("qwen3"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::qwen3::Qwen3Config::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::qwen3::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& a = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  EXPECT_TRUE(a.qk_norm);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, Qwen3MoeComposesRoutedStack) {
  auto parsed = ParsedConfig::Parse(Config(
      "qwen3_moe",
      "\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,"
      "\"shared_expert_intermediate_size\":32,\"norm_topk_prob\":true"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::qwen3_moe::Qwen3MoeConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::qwen3_moe::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& moe = std::get<layers::MoeConfig>(stack->blocks[0].feed_forward);
  EXPECT_EQ(moe.num_experts, 4);
  EXPECT_EQ(moe.shared_intermediate_size, 32);
  EXPECT_FALSE(moe.gate_shared_expert);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, Qwen3NextComposesHybridSchedule) {
  auto parsed = ParsedConfig::Parse(Config(
      "qwen3_next",
      "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
      "\"full_attention_interval\":2,"
      "\"linear_num_key_heads\":2,\"linear_num_value_heads\":2,"
      "\"linear_key_head_dim\":64,\"linear_value_head_dim\":64,"
      "\"linear_conv_kernel_dim\":4,"
      "\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32,"
      "\"shared_expert_intermediate_size\":32"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::qwen3_next::Qwen3NextConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::qwen3_next::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  EXPECT_TRUE(std::holds_alternative<layers::GatedDeltaNetConfig>(stack->blocks[0].mixer));
  const auto& full = std::get<layers::AttentionConfig>(stack->blocks[1].mixer);
  EXPECT_EQ(full.output_gate, layers::OutputGate::kSigmoid);
  EXPECT_TRUE(full.qk_norm_plus_one);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, Qwen25ComposesBiasedAttention) {
  auto parsed = ParsedConfig::Parse(
      Config("qwen2", "\"attention_bias\":true,\"hidden_act\":\"silu\""));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::qwen2::Qwen2Config::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::qwen2::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& a = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  EXPECT_TRUE(a.qkv_bias);
  EXPECT_FALSE(a.output_bias);  // Qwen2 biases QKV only, not o_proj.
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, MistralComposesSlidingWindow) {
  auto parsed = ParsedConfig::Parse(
      Config("mistral", "\"sliding_window\":4096,\"use_sliding_window\":true"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::mistral::MistralConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::mistral::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& a = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  EXPECT_EQ(a.sliding_window, 4096);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, Gemma3ComposesSandwichNormsAndWindows) {
  auto parsed = ParsedConfig::Parse(Config(
      "gemma3_text",
      "\"hidden_act\":\"gelu_pytorch_tanh\",\"attention_bias\":true,"
      "\"query_pre_attn_scalar\":256,\"sliding_window\":1024,"
      "\"sliding_window_pattern\":2,\"rope_scaling\":{\"rope_type\":\"linear\",\"factor\":8.0},"
      "\"rope_local_base_freq\":10000"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::gemma3::Gemma3Config::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::gemma3::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  EXPECT_NEAR(stack->embedding_scale, std::sqrt(32.0f), 1e-4f);
  const auto& first = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  const auto& second = std::get<layers::AttentionConfig>(stack->blocks[1].mixer);
  EXPECT_EQ(first.sliding_window, 1024);  // (0 + 1) % 2 != 0 -> local.
  EXPECT_EQ(second.sliding_window, 0);    // (1 + 1) % 2 == 0 -> global.
  EXPECT_NEAR(first.scale_override, 1.0f / std::sqrt(256.0f), 1e-6f);
  EXPECT_FLOAT_EQ(first.rotary.theta, 10000.0f);  // Local layers use the local base.
  EXPECT_EQ(second.rotary.scaling.flavor, ops::RopeFlavor::kLinear);  // Global scales by 8.
  EXPECT_FLOAT_EQ(second.rotary.scaling.factor, 8.0f);
  const auto& ffn = std::get<layers::GatedMlpConfig>(stack->blocks[0].feed_forward);
  EXPECT_EQ(ffn.activation, ops::Activation::kGeluTanh);
  EXPECT_EQ(stack->blocks[0].residual, layers::ResidualStyle::kOutputNorm);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, MixtralComposesSoftmaxRoutedExperts) {
  auto parsed = ParsedConfig::Parse(Config(
      "mixtral", "\"num_local_experts\":8,"
                 "\"num_experts_per_tok\":2,\"norm_topk_prob\":true,"
                 "\"sliding_window\":4096,\"use_sliding_window\":true"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::mixtral::MixtralConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::mixtral::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& moe = std::get<layers::MoeConfig>(stack->blocks[0].feed_forward);
  EXPECT_EQ(moe.num_experts, 8);
  EXPECT_EQ(moe.experts_per_token, 2);
  EXPECT_EQ(moe.routing.scoring, ops::RouterScoring::kSoftmaxTopkRenorm);
  EXPECT_TRUE(moe.routing.normalize);
  EXPECT_EQ(moe.shared_intermediate_size, 0);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, GptOssComposesSinksAndFusedExperts) {
  auto parsed = ParsedConfig::Parse(Config(
      "gpt_oss",
      "\"num_local_experts\":32,\"num_experts_per_tok\":4,\"swiglu_limit\":7.0,"
      "\"attention_bias\":true,\"sliding_window\":128,\"rms_norm_eps\":0.00001,"
      "\"layer_types\":[\"sliding_attention\",\"full_attention\"]"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::gpt_oss::GptOssConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::gpt_oss::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& first = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  EXPECT_TRUE(first.sinks);
  EXPECT_EQ(first.sliding_window, 128);
  const auto& second = std::get<layers::AttentionConfig>(stack->blocks[1].mixer);
  EXPECT_EQ(second.sliding_window, 0);
  const auto& moe = std::get<layers::MoeConfig>(stack->blocks[0].feed_forward);
  EXPECT_TRUE(moe.has_router_bias);
  EXPECT_TRUE(moe.fused_mxfp4_experts);
  EXPECT_EQ(moe.activation, ops::Activation::kSiluOai);
  EXPECT_NEAR(moe.oai_limit, 7.0f, 1e-6f);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, DeepseekV3ComposesMlaAndGroupedRouting) {
  auto parsed = ParsedConfig::Parse(Config(
      "deepseek_v3",
      "\"q_lora_rank\":64,\"kv_lora_rank\":32,\"qk_nope_head_dim\":32,"
      "\"qk_rope_head_dim\":16,\"v_head_dim\":24,\"first_k_dense_replace\":1,"
      "\"n_routed_experts\":16,\"num_experts_per_tok\":4,\"n_group\":4,"
      "\"topk_group\":2,\"norm_topk_prob\":true,\"routed_scaling_factor\":2.5,"
      "\"moe_intermediate_size\":48"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::deepseek_v3::DeepseekV3Config::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  auto stack = models::deepseek_v3::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& mla = std::get<layers::MlaConfig>(stack->blocks[0].mixer);
  EXPECT_EQ(mla.head_dim(), 48);
  EXPECT_EQ(mla.rotary.rotary_dim, 16);
  EXPECT_TRUE(std::holds_alternative<layers::GatedMlpConfig>(
      stack->blocks[0].feed_forward));  // Dense prefix.
  const auto& moe = std::get<layers::MoeConfig>(stack->blocks[1].feed_forward);
  EXPECT_EQ(moe.routing.scoring, ops::RouterScoring::kSigmoidGroupTopk);
  EXPECT_EQ(moe.routing.group_count, 4);
  EXPECT_EQ(moe.routing.group_topk, 2);
  EXPECT_NEAR(moe.routing.routing_scale, 2.5f, 1e-6f);
  EXPECT_TRUE(moe.has_correction_bias);
  EXPECT_TRUE(stack->ValidateExecutable().ok());
}

TEST(ArchitectureConfigTest, MissingRequiredFieldsErrorWithTheFieldPath) {
  auto parsed = ParsedConfig::Parse(Config("qwen3_moe"));  // No expert keys.
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const auto config = models::qwen3_moe::Qwen3MoeConfig::Parse((*parsed)());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(config.status().message().find("num_experts"), std::string::npos);
}

TEST(ArchitectureConfigTest, MalformedFieldsReturnStatusInsteadOfThrowing) {
  auto parsed = ParsedConfig::Parse(Config("qwen3", "\"attention_bias\":\"invalid\""));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(models::qwen3::Qwen3Config::Parse((*parsed)()).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_FALSE(ParsedConfig::Parse("[]").ok());
  EXPECT_FALSE(ParsedConfig::Parse("not json").ok());
}

TEST(ArchitectureConfigTest, DenseQwen3RejectsHeterogeneousLayerTypes) {
  auto parsed = ParsedConfig::Parse(
      Config("qwen3", "\"layer_types\":[\"linear_attention\",\"full_attention\"]"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  const auto config = models::qwen3::Qwen3Config::Parse((*parsed)());
  EXPECT_EQ(config.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ArchitectureConfigTest, Qwen3NextWithoutLinearDimsIsRejected) {
  auto parsed = ParsedConfig::Parse(Config(
      "qwen3_next",
      "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
      "\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":32"));
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  // The linear attention layer needs its head and kernel dimensions.
  EXPECT_FALSE(models::qwen3_next::Qwen3NextConfig::Parse((*parsed)()).ok());
}

/// Factories must remain reachable through the registry in static builds.
TEST(ArchitectureRegistryTest, BuiltinsResolve) {
  for (const char* type : {"llama", "qwen2", "qwen3", "qwen3_moe", "qwen3_next",
                           "mistral", "gemma3_text", "mixtral", "gpt_oss", "deepseek_v3"}) {
    const auto entry = models::ArchitectureRegistry::Builtins().Resolve(type, "");
    ASSERT_TRUE(entry.ok()) << type << ": " << entry.status();
    EXPECT_NE((*entry)->build, nullptr);
  }
}

TEST(ArchitectureRegistryTest, ArchitectureAliasesSelectTheSameFactory) {
  const auto& registry = models::ArchitectureRegistry::Builtins();
  for (const char* type : {"llama", "qwen3", "qwen3_moe", "qwen3_next"}) {
    const auto by_type = registry.Resolve(type, "");
    ASSERT_TRUE(by_type.ok()) << type;
    const auto by_arch = registry.Resolve("", (*by_type)->architecture);
    ASSERT_TRUE(by_arch.ok()) << by_arch.status();
    EXPECT_EQ(*by_arch, *by_type);
  }
}

TEST(ArchitectureRegistryTest, ConflictingIdentitiesAreRejected) {
  const auto resolved = models::ArchitectureRegistry::Builtins().Resolve("llama",
                                                                         "Qwen3ForCausalLM");
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ArchitectureRegistryTest, UnknownModelTypeIsRejected) {
  const auto resolved = models::ArchitectureRegistry::Builtins().Resolve("flux", "");
  EXPECT_EQ(resolved.status().code(), absl::StatusCode::kUnimplemented);
  EXPECT_NE(resolved.status().message().find("supported"), std::string::npos);
}

TEST(ArchitectureRegistryTest, AmbiguousRegistrationIsRejected) {
  models::ArchitectureRegistry registry;
  models::RegisterArchitecture<models::llama::LlamaForCausalLM>(registry, "LlamaForCausalLM",
                                                                "llama");
  const auto duplicate = registry.Register(
      models::ArchitectureEntry{"LlamaForCausalLM", "other", nullptr});
  EXPECT_FALSE(duplicate.ok());
  const auto model_type_conflict = registry.Register(
      models::ArchitectureEntry{"OtherForCausalLM", "llama", nullptr});
  EXPECT_FALSE(model_type_conflict.ok());
}

TEST(ArchitectureConfigTest, RealGptOssConfigParsesWhenPresent) {
  std::ifstream config_file("models/gpt-oss-20b/config.json");
  if (!config_file.good()) GTEST_SKIP() << "local gpt-oss checkpoint not present";
  std::string json((std::istreambuf_iterator<char>(config_file)),
                   std::istreambuf_iterator<char>());
  auto parsed = ParsedConfig::Parse(json);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  auto config = models::gpt_oss::GptOssConfig::Parse((*parsed)());
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_EQ(config->num_hidden_layers, 24);
  auto stack = models::gpt_oss::BuildStackConfig(*config);
  ASSERT_TRUE(stack.ok()) << stack.status();
  const auto& moe = std::get<layers::MoeConfig>(stack->blocks[0].feed_forward);
  EXPECT_EQ(moe.num_experts, 32);
  EXPECT_EQ(moe.experts_per_token, 4);
  EXPECT_EQ(moe.oai_limit, 7.0f);
  const auto& a = std::get<layers::AttentionConfig>(stack->blocks[0].mixer);
  EXPECT_TRUE(a.sinks);
  EXPECT_EQ(a.sliding_window, 128);  // Layer 0 is sliding_attention.
  EXPECT_EQ(a.rotary.scaling.flavor, ops::RopeFlavor::kYarn);
  EXPECT_NEAR(a.rotary.scaling.factor, 32.0f, 1e-3f);
  EXPECT_TRUE(stack->ValidateExecutable().ok()) << stack->ValidateExecutable().ToString();
}

TEST(ArchitectureConfigTest, BuilderValidatesThenFailsAtMissingWeights) {
  // A config-only checkpoint directory: with every block executable, the
  // factory must pass validation and fail at the (absent) weights instead.
  const std::string dir = ::testing::TempDir() + "inferx-config-only";
  std::ofstream(dir + "/config.json")
      << Config("qwen3_next",
                "\"layer_types\":[\"linear_attention\",\"full_attention\"],"
                "\"full_attention_interval\":2,"
                "\"linear_num_key_heads\":2,\"linear_num_value_heads\":2,"
                "\"linear_key_head_dim\":64,\"linear_value_head_dim\":64,"
                "\"linear_conv_kernel_dim\":4,"
                "\"num_experts\":4,\"num_experts_per_tok\":2,"
                "\"moe_intermediate_size\":32,"
                "\"shared_expert_intermediate_size\":32");
  models::ModelLoadOptions options;
  options.directory = dir;
  options.device = DeviceId::Cpu();
  options.max_tokens = 8;
  options.max_seqs = 2;
  const auto model = models::LoadModel(options);
  EXPECT_EQ(model.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace inferx
