#include <string>

#include "gtest/gtest.h"
#include "inferx/models/llama/llama_config.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/models/qwen3_moe/qwen3_moe_config.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace models {
namespace {

const char* kQwen3 = R"({
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "vocab_size": 1024,
  "hidden_size": 256,
  "num_hidden_layers": 4,
  "num_attention_heads": 8,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "intermediate_size": 512,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true,
  "eos_token_id": [151645, 151643],
  "torch_dtype": "bfloat16"
})";

TEST(Qwen3ConfigTest, ParsesAllFields) {
  auto config = HfConfig::FromJson(kQwen3);
  ASSERT_TRUE(config.ok()) << config.status();
  auto parsed = qwen3::Qwen3Config::FromHf(*config);
  ASSERT_TRUE(parsed.ok()) << parsed.status();

  EXPECT_EQ(parsed->vocab_size, 1024);
  EXPECT_EQ(parsed->hidden_size, 256);
  EXPECT_EQ(parsed->num_hidden_layers, 4);
  EXPECT_EQ(parsed->num_attention_heads, 8);
  EXPECT_EQ(parsed->num_key_value_heads, 4);
  EXPECT_EQ(parsed->head_dim, 64);
  EXPECT_EQ(parsed->intermediate_size, 512);
  EXPECT_TRUE(parsed->tie_word_embeddings);
  EXPECT_EQ(parsed->eos_token_ids, (std::vector<TokenId>{151645, 151643}));
  EXPECT_FLOAT_EQ(parsed->rms_norm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(parsed->rotary_params.theta, 1e6f);
  EXPECT_EQ(parsed->rotary_params.scaling.flavor, ops::RopeFlavor::kDefault);
  EXPECT_EQ(parsed->num_attention_heads % 2, 0);  // TP 2 shards evenly.
}

TEST(Qwen3ConfigTest, RejectsBrokenGeometry) {
  const std::string broken = R"({
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3", "vocab_size": 100, "hidden_size": 64,
    "num_hidden_layers": 1, "num_attention_heads": 6,
    "num_key_value_heads": 4, "intermediate_size": 32,
    "max_position_embeddings": 128
  })";
  auto config = HfConfig::FromJson(broken);
  ASSERT_TRUE(config.ok()) << config.status();
  EXPECT_FALSE(qwen3::Qwen3Config::FromHf(*config).ok());
}

TEST(Qwen3MoeConfigTest, ParsesExpertFields) {
  const std::string moe = std::string(R"({
    "architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe",
    "vocab_size": 1024, "hidden_size": 256, "num_hidden_layers": 4,
    "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 64,
    "intermediate_size": 512, "max_position_embeddings": 4096,
    "num_experts": 16, "num_experts_per_tok": 4, "norm_topk_prob": true,
    "shared_expert_intermediate_size": 256, "shared_expert_gate": true,
    "eos_token_id": 151645
  })");
  auto config = HfConfig::FromJson(moe);
  ASSERT_TRUE(config.ok()) << config.status();
  auto parsed = qwen3_moe::Qwen3MoeConfig::FromHf(*config);
  ASSERT_TRUE(parsed.ok()) << parsed.status();

  EXPECT_EQ(parsed->num_experts, 16);
  EXPECT_EQ(parsed->num_experts_per_tok, 4);
  EXPECT_TRUE(parsed->norm_topk_prob);
  EXPECT_EQ(parsed->shared_expert_intermediate, 256);
  EXPECT_TRUE(parsed->shared_expert_gate);
  EXPECT_EQ(parsed->base.hidden_size, 256);
  EXPECT_EQ(parsed->Routing().topk, 4);
  EXPECT_TRUE(parsed->Routing().normalize);
  EXPECT_EQ(parsed->base.eos_token_ids, std::vector<TokenId>{151645});
}

TEST(LlamaConfigTest, ParsesAndDefaults) {
  const std::string llama = R"({
    "architectures": ["LlamaForCausalLM"], "model_type": "llama",
    "vocab_size": 512, "hidden_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 4, "intermediate_size": 256,
    "max_position_embeddings": 2048, "rms_norm_eps": 1e-05,
    "rope_theta": 500000.0, "eos_token_id": 128001
  })";
  auto config = HfConfig::FromJson(llama);
  ASSERT_TRUE(config.ok()) << config.status();
  auto parsed = llama::LlamaConfig::FromHf(*config);
  ASSERT_TRUE(parsed.ok()) << parsed.status();

  // num_key_value_heads absent falls back to num_attention_heads (MQA off),
  // and head_dim derives from hidden/heads.
  EXPECT_EQ(parsed->num_key_value_heads, 4);
  EXPECT_EQ(parsed->head_dim, 32);
  EXPECT_FLOAT_EQ(parsed->rope.theta, 5e5f);
  EXPECT_FALSE(parsed->tie_word_embeddings);
}

TEST(LlamaConfigTest, ParsesLlama3Scaling) {
  const std::string llama = R"({
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama", "vocab_size": 512, "hidden_size": 128,
    "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 256,
    "max_position_embeddings": 131072, "head_dim": 64,
    "rope_scaling": {"rope_type": "llama3", "factor": 8.0,
                     "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                     "original_max_position_embeddings": 8192}
  })";
  auto config = HfConfig::FromJson(llama);
  ASSERT_TRUE(config.ok()) << config.status();
  auto parsed = llama::LlamaConfig::FromHf(*config);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ(parsed->rope.scaling.flavor, ops::RopeFlavor::kLlama3);
  EXPECT_FLOAT_EQ(parsed->rope.scaling.factor, 8.0f);
  EXPECT_FLOAT_EQ(parsed->rope.scaling.original_max_position, 8192.0f);
}

}  // namespace
}  // namespace models
}  // namespace inferx
