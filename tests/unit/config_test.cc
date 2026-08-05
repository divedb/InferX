#include "inferx/model/config.h"

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

namespace inferx::model {
namespace {

// The real Qwen2.5-3B-Instruct config, verbatim.
constexpr char kQwen3B[] = R"({
  "architectures": ["Qwen2ForCausalLM"],
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "intermediate_size": 11008,
  "max_position_embeddings": 32768,
  "model_type": "qwen2",
  "num_attention_heads": 16,
  "num_hidden_layers": 36,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "vocab_size": 151936
})";

TEST(ModelConfig, ParsesQwen25) {
  auto c = ModelConfig::FromJson(kQwen3B);
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_EQ(c->architecture, Architecture::kQwen2);
  EXPECT_EQ(c->hidden_size, 2048);
  EXPECT_EQ(c->intermediate_size, 11008);
  EXPECT_EQ(c->num_hidden_layers, 36);
  EXPECT_EQ(c->num_attention_heads, 16);
  EXPECT_EQ(c->num_key_value_heads, 2);
  EXPECT_EQ(c->vocab_size, 151936);
  EXPECT_TRUE(c->tie_word_embeddings);
  EXPECT_EQ(c->weight_dtype, DataType::kBFloat16);

  // head_dim is absent from this config and must come out as hidden/heads.
  EXPECT_EQ(c->head_dim, 128);

  // GQA: 16 Q heads over 2 KV heads.
  EXPECT_EQ(c->gqa_group_size(), 8);
  EXPECT_EQ(c->q_dim(), 2048);
  EXPECT_EQ(c->kv_dim(), 256);  // matches k_proj's [256, 2048] in the checkpoint

  EXPECT_DOUBLE_EQ(c->rope_theta, 1e6);
  EXPECT_DOUBLE_EQ(c->rms_norm_eps, 1e-6);

  // Qwen2 biases Q/K/V without saying so in the config.
  EXPECT_TRUE(c->attention_bias);
}

TEST(ModelConfig, LlamaDefaultsDifferFromQwen) {
  auto c = ModelConfig::FromJson(R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096, "intermediate_size": 11008,
    "num_hidden_layers": 32, "num_attention_heads": 32,
    "vocab_size": 32000, "rms_norm_eps": 1e-05
  })");
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_EQ(c->architecture, Architecture::kLlama);
  // No attention bias on Llama, unlike Qwen2.
  EXPECT_FALSE(c->attention_bias);
  // No num_key_value_heads means MHA, not GQA.
  EXPECT_EQ(c->num_key_value_heads, 32);
  EXPECT_EQ(c->gqa_group_size(), 1);
  // rope_theta defaults to 10000 when unstated.
  EXPECT_DOUBLE_EQ(c->rope_theta, 10000.0);
}

TEST(ModelConfig, ExplicitHeadDimWins) {
  // hidden/heads would be 64; the file says 96 and must be believed.
  auto c = ModelConfig::FromJson(R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 1024, "intermediate_size": 2048,
    "num_hidden_layers": 4, "num_attention_heads": 16,
    "head_dim": 96, "vocab_size": 1000
  })");
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_EQ(c->head_dim, 96);
  EXPECT_EQ(c->q_dim(), 1536);
}

TEST(ModelConfig, RejectsUnknownArchitecture) {
  const auto c = ModelConfig::FromJson(R"({
    "architectures": ["MambaForCausalLM"],
    "hidden_size": 1024, "intermediate_size": 2048,
    "num_hidden_layers": 4, "num_attention_heads": 16, "vocab_size": 1000
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelConfig, RejectsNonIntegralGqaRatio) {
  const auto c = ModelConfig::FromJson(R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 1024, "intermediate_size": 2048,
    "num_hidden_layers": 4, "num_attention_heads": 10,
    "num_key_value_heads": 4, "vocab_size": 1000
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_NE(c.status().message().find("GQA group size"), std::string_view::npos)
      << c.status();
}

TEST(ModelConfig, RejectsOddHeadDim) {
  const auto c = ModelConfig::FromJson(R"({
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 1024, "intermediate_size": 2048,
    "num_hidden_layers": 4, "num_attention_heads": 16,
    "head_dim": 33, "vocab_size": 1000
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_NE(c.status().message().find("RoPE"), std::string_view::npos)
      << c.status();
}

TEST(ModelConfig, RejectsMissingRequiredFields) {
  EXPECT_FALSE(ModelConfig::FromJson(R"({"architectures":["LlamaForCausalLM"]})").ok());
  EXPECT_FALSE(ModelConfig::FromJson("{}").ok());
  EXPECT_FALSE(ModelConfig::FromJson("[]").ok());
  EXPECT_FALSE(ModelConfig::FromJson("not json").ok());
}

// --- MoE and MLA (M9) --------------------------------------------------------

TEST(ModelConfig, DenseCheckpointsAreNeitherMoeNorMla) {
  // The fields are read unconditionally, so the thing to check is that a
  // checkpoint that mentions none of them is still dense rather than a MoE
  // model with zero experts.
  auto c = ModelConfig::FromJson(kQwen3B);
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_FALSE(c->is_moe());
  EXPECT_FALSE(c->is_mla());
  EXPECT_EQ(c->num_experts, 0);
  EXPECT_EQ(c->KvElementsPerTokenPerLayer(), 2 * 2 * 128);
}

TEST(ModelConfig, ParsesAQwen2MoeCheckpoint) {
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["Qwen2MoeForCausalLM"],
      "hidden_size": 2048, "intermediate_size": 5632,
      "num_hidden_layers": 24, "num_attention_heads": 16,
      "num_key_value_heads": 16, "vocab_size": 151936,
      "num_experts": 60, "num_experts_per_tok": 4,
      "moe_intermediate_size": 1408, "norm_topk_prob": false,
      "shared_expert_intermediate_size": 5632
  })");
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_EQ(c->architecture, Architecture::kQwen2Moe);
  EXPECT_TRUE(c->is_moe());
  EXPECT_EQ(c->num_experts, 60);
  EXPECT_EQ(c->num_experts_per_tok, 4);
  EXPECT_EQ(c->moe_intermediate_size, 1408);
  EXPECT_EQ(c->shared_expert_intermediate_size, 5632);

  // Read, not assumed: this checkpoint turns renormalization off.
  EXPECT_FALSE(c->norm_topk_prob);

  // The expert width is the narrow one; the dense intermediate_size still
  // describes the shared expert and must not be overwritten by it.
  EXPECT_EQ(c->intermediate_size, 5632);
}

TEST(ModelConfig, RejectsRoutingToMoreExpertsThanExist) {
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["Qwen2MoeForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "num_experts": 4, "num_experts_per_tok": 8, "moe_intermediate_size": 32
  })");

  EXPECT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ModelConfig, RejectsExpertsNobodyRoutesTo) {
  // num_experts without num_experts_per_tok would run as a dense model while
  // the checkpoint carries expert weights -- plausible output, silently wrong.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["Qwen2MoeForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100, "num_experts": 8
  })");

  EXPECT_FALSE(c.ok());
}

TEST(ModelConfig, ParsesMlaHeadDimensions) {
  // DeepSeek-V2-Lite's shape.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 2048, "intermediate_size": 10944,
      "num_hidden_layers": 27, "num_attention_heads": 16,
      "num_key_value_heads": 16, "vocab_size": 102400,
      "kv_lora_rank": 512, "q_lora_rank": 0,
      "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128
  })");
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_TRUE(c->is_mla());
  EXPECT_EQ(c->kv_lora_rank, 512);
  EXPECT_EQ(c->qk_rope_head_dim, 64);

  // The point of MLA, in one number: 576 elements per token per layer instead
  // of the 4096 the same head count would cost as GQA.
  EXPECT_EQ(c->KvElementsPerTokenPerLayer(), 576);
}

TEST(ModelConfig, RejectsMlaHeadDimensionsWithoutALatent) {
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16
  })");

  EXPECT_FALSE(c.ok());
}

TEST(ModelConfig, RejectsALatentNarrowerThanTheHeadsItReconstructs) {
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "kv_lora_rank": 8,
      "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16
  })");

  EXPECT_FALSE(c.ok());
}

TEST(ModelConfig, ReadsTheRealCheckpointDirectory) {
  const char* home = std::getenv("HOME");
  if (home == nullptr) GTEST_SKIP() << "no HOME";

  std::string dir;
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) {
    dir = env;
  } else {
    dir = std::string(home) +
          "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
          "aa8e72537993ba99e69dfaafa59ed015b17504d1";
  }

  auto c = ModelConfig::FromDirectory(dir);
  if (!c.ok()) GTEST_SKIP() << "no checkpoint at " << dir;

  EXPECT_EQ(c->architecture, Architecture::kQwen2);
  EXPECT_EQ(c->hidden_size, 2048);
  EXPECT_EQ(c->num_hidden_layers, 36);
  EXPECT_EQ(c->kv_dim(), 256);
  EXPECT_TRUE(c->tie_word_embeddings);
}

}  // namespace
}  // namespace inferx::model
