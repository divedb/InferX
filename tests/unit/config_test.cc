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

// The real DeepSeek-V2-Lite config, verbatim except for HF bookkeeping keys
// (use_cache, transformers_version, ...) the parser ignores anyway.
constexpr char kDeepSeekV2Lite[] = R"({
  "architectures": ["DeepseekV2ForCausalLM"],
  "attention_bias": false,
  "aux_loss_alpha": 0.001,
  "bos_token_id": 100000,
  "eos_token_id": 100001,
  "first_k_dense_replace": 1,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 10944,
  "kv_lora_rank": 512,
  "max_position_embeddings": 163840,
  "model_type": "deepseek_v2",
  "moe_intermediate_size": 1408,
  "moe_layer_freq": 1,
  "n_group": 1,
  "n_routed_experts": 64,
  "n_shared_experts": 2,
  "norm_topk_prob": false,
  "num_attention_heads": 16,
  "num_experts_per_tok": 6,
  "num_hidden_layers": 27,
  "num_key_value_heads": 16,
  "q_lora_rank": null,
  "qk_nope_head_dim": 128,
  "qk_rope_head_dim": 64,
  "rms_norm_eps": 1e-06,
  "rope_scaling": {
    "beta_fast": 32,
    "beta_slow": 1,
    "factor": 40,
    "mscale": 0.707,
    "mscale_all_dim": 0.707,
    "original_max_position_embeddings": 4096,
    "type": "yarn"
  },
  "rope_theta": 10000,
  "routed_scaling_factor": 1.0,
  "scoring_func": "softmax",
  "seq_aux": true,
  "tie_word_embeddings": false,
  "topk_group": 1,
  "topk_method": "greedy",
  "torch_dtype": "bfloat16",
  "v_head_dim": 128,
  "vocab_size": 102400
})";

TEST(ModelConfig, ParsesDeepSeekV2Lite) {
  auto c = ModelConfig::FromJson(kDeepSeekV2Lite);
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_EQ(c->architecture, Architecture::kDeepSeekV2);
  EXPECT_EQ(c->hidden_size, 2048);
  EXPECT_EQ(c->intermediate_size, 10944);
  EXPECT_EQ(c->num_hidden_layers, 27);
  EXPECT_EQ(c->num_attention_heads, 16);
  EXPECT_EQ(c->vocab_size, 102400);
  EXPECT_FALSE(c->tie_word_embeddings);
  EXPECT_FALSE(c->attention_bias);
  EXPECT_EQ(c->weight_dtype, DataType::kBFloat16);

  // MLA, including the JSON-null q_lora_rank meaning "no Q down-projection".
  EXPECT_TRUE(c->is_mla());
  EXPECT_EQ(c->kv_lora_rank, 512);
  EXPECT_EQ(c->q_lora_rank, 0);
  EXPECT_EQ(c->qk_nope_head_dim, 128);
  EXPECT_EQ(c->qk_rope_head_dim, 64);
  EXPECT_EQ(c->v_head_dim, 128);
  EXPECT_EQ(c->KvElementsPerTokenPerLayer(), 576);

  // MoE under DeepSeek's spellings: n_routed_experts, and n_shared_experts as
  // a count whose width is count x moe_intermediate_size.
  EXPECT_TRUE(c->is_moe());
  EXPECT_EQ(c->num_experts, 64);
  EXPECT_EQ(c->num_experts_per_tok, 6);
  EXPECT_EQ(c->moe_intermediate_size, 1408);
  EXPECT_EQ(c->shared_expert_intermediate_size, 2816);
  EXPECT_FALSE(c->norm_topk_prob);
  EXPECT_DOUBLE_EQ(c->routed_scaling_factor, 1.0);

  // Layer 0 is dense; every later layer is MoE.
  EXPECT_EQ(c->first_k_dense_replace, 1);
  EXPECT_EQ(c->moe_layer_freq, 1);
  EXPECT_FALSE(c->IsMoeLayer(0));
  EXPECT_TRUE(c->IsMoeLayer(1));
  EXPECT_TRUE(c->IsMoeLayer(26));
  EXPECT_FALSE(c->IsMoeLayer(27));  // out of range, not "the next layer"

  // YaRN under the `type` spelling, with DeepSeek's mscale pair.
  EXPECT_TRUE(c->is_yarn());
  EXPECT_DOUBLE_EQ(c->yarn_factor, 40.0);
  EXPECT_EQ(c->yarn_original_max_position, 4096);
  EXPECT_DOUBLE_EQ(c->yarn_mscale, 0.707);
  EXPECT_DOUBLE_EQ(c->yarn_mscale_all_dim, 0.707);
  EXPECT_EQ(c->max_position_embeddings, 163840);
}

TEST(ModelConfig, RejectsRopeScalingWithNeitherTypeKey) {
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "rope_scaling": {"factor": 8.0}
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ModelConfig, RejectsScoringFunctionsWeHaveNotImplemented) {
  // DeepSeek-V3's sigmoid scoring selects different experts than softmax; run
  // as softmax it routes wrongly with no error downstream.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "n_routed_experts": 8, "num_experts_per_tok": 2,
      "moe_intermediate_size": 32, "scoring_func": "sigmoid"
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelConfig, RejectsTopKMethodsWeHaveNotImplemented) {
  // DeepSeek-V2-236B's group_limited_greedy and V3's noaux_tc are not greedy
  // top-k over all experts.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "n_routed_experts": 8, "num_experts_per_tok": 2,
      "moe_intermediate_size": 32, "topk_method": "group_limited_greedy"
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelConfig, DeepSeekArchitectureRequiresMlaAndExperts) {
  // Without MLA fields, DeepseekV2ForCausalLM is a mis-read checkpoint.
  auto without_mla = ModelConfig::FromJson(R"({
      "architectures": ["DeepseekV2ForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "n_routed_experts": 8, "num_experts_per_tok": 2,
      "moe_intermediate_size": 32
  })");
  ASSERT_FALSE(without_mla.ok());
  EXPECT_NE(without_mla.status().message().find("kv_lora_rank"),
            std::string_view::npos)
      << without_mla.status();

  // Without experts it would silently run dense -- the exact failure the
  // n_routed_experts alias exists to prevent.
  auto without_moe = ModelConfig::FromJson(R"({
      "architectures": ["DeepseekV2ForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "kv_lora_rank": 32, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
      "v_head_dim": 16
  })");
  ASSERT_FALSE(without_moe.ok());
  EXPECT_NE(without_moe.status().message().find("n_routed_experts"),
            std::string_view::npos)
      << without_moe.status();
}

TEST(ModelConfig, RejectsAScheduleThatLeavesNoMoeLayer) {
  // first_k_dense_replace >= num_hidden_layers makes every layer dense while
  // still declaring experts -- the silent-dense failure, at one remove.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "n_routed_experts": 8, "num_experts_per_tok": 2,
      "moe_intermediate_size": 32, "first_k_dense_replace": 2
  })");

  ASSERT_FALSE(c.ok());
  EXPECT_NE(c.status().message().find("no MoE layer"), std::string_view::npos)
      << c.status();
}

TEST(ModelConfig, ParsesGptOss) {
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["GptOssForCausalLM"],
      "hidden_size": 2880, "intermediate_size": 2880,
      "num_hidden_layers": 4, "num_attention_heads": 64,
      "num_key_value_heads": 8, "head_dim": 64, "vocab_size": 201088,
      "num_local_experts": 32, "num_experts_per_tok": 4,
      "swiglu_limit": 7.0, "sliding_window": 128, "rope_theta": 150000,
      "rms_norm_eps": 1e-5, "attention_bias": true,
      "layer_types": ["sliding_attention", "full_attention",
                      "sliding_attention", "full_attention"],
      "rope_scaling": {"rope_type": "yarn", "factor": 32.0,
                       "beta_fast": 32.0, "beta_slow": 1.0,
                       "original_max_position_embeddings": 4096,
                       "truncate": false}
  })");
  ASSERT_TRUE(c.ok()) << c.status();

  EXPECT_EQ(c->architecture, Architecture::kGptOss);
  EXPECT_TRUE(c->is_moe());

  // num_local_experts is gpt-oss's spelling; nothing else uses it.
  EXPECT_EQ(c->num_experts, 32);
  EXPECT_EQ(c->num_experts_per_tok, 4);

  // No moe_intermediate_size in this config, so an expert is as wide as the
  // model's intermediate_size. That fallback is load-bearing here.
  EXPECT_EQ(c->moe_intermediate_size, 2880);

  EXPECT_DOUBLE_EQ(c->swiglu_limit, 7.0);
  EXPECT_EQ(c->sliding_window, 128);

  // Decoupled head_dim: 64 heads x 64 = 4096 against a hidden size of 2880.
  EXPECT_EQ(c->q_dim(), 4096);
  EXPECT_EQ(c->kv_dim(), 512);

  // The alternation is read, not assumed.
  ASSERT_EQ(c->layer_is_sliding.size(), 4u);
  EXPECT_TRUE(c->IsSlidingLayer(0));
  EXPECT_FALSE(c->IsSlidingLayer(1));
  EXPECT_TRUE(c->IsSlidingLayer(2));
  EXPECT_FALSE(c->IsSlidingLayer(3));

  EXPECT_TRUE(c->is_yarn());
  EXPECT_DOUBLE_EQ(c->yarn_factor, 32.0);
  EXPECT_EQ(c->yarn_original_max_position, 4096);
  EXPECT_FALSE(c->yarn_truncate);
}

TEST(ModelConfig, RejectsLayerTypesThatDoNotCoverTheModel) {
  // A truncated list would silently run the missing layers as full attention,
  // which is a plausible-looking model with the wrong receptive field.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["GptOssForCausalLM"],
      "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 4,
      "num_attention_heads": 4, "vocab_size": 100,
      "num_local_experts": 4, "num_experts_per_tok": 2,
      "sliding_window": 8,
      "layer_types": ["sliding_attention", "full_attention"]
  })");

  EXPECT_FALSE(c.ok());
}

TEST(ModelConfig, RejectsRopeScalingWeHaveNotImplemented) {
  // linear/dynamic/longrope each change the frequencies differently. Running
  // one as another gives a model with no long-range coherence and no error.
  auto c = ModelConfig::FromJson(R"({
      "architectures": ["LlamaForCausalLM"],
      "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 2,
      "num_attention_heads": 4, "vocab_size": 100,
      "rope_scaling": {"rope_type": "linear", "factor": 4.0}
  })");

  EXPECT_FALSE(c.ok());
  EXPECT_EQ(c.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelConfig, ReadsTheRealGptOssCheckpointDirectory) {
  const char* home = std::getenv("HOME");
  if (home == nullptr) GTEST_SKIP() << "no HOME";

  std::string dir;
  if (const char* env = std::getenv("INFERX_TEST_GPTOSS_CHECKPOINT")) {
    dir = env;
  } else {
    dir = std::string(home) +
          "/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
          "6cee5e81ee83917806bbde320786a8fb61efebee";
  }

  auto c = ModelConfig::FromDirectory(dir);
  if (!c.ok()) GTEST_SKIP() << "no gpt-oss checkpoint at " << dir;

  EXPECT_EQ(c->architecture, Architecture::kGptOss);
  EXPECT_EQ(c->num_hidden_layers, 24);
  EXPECT_EQ(c->num_experts, 32);
  EXPECT_EQ(c->num_experts_per_tok, 4);
  EXPECT_EQ(c->hidden_size, 2880);
  EXPECT_EQ(c->q_dim(), 4096);
  EXPECT_EQ(c->sliding_window, 128);
  EXPECT_TRUE(c->is_yarn());
  EXPECT_FALSE(c->tie_word_embeddings);

  ASSERT_EQ(c->layer_is_sliding.size(), 24u);
  EXPECT_TRUE(c->IsSlidingLayer(0));
  EXPECT_FALSE(c->IsSlidingLayer(23));
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
