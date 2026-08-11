#include "inferx/model/parallel/tp_layout.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "inferx/model/parallel/partition.h"
#include "inferx/model/parallel/tp_dims.h"

namespace inferx::model::parallel {
namespace {

using ::testing::HasSubstr;

ModelConfig Qwen2Config() {
  ModelConfig config;
  config.num_attention_heads = 16;
  config.num_key_value_heads = 4;
  config.head_dim = 8;
  config.intermediate_size = 64;
  return config;
}

ModelConfig GptOssConfig() {
  ModelConfig config;
  config.num_attention_heads = 64;
  config.num_key_value_heads = 8;
  config.head_dim = 64;
  config.moe_intermediate_size = 2880;
  return config;
}

TEST(TpLayoutTest, DescribesQwen2CheckpointSharding) {
  const TpLayout layout = Qwen2TpLayout(Qwen2Config());

  EXPECT_EQ(layout.SpecFor("model.layers.3.self_attn.q_proj.weight"),
            (ShardSpec{Partition::kRows, 8}));
  EXPECT_EQ(layout.SpecFor("model.layers.3.self_attn.o_proj.weight"),
            (ShardSpec{Partition::kCols, 8}));
  EXPECT_EQ(layout.SpecFor("model.layers.3.mlp.down_proj.weight"),
            (ShardSpec{Partition::kCols, 1}));
  EXPECT_EQ(layout.SpecFor("model.layers.3.input_layernorm.weight"),
            (ShardSpec{Partition::kReplicated, 1}));
  EXPECT_EQ(layout.kv_sharding(), KvSharding::kHeads);
  EXPECT_EQ(layout.collective_schedule(),
            (std::vector{CollectivePoint::kAfterAttentionOutput,
                         CollectivePoint::kAfterFfnOutput}));
}

TEST(TpLayoutTest, ValidatesTopologyAgainstModelDimensions) {
  const ModelConfig config = Qwen2Config();
  const TpLayout layout = Qwen2TpLayout(config);

  EXPECT_TRUE(layout.Validate(config, 1).ok());
  EXPECT_TRUE(layout.Validate(config, 2).ok());

  const Status invalid = layout.Validate(config, 3);
  EXPECT_EQ(invalid.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(invalid.message()), HasSubstr("attention heads 16"));
}

TEST(TpDimsTest, DerivesEveryQwen2LocalDimensionOnce) {
  const ModelConfig config = Qwen2Config();
  const TpLayout layout = Qwen2TpLayout(config);

  const auto dims = TpDims::For(config, layout, 2);
  ASSERT_TRUE(dims.ok()) << dims.status();
  EXPECT_EQ(dims->local_heads, 8);
  EXPECT_EQ(dims->local_kv_heads, 2);
  EXPECT_EQ(dims->local_q_dim, 64);
  EXPECT_EQ(dims->local_kv_dim, 16);
  EXPECT_EQ(dims->local_intermediate, 32);
}

TEST(ShardSpecTest, ResolvesNestedExpertAxes) {
  EXPECT_EQ(*ResolveShardAxis({Partition::kRows, 1}, 2), 0);
  EXPECT_EQ(*ResolveShardAxis({Partition::kCols, 1}, 2), 1);
  EXPECT_EQ(*ResolveShardAxis({Partition::kRows, 1, 1}, 3), 1);
  EXPECT_EQ(*ResolveShardAxis({Partition::kCols, 1, 2}, 3), 2);

  const auto invalid = ResolveShardAxis({Partition::kRows, 1, 3}, 3);
  ASSERT_FALSE(invalid.ok());
  EXPECT_EQ(invalid.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TpLayoutTest, DescribesGptOssAttentionAndPackedExperts) {
  const ModelConfig config = GptOssConfig();
  const TpLayout layout = GptOssTpLayout(config);

  EXPECT_EQ(layout.SpecFor("model.layers.2.self_attn.q_proj.weight"),
            (ShardSpec{Partition::kRows, 64}));
  EXPECT_EQ(layout.SpecFor("model.layers.2.self_attn.o_proj.weight"),
            (ShardSpec{Partition::kCols, 64}));
  EXPECT_EQ(layout.SpecFor("model.layers.2.self_attn.o_proj.bias"),
            (ShardSpec{Partition::kReplicated, 1}));
  EXPECT_EQ(layout.SpecFor("model.layers.2.mlp.experts.gate_up_proj_blocks"),
            (ShardSpec{Partition::kRows, 1, 1, 2}));
  EXPECT_EQ(layout.SpecFor("model.layers.2.mlp.experts.down_proj_blocks"),
            (ShardSpec{Partition::kCols, 16, 2}));
  EXPECT_EQ(layout.SpecFor("model.layers.2.mlp.experts.down_proj_bias"),
            (ShardSpec{Partition::kReplicated, 1}));
  EXPECT_EQ(layout.kv_sharding(), KvSharding::kHeads);
}

TEST(TpLayoutTest, ValidatesGptOssMxfp4ShardAlignment) {
  ModelConfig config = GptOssConfig();
  const TpLayout layout = GptOssTpLayout(config);
  EXPECT_TRUE(layout.Validate(config, 2).ok());

  config.moe_intermediate_size = 2864;
  const Status invalid = layout.Validate(config, 2);
  EXPECT_EQ(invalid.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(std::string(invalid.message()),
              HasSubstr("MoE intermediate size"));
  EXPECT_THAT(std::string(invalid.message()), HasSubstr("unit 32"));
}

TEST(TpDimsTest, DerivesGptOssLocalExpertWidth) {
  const ModelConfig config = GptOssConfig();
  const auto dims = TpDims::For(config, GptOssTpLayout(config), 2);
  ASSERT_TRUE(dims.ok()) << dims.status();
  EXPECT_EQ(dims->local_heads, 32);
  EXPECT_EQ(dims->local_kv_heads, 4);
  EXPECT_EQ(dims->local_moe_intermediate, 1440);
}

}  // namespace
}  // namespace inferx::model::parallel
