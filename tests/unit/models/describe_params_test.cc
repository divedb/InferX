// Unit tests for Model::DescribeParams: the declarative parameter manifest
// (docs/executor.md §7.1) checked against a real Qwen3 model built on CPU
// -- names, global shapes, shard specs (incl. KV replication), pipeline
// ownership, dst engagement, and consistency with the layer tensors that
// BindWeights fills.

#include "inferx/dist/parallel_context.h"
#include "inferx/dist/shard.h"
#include "inferx/dist/topology.h"
#include "inferx/models/model.h"
#include "inferx/models/qwen3/qwen3_causal_lm.h"

#include <gtest/gtest.h>

#include <map>
#include <string>
#include <vector>

namespace inferx {
namespace {

constexpr int64_t kHidden = 64;
constexpr int64_t kInter = 128;
constexpr int64_t kVocab = 256;
constexpr int64_t kLayers = 2;
constexpr int64_t kHeads = 4;   // 32 -> local 8 at tp 4 in the doc example.
constexpr int64_t kKvHeads = 2;
constexpr int64_t kDim = 128;

models::qwen3::Qwen3Config MakeConfig() {
  models::qwen3::Qwen3Config config;
  config.hidden_size = kHidden;
  config.intermediate_size = kInter;
  config.vocab_size = kVocab;
  config.num_hidden_layers = kLayers;
  config.num_attention_heads = kHeads;
  config.num_key_value_heads = kKvHeads;
  config.head_dim = kDim;
  config.max_position_embeddings = 512;
  config.rms_norm_eps = 1e-5f;
  config.tie_word_embeddings = false;
  config.eos_token_ids = {1};
  return config;
}

struct Fixture {
  ParallelConfig parallel;
  std::unique_ptr<models::qwen3::Qwen3ForCausalLM> model;

  static StatusOr<Fixture> Make(int tp) {
    Fixture fixture;
    fixture.parallel.tensor_parallel_size = tp;
    fixture.parallel.tensor_parallel_rank = 0;
    INFERX_ASSIGN_OR_RETURN(
        fixture.model, models::qwen3::Qwen3ForCausalLM::Create(MakeConfig(), fixture.parallel,
                                                       DataType::kBFloat16, Device::Cpu()));
    return fixture;
  }
};

/// Index of the manifest by checkpoint name.
std::map<std::string, ParamDecl> ByName(const std::vector<ParamDecl>& params) {
  std::map<std::string, ParamDecl> index;
  for (const ParamDecl& p : params) index.emplace(std::string(p.name), p);
  return index;
}

ParallelContext ContextAt(const Topology& topology, int global) {
  return ParallelContext(topology, topology.Rank(global));
}

ParallelConfig Mesh1tp(int tp) {
  ParallelConfig cfg;
  cfg.tensor_parallel_size = tp;
  return cfg;
}

ClusterSpec Cluster1(int ranks) {
  ClusterSpec cluster;
  cluster.num_nodes = 1;
  cluster.gpus_per_node = ranks;
  return cluster;
}

TEST(DescribeParamsTest, ManifestCountsEveryCheckpointTensor) {
  const auto fixture = Fixture::Make(1);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const std::vector<ParamDecl> params = (*fixture).model->DescribeParams();

  // Per layer: 2 layernorms + 6 attention tensors + 3 MLP tensors = 11;
  // plus embedding, final norm, lm_head.
  ASSERT_EQ(params.size(), 2u * 11 + 3u);
  const auto index = ByName(params);
  EXPECT_TRUE(index.contains("model.embed_tokens.weight"));
  EXPECT_TRUE(index.contains("model.norm.weight"));
  EXPECT_TRUE(index.contains("lm_head.weight"));
  EXPECT_TRUE(index.contains("model.layers.1.self_attn.v_proj.weight"));
  EXPECT_TRUE(index.contains("model.layers.0.mlp.down_proj.weight"));
}

TEST(DescribeParamsTest, zzDump) {
  auto fixture = Fixture::Make(2);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  for (const ParamDecl& p : (*fixture).model->DescribeParams()) {
    std::printf("%s owner=%d layer=%d dst=%d\n", std::string(p.name).c_str(),
                static_cast<int>(p.owner), p.layer, p.dst.has_value() ? 1 : 0);
  }
}

TEST(DescribeParamsTest, GlobalShapesMatchTheConfig) {
  const auto fixture = Fixture::Make(2);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto index = ByName((*fixture).model->DescribeParams());

  EXPECT_EQ(index.at("model.embed_tokens.weight").global_shape,
            (ShardShape{kVocab, kHidden}));
  EXPECT_EQ(index.at("model.layers.0.self_attn.q_proj.weight").global_shape,
            (ShardShape{kHeads * kDim, kHidden}));
  EXPECT_EQ(index.at("model.layers.0.self_attn.k_proj.weight").global_shape,
            (ShardShape{kKvHeads * kDim, kHidden}));
  EXPECT_EQ(index.at("model.layers.0.self_attn.o_proj.weight").global_shape,
            (ShardShape{kHidden, kHeads * kDim}));
  EXPECT_EQ(index.at("model.layers.0.mlp.gate_proj.weight").global_shape,
            (ShardShape{kInter, kHidden}));
  EXPECT_EQ(index.at("model.layers.0.mlp.down_proj.weight").global_shape,
            (ShardShape{kHidden, kInter}));
  EXPECT_EQ(index.at("model.norm.weight").global_shape, (ShardShape{kHidden}));
}

TEST(DescribeParamsTest, LocalShapesMatchTheAllocatedTensors) {
  // The §7.1 contract: LocalShape(decl, ctx) equals the layer's allocated
  // tensor extents on every rank.
  const int tp = 2;
  const auto fixture = Fixture::Make(tp);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const std::vector<ParamDecl> params = (*fixture).model->DescribeParams();

  auto built = Topology::Build(Mesh1tp(tp), Cluster1(tp));
  ASSERT_TRUE(built.ok()) << built.status();

  for (int global = 0; global < tp; ++global) {
    const ParallelContext ctx = ContextAt(*built, global);
    for (const ParamDecl& p : params) {
      if (!p.dst.has_value()) continue;
      const ShardShape local = LocalShape(p, ctx);
      const Tensor& tensor = *p.dst;
      ASSERT_EQ(local[0], tensor.Dim(0))
          << p.name << " rank " << global << " dim 0: " << local[0] << " vs "
          << tensor.Dim(0);
      if (p.rank > 1) {
        ASSERT_EQ(local[1], tensor.Dim(1))
            << p.name << " rank " << global << " dim 1: " << local[1] << " vs "
            << tensor.Dim(1);
      }
    }
  }
}

TEST(DescribeParamsTest, KvReplicationFollowsHeadCounts) {
  // tp=4 over Hkv=2: replicate = 2, so K/V keep 256 global rows but each
  // rank holds 128 -- the §7.1 worked example's geometry at small scale.
  const int tp = 2;
  const auto fixture = Fixture::Make(tp);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto index = ByName((*fixture).model->DescribeParams());

  ASSERT_EQ(index.at("model.layers.0.self_attn.k_proj.weight").shard.split.replicate, 1);
  // Hkv (2) == tp (2): every rank holds a distinct KV head, replicate 1.
  // Bump to tp=4 > Hkv to see replication.
}

TEST(DescribeParamsTest, KvReplicateWhenTpExceedsHeads) {
  const int tp = 4;
  const auto fixture = Fixture::Make(tp);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto index = ByName((*fixture).model->DescribeParams());

  ASSERT_EQ(index.at("model.layers.0.self_attn.k_proj.weight").shard.split.replicate, 2);
  EXPECT_EQ(index.at("model.layers.0.self_attn.v_proj.weight").shard.split.replicate, 2);
  EXPECT_EQ(index.at("model.layers.0.self_attn.q_proj.weight").shard.split.replicate, 1);

  // LocalShape: K/V rows halve (2 distinct pieces of 2 ranks each); Q rows
  // quarter.
  auto built = Topology::Build(Mesh1tp(tp), Cluster1(tp));
  ASSERT_TRUE(built.ok()) << built.status();
  const ParallelContext ctx = ContextAt(*built, 2);
  const ParamDecl& k = index.at("model.layers.0.self_attn.k_proj.weight");
  EXPECT_EQ(LocalShape(k, ctx)[0], kKvHeads * kDim / 2);
  const ParamDecl& q = index.at("model.layers.0.self_attn.q_proj.weight");
  EXPECT_EQ(LocalShape(q, ctx)[0], kHeads * kDim / tp);
}

TEST(DescribeParamsTest, PipelineOwnershipAndTiedHead) {
  const auto fixture = Fixture::Make(1);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const std::vector<ParamDecl> params = (*fixture).model->DescribeParams();
  const auto index = ByName(params);

  EXPECT_EQ(index.at("model.embed_tokens.weight").owner, Owner::kFirstStage);
  EXPECT_EQ(index.at("lm_head.weight").owner, Owner::kLastStage);
  EXPECT_EQ(index.at("model.norm.weight").owner, Owner::kLastStage);
  EXPECT_EQ(index.at("model.layers.0.mlp.up_proj.weight").owner, Owner::kLayer);
  EXPECT_EQ(index.at("model.layers.1.self_attn.o_proj.weight").layer, 1);

  // OwnedHere over a pp=2 world: rank 0 owns the embedding, rank 1 the
  // head; layer 1 belongs only to stage 1.
  ParallelConfig mesh;
  mesh.tensor_parallel_size = 1;
  mesh.pipeline_parallel_size = 2;
  auto built = Topology::Build(mesh, Cluster1(2));
  ASSERT_TRUE(built.ok()) << built.status();
  const ParallelContext first = ContextAt(*built, 0);
  const ParallelContext last = ContextAt(*built, 1);

  const ParamDecl& embed = index.at("model.embed_tokens.weight");
  const ParamDecl& head = index.at("lm_head.weight");
  const ParamDecl& layer1 = index.at("model.layers.1.mlp.up_proj.weight");
  EXPECT_TRUE(OwnedHere(embed, first, kLayers));
  EXPECT_FALSE(OwnedHere(embed, last, kLayers));
  EXPECT_TRUE(OwnedHere(head, last, kLayers));
  EXPECT_FALSE(OwnedHere(layer1, first, kLayers));
  EXPECT_TRUE(OwnedHere(layer1, last, kLayers));
}

TEST(DescribeParamsTest, FusedBufferComponentsLeaveDstEmpty) {
  const auto fixture = Fixture::Make(2);
  ASSERT_TRUE(fixture.ok()) << fixture.status();
  const auto index = ByName((*fixture).model->DescribeParams());

  // Components of packed buffers carry no 1:1 destination.
  EXPECT_FALSE(index.at("model.layers.0.self_attn.q_proj.weight").dst.has_value());
  EXPECT_FALSE(index.at("model.layers.0.mlp.gate_proj.weight").dst.has_value());
  // 1:1 tensors do.
  EXPECT_TRUE(index.at("model.layers.0.self_attn.o_proj.weight").dst.has_value());
  EXPECT_TRUE(index.at("model.layers.0.mlp.down_proj.weight").dst.has_value());
  EXPECT_TRUE(index.at("model.layers.0.input_layernorm.weight").dst.has_value());
}

}  // namespace
}  // namespace inferx
