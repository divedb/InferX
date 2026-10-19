// Unit tests for declarative weight sharding (docs/executor.md §7.1):
// LocalShape as the single source of truth for split/replicate, fused, and
// expert specs, and OwnedHere for pipeline ownership.

#include "inferx/dist/shard.h"

#include <gtest/gtest.h>

#include <vector>

#include "inferx/dist/topology.h"

namespace inferx {
namespace {

ClusterSpec Cluster(int nodes, int gpus_per_node) {
  ClusterSpec cluster;
  cluster.num_nodes = nodes;
  cluster.gpus_per_node = gpus_per_node;
  return cluster;
}

ParallelConfig Mesh(int dp, int pp, int cp, int tp) {
  ParallelConfig cfg;
  cfg.data_parallel_size = dp;
  cfg.pipeline_parallel_size = pp;
  cfg.context_parallel_size = cp;
  cfg.tensor_parallel_size = tp;
  return cfg;
}

/// Builds the topology and one context per rank of the world.
struct World {
  std::unique_ptr<Topology> topology;
  std::vector<ParallelContext> contexts;

  static World Make(const ParallelConfig& cfg, int nodes, int gpus) {
    World world;
    auto built = Topology::Build(cfg, Cluster(nodes, gpus));
    if (!built.ok()) return world;
    world.topology = std::make_unique<Topology>(std::move(*built));
    for (int global = 0; global < cfg.WorldSize(); ++global) {
      world.contexts.emplace_back(*world.topology, world.topology->Rank(global));
    }
    return world;
  }
};

ShardShape Shape4(std::int64_t d0, std::int64_t d1 = 1, std::int64_t d2 = 1,
                  std::int64_t d3 = 1) {
  return ShardShape{d0, d1, d2, d3};
}

ParamDecl Decl(ShardShape global, ShardSpec shard) {
  ParamDecl p;
  p.name = "weight";
  p.global_shape = global;
  p.rank = 2;
  p.shard = std::move(shard);
  return p;
}

ShardSpec Split(int dim, Axis axis = Axis::kTensorParallel, int replicate = 1) {
  ShardSpec shard;
  shard.kind = ShardSpec::Kind::kSplit;
  shard.split = SplitSpec{axis, dim, replicate};
  return shard;
}

TEST(ShardTest, ReplicatedParametersKeepTheGlobalShape) {
  World world = World::Make(Mesh(2, 2, 1, 2), 1, 8);
  ASSERT_NE(world.topology, nullptr);
  const ParamDecl p = Decl(Shape4(512, 256), ShardSpec{});

  for (const ParallelContext& ctx : world.contexts) {
    EXPECT_EQ(LocalShape(p, ctx), Shape4(512, 256));
  }
}

TEST(ShardTest, EvenSplitDividesAcrossTheWholeAxis) {
  World world = World::Make(Mesh(1, 1, 1, 4), 1, 4);
  ASSERT_NE(world.topology, nullptr);
  const ParamDecl p = Decl(Shape4(4096, 256), Split(0));

  // Every rank holds 1/4 of the rows.
  for (const ParallelContext& ctx : world.contexts) {
    EXPECT_EQ(LocalShape(p, ctx), Shape4(1024, 256));
  }
}

TEST(ShardTest, UnevenSplitPutsTheRemainderUpFront) {
  World world = World::Make(Mesh(1, 1, 1, 3), 1, 3);
  ASSERT_NE(world.topology, nullptr);
  const ParamDecl p = Decl(Shape4(10, 2), Split(0));

  // Default rank order has tp innermost: ranks 0..2 are tp 0..2.
  ASSERT_EQ(world.contexts.size(), 3u);
  EXPECT_EQ(LocalShape(p, world.contexts[0]), Shape4(4, 2));  // 10 = 4 + 3 + 3.
  EXPECT_EQ(LocalShape(p, world.contexts[1]), Shape4(3, 2));
  EXPECT_EQ(LocalShape(p, world.contexts[2]), Shape4(3, 2));

  // The non-split dims pass through untouched.
  const ParamDecl wide = Decl(Shape4(10, 7, 3), Split(2));
  EXPECT_EQ(LocalShape(wide, world.contexts[2]), Shape4(10, 7, 1));
}

TEST(ShardTest, ReplicateShrinksTheDistinctPieceCount) {
  // tp = 4, replicate = 2: two distinct pieces, each held by two ranks.
  World world = World::Make(Mesh(1, 1, 1, 4), 1, 4);
  ASSERT_NE(world.topology, nullptr);
  const ParamDecl p = Decl(Shape4(256, 64), Split(0, Axis::kTensorParallel, 2));

  for (const ParallelContext& ctx : world.contexts) {
    EXPECT_EQ(LocalShape(p, ctx), Shape4(128, 64));
  }
}

TEST(ShardTest, FusedQkvMatchesTheWorkedExample) {
  // §7.1: tp = 4, Hq = 32, Hkv = 2, D = 128. Q 4096 split; K and V 256
  // each with replicate = 2. Every rank: 8*128 Q + 128 K + 128 V = 1280.
  World world = World::Make(Mesh(1, 1, 1, 4), 1, 4);
  ASSERT_NE(world.topology, nullptr);

  static constexpr FusedSegment kSegments[] = {
      FusedSegment{4096, false, 1},
      FusedSegment{256, false, 2},
      FusedSegment{256, false, 2},
  };
  ShardSpec shard;
  shard.kind = ShardSpec::Kind::kFused;
  shard.split = SplitSpec{Axis::kTensorParallel, 0, 1};
  shard.segments = kSegments;
  const ParamDecl p = Decl(Shape4(4608, 512), shard);

  for (const ParallelContext& ctx : world.contexts) {
    const ShardShape local = LocalShape(p, ctx);
    EXPECT_EQ(local, Shape4(1280, 512));
  }
}

TEST(ShardTest, FusedReplicatedSegmentStaysWhole) {
  World world = World::Make(Mesh(1, 1, 1, 4), 1, 4);
  ASSERT_NE(world.topology, nullptr);

  static constexpr FusedSegment kSegments[] = {
      FusedSegment{128, false, 1},
      FusedSegment{96, true, 1},
  };
  ShardSpec shard;
  shard.kind = ShardSpec::Kind::kFused;
  shard.split = SplitSpec{Axis::kTensorParallel, 0, 1};
  shard.segments = kSegments;
  const ParamDecl p = Decl(Shape4(224, 16), shard);

  // 128/4 + 96 = 128 rows per rank.
  for (const ParallelContext& ctx : world.contexts) {
    EXPECT_EQ(LocalShape(p, ctx), Shape4(128, 16));
  }
}

TEST(ShardTest, SplitDimensionOneKeepsFusedOutputIntact) {
  // Splitting along dim 1 (columns) leaves dim 0 alone.
  World world = World::Make(Mesh(1, 1, 1, 2), 1, 2);
  ASSERT_NE(world.topology, nullptr);
  const ParamDecl p = Decl(Shape4(64, 512), Split(1));

  for (const ParallelContext& ctx : world.contexts) {
    EXPECT_EQ(LocalShape(p, ctx), Shape4(64, 256));
  }
}

TEST(ShardTest, SplitOverNonTensorAxisUsesThatAxisSize) {
  // dp = 2: a DP-split tensor halves on both tp ranks of each replica.
  World world = World::Make(Mesh(2, 1, 1, 2), 1, 4);
  ASSERT_NE(world.topology, nullptr);
  const ParamDecl p = Decl(Shape4(800, 8), Split(0, Axis::kDataParallel));

  for (const ParallelContext& ctx : world.contexts) {
    EXPECT_EQ(LocalShape(p, ctx), Shape4(400, 8));
  }
}

TEST(ShardTest, ExpertSplitPartitionsExpertsOverEp) {
  // dp_attention dp=2 x tp=1 with EP: plane = 2, ep = 2, so experts split
  // over the two dp ranks; moe_tp = 1 leaves the dense dims whole.
  ParallelConfig cfg = Mesh(2, 1, 1, 1);
  cfg.dp_attention = true;
  cfg.enable_expert_parallel = true;
  World world = World::Make(cfg, 1, 2);
  ASSERT_NE(world.topology, nullptr);

  ShardSpec shard;
  shard.kind = ShardSpec::Kind::kExpert;
  shard.expert_dim = 0;
  shard.split = SplitSpec{Axis::kTensorParallel, 1, 1};
  const ParamDecl p = Decl(Shape4(8, 64, 32), shard);

  // 8 experts over ep = 2: 4 each; dims 1..2 untouched (moe_tp = 1).
  for (const ParallelContext& ctx : world.contexts) {
    const MoeLayout moe = ctx.Moe();
    ASSERT_EQ(moe.ep, 2);
    ASSERT_EQ(moe.moe_tp, 1);
    EXPECT_EQ(LocalShape(p, ctx), Shape4(4, 64, 32));
  }

  // Uneven experts put the remainder on the lower ep rank: rank 0 holds 3.
  const ParamDecl odd = Decl(Shape4(5, 64, 32), shard);
  ASSERT_EQ(world.contexts.size(), 2u);
  EXPECT_EQ(LocalShape(odd, world.contexts[0]), Shape4(3, 64, 32));
  EXPECT_EQ(LocalShape(odd, world.contexts[1]), Shape4(2, 64, 32));
}

TEST(ShardTest, MoeLayoutRanksFollowThePlaneOrdering) {
  ParallelConfig cfg = Mesh(1, 1, 1, 4);
  cfg.enable_expert_parallel = true;
  World world = World::Make(cfg, 1, 4);
  ASSERT_NE(world.topology, nullptr);

  // Pure EP over tp: ep = 4, moe_tp = 1; ep_rank follows plane_index = tp.
  for (size_t g = 0; g < world.contexts.size(); ++g) {
    const MoeLayout moe = world.contexts[g].Moe();
    EXPECT_EQ(moe.ep, 4);
    EXPECT_EQ(moe.moe_tp, 1);
    EXPECT_EQ(moe.ep_rank, static_cast<int>(g));
    EXPECT_EQ(moe.moe_tp_rank, 0);
  }
}

TEST(ShardTest, OwnedHereFollowsPipelineOwnership) {
  // pp = 2 over 8 ranks (tp = 4): ranks 0..3 stage 0 (layers [0, 12)),
  // ranks 4..7 stage 1 (layers [12, 24)).
  World world = World::Make(Mesh(1, 2, 1, 4), 2, 4);
  ASSERT_NE(world.topology, nullptr);

  const auto layer = [](int index) {
    ParamDecl p = Decl(Shape4(8), ShardSpec{});
    p.owner = Owner::kLayer;
    p.layer = index;
    return p;
  };
  const auto owned = [](Owner owner) {
    ParamDecl p = Decl(Shape4(8), ShardSpec{});
    p.owner = owner;
    return p;
  };

  const ParallelContext& first = world.contexts[0];
  const ParallelContext& last = world.contexts[4];

  EXPECT_EQ(first.LayerRange(24), (std::pair<int, int>({0, 12})));
  EXPECT_EQ(last.LayerRange(24), (std::pair<int, int>({12, 24})));

  EXPECT_TRUE(OwnedHere(layer(0), first, 24));
  EXPECT_TRUE(OwnedHere(layer(11), first, 24));
  EXPECT_FALSE(OwnedHere(layer(12), first, 24));
  EXPECT_TRUE(OwnedHere(layer(23), last, 24));
  EXPECT_FALSE(OwnedHere(layer(0), last, 24));

  EXPECT_TRUE(OwnedHere(owned(Owner::kAllStages), first, 24));
  EXPECT_TRUE(OwnedHere(owned(Owner::kAllStages), last, 24));
  EXPECT_TRUE(OwnedHere(owned(Owner::kFirstStage), first, 24));
  EXPECT_FALSE(OwnedHere(owned(Owner::kFirstStage), last, 24));
  EXPECT_FALSE(OwnedHere(owned(Owner::kLastStage), first, 24));
  EXPECT_TRUE(OwnedHere(owned(Owner::kLastStage), last, 24));

  // Stage-chain navigation (ranks within the Pp group).
  EXPECT_EQ(first.PrevStageRank(), -1);
  EXPECT_EQ(first.NextStageRank(), 1);  // Positions within the Pp group.
  EXPECT_EQ(last.PrevStageRank(), 0);
  EXPECT_EQ(last.NextStageRank(), -1);
}

}  // namespace
}  // namespace inferx
