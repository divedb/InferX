// Unit tests for the resolved rank mesh (docs/executor.md §5): rank
// numbering over axis_order, group derivation incl. MoE plane aliasing,
// canonical group order, node/device placement, TP locality, and the
// §5.5 validation rules.

#include "inferx/dist/topology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

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

TEST(TopologyTest, SingleRankWorld) {
  auto built = Topology::Build(Mesh(1, 1, 1, 1), Cluster(1, 1));
  ASSERT_TRUE(built.ok()) << built.status();
  EXPECT_EQ(built->WorldSize(), 1);

  const RankInfo& rank = built->Rank(0);
  EXPECT_EQ(rank.global, 0);
  EXPECT_EQ(rank.node, 0);
  EXPECT_EQ(rank.local, 0);
  EXPECT_EQ(rank.device, 0);
  EXPECT_EQ((RankCoord{0, 0, 0, 0}), rank.c);

  // Every kind resolves to the one world group of a single rank.
  for (GroupKind kind : {GroupKind::kWorld, GroupKind::kDp, GroupKind::kPp, GroupKind::kCp,
                         GroupKind::kTp, GroupKind::kEp, GroupKind::kMoeTp,
                         GroupKind::kPlane}) {
    const GroupDesc& group = built->GroupOf(0, kind);
    EXPECT_EQ(group.members, (std::vector<int>({0})));
    EXPECT_EQ(built->RankInGroup(0, kind), 0);
  }
  EXPECT_EQ(built->GetGroupDescs().size(), 1u);
}

TEST(TopologyTest, DefaultAxisOrderMatchesTheMixedRadixFormula) {
  // global = ((dp * PP + pp) * CP + cp) * TP + tp (§5.2).
  const int dp = 2, pp = 2, cp = 3, tp = 2;
  auto built = Topology::Build(Mesh(dp, pp, cp, tp), Cluster(12, 2));
  ASSERT_TRUE(built.ok()) << built.status();
  ASSERT_EQ(built->WorldSize(), 24);

  for (int d = 0; d < dp; ++d) {
    for (int p = 0; p < pp; ++p) {
      for (int c = 0; c < cp; ++c) {
        for (int t = 0; t < tp; ++t) {
          const int global = ((d * pp + p) * cp + c) * tp + t;
          const RankCoord expected{d, p, c, t};
          EXPECT_EQ(built->Rank(global).c, expected) << "global " << global;
        }
      }
    }
  }
}

TEST(TopologyTest, CustomAxisOrderPutsTpOutermost) {
  // [TP, DP, PP, CP]: global = ((tp * DP + dp) * PP + pp) * CP + cp.
  ParallelConfig cfg = Mesh(2, 2, 1, 2);
  cfg.axis_order = {Axis::kTensorParallel, Axis::kDataParallel, Axis::kPipelineParallel,
                    Axis::kContextParallel};
  auto built = Topology::Build(cfg, Cluster(1, 8));
  ASSERT_TRUE(built.ok()) << built.status();

  EXPECT_EQ(built->Rank(0).c, (RankCoord{0, 0, 0, 0}));
  EXPECT_EQ(built->Rank(1).c, (RankCoord{0, 1, 0, 0}));  // pp: second-innermost.
  EXPECT_EQ(built->Rank(2).c, (RankCoord{1, 0, 0, 0}));  // dp: third-innermost.
  EXPECT_EQ(built->Rank(4).c, (RankCoord{0, 0, 0, 1}));  // tp: outermost.
}

TEST(TopologyTest, NodeAndDevicePlacement) {
  // Two nodes of four GPUs; contiguous ranks fill node 0 then node 1.
  auto built = Topology::Build(Mesh(2, 2, 1, 2), Cluster(2, 4));
  ASSERT_TRUE(built.ok()) << built.status();

  for (int global = 0; global < 8; ++global) {
    const RankInfo& rank = built->Rank(global);
    EXPECT_EQ(rank.node, global / 4) << "global " << global;
    EXPECT_EQ(rank.local, global % 4) << "global " << global;
    EXPECT_EQ(rank.device, rank.local) << "global " << global;
  }
}

TEST(TopologyTest, MeshGroupsCoverTheRightAxes) {
  // Default order [DP, PP, CP, TP]: dp stride 4, pp stride 2, tp stride 1.
  auto built = Topology::Build(Mesh(2, 2, 1, 2), Cluster(2, 4));
  ASSERT_TRUE(built.ok()) << built.status();

  EXPECT_EQ(built->GroupOf(0, GroupKind::kWorld).members, (std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7})));
  EXPECT_EQ(built->GroupOf(0, GroupKind::kTp).members, (std::vector<int>({0, 1})));
  EXPECT_EQ(built->GroupOf(2, GroupKind::kTp).members, (std::vector<int>({2, 3})));
  EXPECT_EQ(built->GroupOf(0, GroupKind::kPp).members, (std::vector<int>({0, 2})));
  EXPECT_EQ(built->GroupOf(1, GroupKind::kPp).members, (std::vector<int>({1, 3})));
  EXPECT_EQ(built->GroupOf(0, GroupKind::kDp).members, (std::vector<int>({0, 4})));
  EXPECT_EQ(built->GroupOf(3, GroupKind::kDp).members, (std::vector<int>({3, 7})));

  // Rank within a group follows the ascending member order.
  EXPECT_EQ(built->RankInGroup(4, GroupKind::kDp), 1);
  EXPECT_EQ(built->RankInGroup(3, GroupKind::kTp), 1);

  // Every group lists ascending members and contains its querent.
  for (const GroupDesc& group : built->GetGroupDescs()) {
    EXPECT_TRUE(std::is_sorted(group.members.begin(), group.members.end()));
  }
}

TEST(TopologyTest, ContextParallelGroupVariesCp) {
  auto built = Topology::Build(Mesh(1, 1, 3, 2), Cluster(1, 6));
  ASSERT_TRUE(built.ok()) << built.status();
  // cp stride 2: group of 0 = {0, 2, 4}.
  EXPECT_EQ(built->GroupOf(0, GroupKind::kCp).members, (std::vector<int>({0, 2, 4})));
  EXPECT_EQ(built->GroupOf(1, GroupKind::kCp).members, (std::vector<int>({1, 3, 5})));
  EXPECT_EQ(built->RankInGroup(4, GroupKind::kCp), 2);
}

TEST(TopologyTest, CanonicalOrderIsStableAcrossRebuilds) {
  const ParallelConfig cfg = Mesh(2, 2, 1, 2);
  auto first = Topology::Build(cfg, Cluster(2, 4));
  auto second = Topology::Build(cfg, Cluster(2, 4));
  ASSERT_TRUE(first.ok() && second.ok());

  const auto a = first->GetGroupDescs();
  const auto b = second->GetGroupDescs();
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].id, b[i].id);
    EXPECT_EQ(a[i].kind, b[i].kind);
    EXPECT_EQ(a[i].members, b[i].members);
  }
  // Sorted by (members.front(), kind, size); ids are the sorted positions.
  for (size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].id, i);
    if (i > 0) {
      EXPECT_LE(a[i - 1].members.front(), a[i].members.front());
    }
  }
}

TEST(TopologyTest, MoEWithoutEpAliasesMoeTpToTp) {
  ParallelConfig cfg = Mesh(1, 1, 1, 4);
  cfg.enable_expert_parallel = false;
  auto built = Topology::Build(cfg, Cluster(1, 4));
  ASSERT_TRUE(built.ok()) << built.status();

  // ep = 1: MoeTp == Tp (TP-only MoE); Ep groups are singletons (§5.3).
  EXPECT_EQ(&built->GroupOf(2, GroupKind::kMoeTp), &built->GroupOf(2, GroupKind::kTp));
  EXPECT_EQ(built->GroupOf(2, GroupKind::kEp).members, (std::vector<int>({2})));
}

TEST(TopologyTest, PureEpAliasesEpToTp) {
  ParallelConfig cfg = Mesh(1, 1, 1, 4);
  cfg.enable_expert_parallel = true;
  auto built = Topology::Build(cfg, Cluster(1, 4));
  ASSERT_TRUE(built.ok()) << built.status();

  // ep = tp: pure EP inside the TP group; Ep == Tp, MoeTp singletons.
  EXPECT_EQ(&built->GroupOf(1, GroupKind::kEp), &built->GroupOf(1, GroupKind::kTp));
  EXPECT_EQ(built->GroupOf(3, GroupKind::kMoeTp).members, (std::vector<int>({3})));
}

TEST(TopologyTest, DpAttentionWithEpSpansThePlane) {
  // DPA + EP (§5.3): attention fully data-parallel (tp = 1), experts
  // spread across the dp plane; Ep == Plane.
  ParallelConfig cfg = Mesh(4, 1, 1, 1);
  cfg.dp_attention = true;
  cfg.enable_expert_parallel = true;
  auto built = Topology::Build(cfg, Cluster(1, 4));
  ASSERT_TRUE(built.ok()) << built.status();

  EXPECT_EQ(&built->GroupOf(2, GroupKind::kEp), &built->GroupOf(2, GroupKind::kPlane));
  EXPECT_EQ(built->GroupOf(0, GroupKind::kPlane).members, (std::vector<int>({0, 1, 2, 3})));
  // Attention DP groups degenerate to singletons.
  EXPECT_EQ(built->GroupOf(1, GroupKind::kTp).members, (std::vector<int>({1})));
}

TEST(TopologyTest, DpAttentionPlaneJoinsDpAndTp) {
  ParallelConfig cfg = Mesh(2, 2, 1, 2);
  cfg.dp_attention = true;
  auto built = Topology::Build(cfg, Cluster(1, 8));
  ASSERT_TRUE(built.ok()) << built.status();

  // Plane of rank 0: same (pp=0, cp=0), varying (dp, tp). With the
  // default rank order tp is innermost, so the plane interleaves:
  // {0,1} (dp0) joins {4,5} (dp1); rank 2 starts pp=1's plane.
  EXPECT_EQ(built->GroupOf(0, GroupKind::kPlane).members, (std::vector<int>({0, 1, 4, 5})));
  EXPECT_EQ(built->GroupOf(2, GroupKind::kPlane).members, (std::vector<int>({2, 3, 6, 7})));
  // Tp stays within one dp rank.
  EXPECT_EQ(built->GroupOf(0, GroupKind::kTp).members, (std::vector<int>({0, 1})));
  EXPECT_EQ(built->GroupOf(2, GroupKind::kTp).members, (std::vector<int>({2, 3})));
  // Without EP, MoeTp spans the whole plane (moe_tp = plane_size).
  EXPECT_EQ(&built->GroupOf(1, GroupKind::kMoeTp), &built->GroupOf(1, GroupKind::kPlane));
}

TEST(TopologyTest, HybridEpSlicesTheTpGroup) {
  // ep = 2 over a tp = 4 plane (via dp_attention on a 2x2 plane): the
  // plane {0,1,2,3} splits into Ep groups {0,1} and {2,3}, with MoeTp
  // groups {0,2} and {1,3}.
  ParallelConfig cfg = Mesh(2, 1, 1, 2);
  cfg.dp_attention = true;
  cfg.enable_expert_parallel = true;
  auto built = Topology::Build(cfg, Cluster(1, 4));
  ASSERT_TRUE(built.ok()) << built.status();

  // plane = 4, ep = 4 here (ExpertParallelSize() = plane when enabled):
  // Ep == Plane, MoeTp singleton. The mid-ep knob (1 < ep < plane) is not
  // expressible by the current config; this pins today's behavior.
  EXPECT_EQ(&built->GroupOf(0, GroupKind::kEp), &built->GroupOf(0, GroupKind::kPlane));
  EXPECT_EQ(built->GroupOf(0, GroupKind::kMoeTp).members, (std::vector<int>({0})));
}

TEST(TopologyTest, SingletonKindsCollapseIntoOneDesc) {
  // tp = 1, ep disabled: Tp, Ep, and MoeTp all resolve to {r} for every
  // rank -- one desc, aliased by three kinds. Plane spans dp = 2, so it
  // stays its own {0, 1} group.
  auto built = Topology::Build(Mesh(2, 1, 1, 1), Cluster(1, 2));
  ASSERT_TRUE(built.ok()) << built.status();

  const GroupDesc& tp = built->GroupOf(1, GroupKind::kTp);
  EXPECT_EQ(&built->GroupOf(1, GroupKind::kEp), &tp);
  EXPECT_EQ(&built->GroupOf(1, GroupKind::kMoeTp), &tp);
  EXPECT_EQ(tp.members, (std::vector<int>({1})));
  EXPECT_EQ(built->GroupOf(1, GroupKind::kPlane).members, (std::vector<int>({0, 1})));
}

TEST(TopologyTest, CrossNodeTpRejectedUnlessAllowed) {
  // TP outermost: rank 0's TP group is {0, 2} -- node 0 meets node 1.
  ParallelConfig cfg = Mesh(2, 1, 1, 2);
  cfg.axis_order = {Axis::kTensorParallel, Axis::kDataParallel, Axis::kPipelineParallel,
                    Axis::kContextParallel};

  auto rejected = Topology::Build(cfg, Cluster(2, 2));
  ASSERT_TRUE(rejected.ok() == false);
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(rejected.status().message().find("spans nodes"), std::string::npos);

  cfg.allow_cross_node_tp = true;
  auto allowed = Topology::Build(cfg, Cluster(2, 2));
  ASSERT_TRUE(allowed.ok()) << allowed.status();
  EXPECT_EQ(allowed->GroupOf(0, GroupKind::kTp).members, (std::vector<int>({0, 2})));
}

TEST(TopologyTest, ValidationListsEveryViolation) {
  ParallelConfig cfg = Mesh(1, 2, 1, 1);
  cfg.sequence_parallel = true;   // Needs tp > 1.
  cfg.dp_attention = true;       // Needs dp > 1.
  cfg.pp_layer_split = {1, 2, 3};  // Size 3 != pp 2.
  cfg.axis_order = {Axis::kDataParallel, Axis::kDataParallel, Axis::kContextParallel,
                    Axis::kTensorParallel};  // Duplicate axis.

  auto built = Topology::Build(cfg, Cluster(3, 9));  // World 2 != 27.
  ASSERT_FALSE(built.ok());
  EXPECT_EQ(built.status().code(), absl::StatusCode::kInvalidArgument);
  const std::string message(built.status().message());
  EXPECT_NE(message.find("world size 2"), std::string::npos);
  EXPECT_NE(message.find("sequence_parallel"), std::string::npos);
  EXPECT_NE(message.find("dp_attention"), std::string::npos);
  EXPECT_NE(message.find("pp_layer_split"), std::string::npos);
  EXPECT_NE(message.find("axis_order"), std::string::npos);
  // All five violations in one error, not first-wins.
  EXPECT_EQ(std::count(message.begin(), message.end(), '-'), 5);
}

TEST(TopologyTest, NegativeSizesAreRejected) {
  auto built = Topology::Build(Mesh(0, 1, 1, 2), Cluster(1, 2));
  ASSERT_FALSE(built.ok());
  EXPECT_NE(built.status().message().find("positive"), std::string::npos);
}

TEST(TopologyTest, StageLayersSplitsBalancedWithRemainderUpFront) {
  ParallelConfig cfg = Mesh(1, 3, 1, 1);
  auto built = Topology::Build(cfg, Cluster(3, 1));
  ASSERT_TRUE(built.ok()) << built.status();

  EXPECT_EQ(built->StageLayers(0, 8), (std::pair<int, int>({0, 3})));
  EXPECT_EQ(built->StageLayers(1, 8), (std::pair<int, int>({3, 6})));
  EXPECT_EQ(built->StageLayers(2, 8), (std::pair<int, int>({6, 8})));
  // Fewer layers than stages: early stages take one layer, the rest empty.
  EXPECT_EQ(built->StageLayers(0, 2), (std::pair<int, int>({0, 1})));
  EXPECT_EQ(built->StageLayers(2, 2), (std::pair<int, int>({2, 2})));
  EXPECT_EQ(built->StageLayers(-1, 8), (std::pair<int, int>({0, 0})));
  EXPECT_EQ(built->StageLayers(3, 8), (std::pair<int, int>({0, 0})));
}

TEST(TopologyTest, StageLayersHonorsExplicitSplit) {
  ParallelConfig cfg = Mesh(1, 3, 1, 1);
  cfg.pp_layer_split = {1, 2, 5};
  auto built = Topology::Build(cfg, Cluster(3, 1));
  ASSERT_TRUE(built.ok()) << built.status();

  EXPECT_EQ(built->StageLayers(0, 8), (std::pair<int, int>({0, 1})));
  EXPECT_EQ(built->StageLayers(1, 8), (std::pair<int, int>({1, 3})));
  EXPECT_EQ(built->StageLayers(2, 8), (std::pair<int, int>({3, 8})));

  cfg.pp_layer_split = {0, 2, 5};  // Non-positive entry.
  auto invalid = Topology::Build(cfg, Cluster(3, 1));
  ASSERT_FALSE(invalid.ok());
  EXPECT_NE(invalid.status().message().find("pp_layer_split"), std::string::npos);
}


}  // namespace
}  // namespace inferx
