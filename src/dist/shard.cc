#include "inferx/dist/shard.h"

#include <algorithm>

namespace inferx {
namespace {

/// Local extent of `global` for piece `piece` of `pieces` (remainder up
/// front, matching StageLayers' convention).
int64_t PieceExtent(int64_t global, int pieces, int piece) {
  if (pieces <= 1) return global;
  const int64_t base = global / pieces;
  const int64_t remainder = global % pieces;
  return base + (piece < remainder ? 1 : 0);
}

/// Distinct pieces along an axis of `n` ranks at `replicate` replication.
int DistinctPieces(int n, int replicate) {
  return std::max(1, n / std::max(1, replicate));
}

/// Local extent of one split segment (docs/executor.md §7.1): with
/// `replicate > 1`, groups of `replicate` ranks share a piece.
int64_t SplitExtent(int64_t global, const ParallelContext& ctx, Axis axis, int replicate) {
  const int n = ctx.AxisSize(axis);
  const int rank = ctx.AxisRank(axis);
  const int piece = rank / std::max(1, replicate);
  return PieceExtent(global, DistinctPieces(n, replicate), piece);
}

}  // namespace

ShardShape LocalShape(const ParamDecl& p, const ParallelContext& ctx) {
  ShardShape local = p.global_shape;
  const ShardSpec& shard = p.shard;

  switch (shard.kind) {
    case ShardSpec::Kind::kReplicated:
      return local;

    case ShardSpec::Kind::kSplit: {
      const int dim = shard.split.dim;
      local[static_cast<size_t>(dim)] =
          SplitExtent(p.global_shape[static_cast<size_t>(dim)], ctx, shard.split.axis,
                      shard.split.replicate);
      return local;
    }

    case ShardSpec::Kind::kFused: {
      // Each segment shrinks independently along split.dim; the local
      // extent is their sum (§7.1's fused-QKV example).
      int64_t extent = 0;
      for (const FusedSegment& segment : shard.segments) {
        extent += segment.replicated
                      ? segment.size
                      : SplitExtent(segment.size, ctx, shard.split.axis, segment.replicate);
      }
      local[static_cast<size_t>(shard.split.dim)] = extent;
      return local;
    }

    case ShardSpec::Kind::kExpert: {
      const MoeLayout moe = ctx.Moe();
      local[static_cast<size_t>(shard.expert_dim)] = PieceExtent(
          p.global_shape[static_cast<size_t>(shard.expert_dim)], moe.ep, moe.ep_rank);
      local[static_cast<size_t>(shard.split.dim)] =
          SplitExtent(p.global_shape[static_cast<size_t>(shard.split.dim)], ctx,
                      shard.split.axis, shard.split.replicate);
      // The expert dim is partitioned over Ep, never also over the split
      // axis: MoeTp (the split's group for experts) never shares a dim with
      // the expert partition (§5.3).
      if (shard.split.dim == shard.expert_dim) {
        local[static_cast<size_t>(shard.expert_dim)] = PieceExtent(
            p.global_shape[static_cast<size_t>(shard.expert_dim)], moe.ep, moe.ep_rank);
      }
      return local;
    }
  }
  return local;
}

bool OwnedHere(const ParamDecl& p, const ParallelContext& ctx, int num_layers) {
  switch (p.owner) {
    case Owner::kAllStages:
      return true;
    case Owner::kFirstStage:
      return ctx.IsFirstStage();
    case Owner::kLastStage:
      return ctx.IsLastStage();
    case Owner::kLayer: {
      const auto [first, last] = ctx.LayerRange(num_layers);
      return p.layer >= first && p.layer < last;
    }
  }
  return false;
}

}  // namespace inferx
