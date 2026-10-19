#ifndef INFERX_DIST_SHARD_H_
#define INFERX_DIST_SHARD_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <span>
#include <string_view>

#include "inferx/config/parallel_config.h"
#include "inferx/core/dtype.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/parallel_context.h"

namespace inferx {

/// \brief A parameter's global extents. Named to avoid colliding with
///        core::Shape; rank-4 covers every sharded parameter, with
///        trailing 1s padding shorter tensors.
using ShardShape = std::array<std::int64_t, 4>;

/// \brief Split dimension `dim` over `axis`. If `replicate > 1`, the axis
///        size is divided into groups of `replicate` ranks that hold the
///        SAME piece (e.g. KV heads when tp > num_kv_heads).
struct SplitSpec {
  Axis axis = Axis::kTensorParallel;
  int dim = 0;
  int replicate = 1;
};

/// \brief One segment of a fused projection (QKV, gate_up) along
///        `split.dim`.
struct FusedSegment {
  /// Global extent of this segment.
  std::int64_t size = 0;
  /// True: segment is whole on every rank.
  bool replicated = false;
  /// Like SplitSpec::replicate for this segment.
  int replicate = 1;
};

/// \brief How one parameter is partitioned (docs/executor.md §7.1).
struct ShardSpec {
  enum class Kind : std::uint8_t {
    /// Full tensor on every rank of the stage.
    kReplicated,
    /// `split` over its axis.
    kSplit,
    /// `segments` concatenated along split.dim, each segment
    /// split/replicated independently.
    kFused,
    /// dim `expert_dim` indexes experts, partitioned over Ep; other dims
    /// by `split` over MoeTp.
    kExpert
  } kind = Kind::kReplicated;
  SplitSpec split{};
  std::span<const FusedSegment> segments{};
  int expert_dim = 0;
};

/// \brief Which pipeline stages hold a parameter.
enum class Owner : std::uint8_t { kLayer, kFirstStage, kLastStage, kAllStages };

/// \brief One parameter's declaration: the checkpoint's view plus the
///        partitioning plan. The model allocates `dst` from LocalShape()
///        once, before loading.
struct ParamDecl {
  /// Checkpoint tensor name. Owned: declarations build names from
  /// composed prefixes, so a view would dangle.
  std::string name;
  ShardShape global_shape{};
  /// Tensor rank (number of significant dims); trailing extents are 1.
  std::uint8_t rank = 0;
  DataType dtype = DataType::kFloat32;
  ShardSpec shard{};
  Owner owner = Owner::kLayer;
  /// Global layer index when owner == kLayer.
  int layer = -1;
  /// Local storage; engaged by the model after allocation.
  std::optional<Tensor> dst{};
};

/// \brief Single source of truth: local shape of `p` on this rank
///        (docs/executor.md §7.1).
///
/// Splits divide their dimension into one piece per distinct holder —
/// axis size / replicate — with any remainder taken by the leading
/// pieces. Fused specs sum their segments' local extents; expert specs
/// partition `expert_dim` over Ep and apply `split` over MoeTp.
[[nodiscard]] ShardShape LocalShape(const ParamDecl& p, const ParallelContext& ctx);

/// \brief Does this rank hold `p` at all (PP ownership)?
[[nodiscard]] bool OwnedHere(const ParamDecl& p, const ParallelContext& ctx, int num_layers);

}  // namespace inferx

#endif  // INFERX_DIST_SHARD_H_
