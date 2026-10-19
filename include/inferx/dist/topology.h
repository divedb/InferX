#ifndef INFERX_DIST_TOPOLOGY_H_
#define INFERX_DIST_TOPOLOGY_H_

#include <array>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "inferx/config/parallel_config.h"

namespace inferx {

/// \brief The communicable groups a rank can belong to
///        (docs/executor.md §5.2).
enum class GroupKind : std::uint8_t {
  kWorld,
  kDp,
  kPp,
  kCp,
  kTp,
  kEp,     ///< Ranks exchanging routed tokens (all-to-all).
  kMoeTp,  ///< Ranks sharing one expert's weights (all-reduce of expert output).
  kPlane   ///< Ranks sharing the FFN plane when dp_attention (token all-gather
           ///  / lockstep barrier).
};

/// \brief One rank's position in the (dp, pp, cp, tp) mesh.
struct RankCoord {
  int dp = 0;
  int pp = 0;
  int cp = 0;
  int tp = 0;

  friend bool operator==(const RankCoord&, const RankCoord&) = default;
};

/// \brief Everything the runtime needs to know about one rank.
struct RankInfo {
  int global = 0;
  int node = 0;
  int local = 0;   ///< Index within node.
  int device = 0;  ///< CUDA device ordinal (== local by default).
  RankCoord c;
};

/// \brief One distinct communicator group: a member set plus its identity.
struct GroupDesc {
  std::uint32_t id = 0;  ///< Stable index in Topology::GetGroupDescs().
  GroupKind kind{};
  /// Global ranks, ascending; identical on every member rank.
  std::vector<int> members;

  [[nodiscard]] int size() const noexcept { return static_cast<int>(members.size()); }
};

/// \brief The resolved rank mesh: immutable, deterministic, identical on
///        every rank (docs/executor.md §5).
///
/// Rank numbering is mixed-radix over `axis_order`, innermost varying
/// fastest; with the default [DP, PP, CP, TP] order,
/// `global = ((dp * PP + pp) * CP + cp) * TP + tp`. Groups with identical
/// member sets are deduplicated into one GroupDesc shared by all their
/// kinds, so aliased groups cost one communicator, not several (§5.3).
class Topology {
 public:
  /// \brief Validates the configuration against the cluster and resolves
  ///        the mesh.
  ///
  /// Collects ALL violations (§5.5) and returns one InvalidArgumentError
  /// listing them — the repo's no-exceptions counterpart of the design's
  /// ConfigError. On success every rank of the world resolves the
  /// identical topology.
  static StatusOr<Topology> Build(const ParallelConfig& config, const ClusterSpec& cluster);

  [[nodiscard]] const ParallelConfig& GetParallelConfig() const noexcept { return cfg_; }
  [[nodiscard]] int WorldSize() const noexcept { return cfg_.WorldSize(); }
  [[nodiscard]] const RankInfo& Rank(int global) const { return ranks_[global]; }

  /// \brief All distinct member sets, in the canonical order
  ///        (members.front(), kind, size) — §5.4. Same list on every rank.
  [[nodiscard]] std::span<const GroupDesc> GetGroupDescs() const noexcept { return groups_; }

  /// \brief The group of `kind` that contains `global`. Aliased kinds
  ///        return the same GroupDesc (e.g. Ep == Plane when
  ///        ep == plane_size()). Size-1 groups exist but need no
  ///        communicator.
  [[nodiscard]] const GroupDesc& GroupOf(int global, GroupKind kind) const;

  /// \brief Index of `global` within GroupOf(global, kind).members.
  [[nodiscard]] int RankInGroup(int global, GroupKind kind) const;

  /// \brief Layer range [first, last) owned by pipeline stage `pp` for a
  ///        model of `num_layers`.
  ///
  /// Balanced split when `pp_layer_split` is empty; otherwise the declared
  /// split's prefix sums. The caller checks the split sums to num_layers
  /// at model load (§5.5).
  [[nodiscard]] std::pair<int, int> StageLayers(int pp, int num_layers) const;

 private:
  Topology() = default;

  ParallelConfig cfg_{};
  std::vector<RankInfo> ranks_;
  std::vector<GroupDesc> groups_;
  /// [global][GroupKind] -> index into groups_.
  std::vector<std::array<std::uint32_t, 8>> group_index_;
};

}  // namespace inferx

#endif  // INFERX_DIST_TOPOLOGY_H_
