#ifndef INFERX_DIST_PARALLEL_CONTEXT_H_
#define INFERX_DIST_PARALLEL_CONTEXT_H_

#include "inferx/config/parallel_config.h"
#include "inferx/dist/topology.h"

namespace inferx {

/// \brief The MoE FFN layout one rank participates in
///        (docs/executor.md §8.1).
struct MoeLayout {
  int ep = 1;
  int moe_tp = 1;
  int ep_rank = 0;
  int moe_tp_rank = 0;

  /// \brief Experts this rank holds of `num_experts` total.
  [[nodiscard]] constexpr int ExpertsPerRank(int num_experts) const noexcept {
    return num_experts / ep;
  }
  /// \brief First expert index this rank holds.
  [[nodiscard]] constexpr int FirstExpert(int num_experts) const noexcept {
    return ep_rank * (num_experts / ep);
  }
};

/// \brief What model and operator code see of the parallel world: an
///        immutable view over the Topology and this rank's place in it
///        (docs/executor.md §8.1).
///
/// Communicators are deliberately absent: they land with the Communicator
/// work (§6), at which point this class gains tp()/ep()/... accessors over
/// a borrowed CommSet. Sizes, ranks, and the MoE layout are complete here.
class ParallelContext {
 public:
  /// \brief Views `topology` from `self`'s position; both must outlive the
  ///        context.
  ParallelContext(const Topology& topology, const RankInfo& self)
      : topology_(topology), self_(self) {}

  /// \name Sizes and ranks
  ///\{
  [[nodiscard]] int DpSize() const noexcept {
    return topology_.GetParallelConfig().data_parallel_size;
  }
  [[nodiscard]] int DpRank() const noexcept { return self_.c.dp; }
  [[nodiscard]] int PpSize() const noexcept {
    return topology_.GetParallelConfig().pipeline_parallel_size;
  }
  [[nodiscard]] int PpRank() const noexcept { return self_.c.pp; }
  [[nodiscard]] int CpSize() const noexcept {
    return topology_.GetParallelConfig().context_parallel_size;
  }
  [[nodiscard]] int CpRank() const noexcept { return self_.c.cp; }
  [[nodiscard]] int TpSize() const noexcept {
    return topology_.GetParallelConfig().tensor_parallel_size;
  }
  [[nodiscard]] int TpRank() const noexcept { return self_.c.tp; }
  [[nodiscard]] bool IsFirstStage() const noexcept { return PpRank() == 0; }
  [[nodiscard]] bool IsLastStage() const noexcept { return PpRank() == PpSize() - 1; }
  [[nodiscard]] bool SequenceParallel() const noexcept {
    return topology_.GetParallelConfig().sequence_parallel;
  }
  [[nodiscard]] bool DpAttention() const noexcept {
    return topology_.GetParallelConfig().dp_attention;
  }
  ///\}

  /// \brief Size of `axis`'s group (e.g. SplitSpec::axis).
  [[nodiscard]] int AxisSize(Axis axis) const noexcept {
    const ParallelConfig& cfg = topology_.GetParallelConfig();
    switch (axis) {
      case Axis::kDataParallel:
        return cfg.data_parallel_size;
      case Axis::kPipelineParallel:
        return cfg.pipeline_parallel_size;
      case Axis::kContextParallel:
        return cfg.context_parallel_size;
      case Axis::kTensorParallel:
        return cfg.tensor_parallel_size;
      default:
        return 1;
    }
  }

  /// \brief This rank's coordinate along `axis`.
  [[nodiscard]] int AxisRank(Axis axis) const noexcept {
    switch (axis) {
      case Axis::kDataParallel: return self_.c.dp;
      case Axis::kPipelineParallel: return self_.c.pp;
      case Axis::kContextParallel: return self_.c.cp;
      case Axis::kTensorParallel: return self_.c.tp;
      default: return 0;
    }
  }

  /// \brief The MoE layout derived from the topology's Ep/MoeTp groups.
  [[nodiscard]] MoeLayout Moe() const noexcept {
    MoeLayout moe;
    moe.ep = topology_.GroupOf(self_.global, GroupKind::kEp).size();
    moe.moe_tp = topology_.GroupOf(self_.global, GroupKind::kMoeTp).size();
    moe.ep_rank = topology_.RankInGroup(self_.global, GroupKind::kEp);
    moe.moe_tp_rank = topology_.RankInGroup(self_.global, GroupKind::kMoeTp);
    return moe;
  }

  [[nodiscard]] const Topology& GetTopology() const noexcept { return topology_; }
  [[nodiscard]] const RankInfo& GetSelf() const noexcept { return self_; }

  /// \brief This stage's layer range [first, last) of a `num_layers` model.
  [[nodiscard]] std::pair<int, int> LayerRange(int num_layers) const {
    return topology_.StageLayers(PpRank(), num_layers);
  }

  /// \brief Rank (within the Pp group) of the previous pipeline stage;
  ///        -1 on the first stage.
  [[nodiscard]] int PrevStageRank() const noexcept {
    const int here = topology_.RankInGroup(self_.global, GroupKind::kPp);
    return here > 0 ? here - 1 : -1;
  }
  /// \brief Rank (within the Pp group) of the next pipeline stage;
  ///        -1 on the last stage.
  [[nodiscard]] int NextStageRank() const noexcept {
    const int here = topology_.RankInGroup(self_.global, GroupKind::kPp);
    return here < PpSize() - 1 ? here + 1 : -1;
  }

 private:
  const Topology& topology_;
  const RankInfo& self_;
};

}  // namespace inferx

#endif  // INFERX_DIST_PARALLEL_CONTEXT_H_
