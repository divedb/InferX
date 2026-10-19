#include "inferx/dist/topology.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <string>

namespace inferx {
namespace {

constexpr int kNumAxes = static_cast<int>(Axis::kAxisCount);
constexpr size_t kNumKinds = 8;  // Must match group_index_'s array width.

int AxisSize(const ParallelConfig& cfg, Axis axis) {
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

/// Stride of each axis position in `cfg.axis_order`, outermost first: the
/// innermost axis (last position) varies fastest with stride 1.
std::array<int, kNumAxes> AxisStrides(const ParallelConfig& cfg) {
  std::array<int, kNumAxes> strides{};
  int running = 1;
  for (int pos = kNumAxes - 1; pos >= 0; --pos) {
    strides[static_cast<size_t>(pos)] = running;
    running *= AxisSize(cfg, cfg.axis_order[static_cast<size_t>(pos)]);
  }
  return strides;
}

int GlobalOf(const ParallelConfig& cfg, const std::array<int, kNumAxes>& strides,
             const RankCoord& c) {
  int global = 0;
  for (int pos = 0; pos < kNumAxes; ++pos) {
    const Axis axis = cfg.axis_order[static_cast<size_t>(pos)];
    const int coord = axis == Axis::kDataParallel     ? c.dp
                      : axis == Axis::kPipelineParallel ? c.pp
                      : axis == Axis::kContextParallel  ? c.cp
                                                        : c.tp;
    global += coord * strides[static_cast<size_t>(pos)];
  }
  return global;
}

}  // namespace

StatusOr<Topology> Topology::Build(const ParallelConfig& cfg, const ClusterSpec& cluster) {
  // ---- validation: collect ALL violations (§5.5) ---------------------------
  std::vector<std::string> violations;
  const int world = cfg.WorldSize();
  const int plane = cfg.PlaneSize();
  const int ep = cfg.ExpertParallelSize();

  if (cfg.data_parallel_size <= 0 || cfg.pipeline_parallel_size <= 0 ||
      cfg.context_parallel_size <= 0 || cfg.tensor_parallel_size <= 0) {
    violations.push_back("parallel sizes must be positive (dp=" +
                         std::to_string(cfg.data_parallel_size) + ", pp=" +
                         std::to_string(cfg.pipeline_parallel_size) + ", cp=" +
                         std::to_string(cfg.context_parallel_size) + ", tp=" +
                         std::to_string(cfg.tensor_parallel_size) + ")");
  }
  if (cluster.num_nodes <= 0 || cluster.gpus_per_node <= 0) {
    violations.push_back("cluster needs positive num_nodes and gpus_per_node");
  } else if (world != cluster.num_nodes * cluster.gpus_per_node) {
    violations.push_back("world size " + std::to_string(world) + " does not match " +
                         std::to_string(cluster.num_nodes) + " nodes x " +
                         std::to_string(cluster.gpus_per_node) + " GPUs");
  }
  if (ep < 1 || plane % ep != 0) {
    violations.push_back("expert parallelism is ill-defined: ep=" + std::to_string(ep) +
                         " must divide plane size " + std::to_string(plane));
  }
  if (cfg.sequence_parallel && cfg.tensor_parallel_size <= 1) {
    violations.push_back("sequence_parallel requires tensor_parallel_size > 1");
  }
  if (cfg.dp_attention && cfg.data_parallel_size <= 1) {
    violations.push_back("dp_attention requires data_parallel_size > 1");
  }
  if (!cfg.pp_layer_split.empty()) {
    if (static_cast<int>(cfg.pp_layer_split.size()) != cfg.pipeline_parallel_size) {
      violations.push_back("pp_layer_split has " + std::to_string(cfg.pp_layer_split.size()) +
                           " entries but pipeline_parallel_size is " +
                           std::to_string(cfg.pipeline_parallel_size));
    }
    for (int layers : cfg.pp_layer_split) {
      if (layers <= 0) {
        violations.push_back("pp_layer_split entries must be positive");
        break;
      }
    }
  }
  {
    bool seen[kNumAxes] = {false, false, false, false};
    bool permutation = true;
    for (Axis axis : cfg.axis_order) {
      if (axis >= Axis::kAxisCount || seen[static_cast<int>(axis)]) {
        permutation = false;
        break;
      }
      seen[static_cast<int>(axis)] = true;
    }
    if (!permutation) {
      violations.push_back("axis_order must list each mesh axis exactly once");
    }
  }
  if (!violations.empty()) {
    std::string message = "invalid parallel topology:";
    for (const std::string& violation : violations) {
      message += "\n  - " + violation;
    }
    return InvalidArgumentError(message);
  }

  // ---- rank mesh (§5.4) ----------------------------------------------------
  Topology topology;
  topology.cfg_ = cfg;
  const std::array<int, kNumAxes> strides = AxisStrides(cfg);
  const int gpus_per_node = cluster.gpus_per_node;

  topology.ranks_.resize(static_cast<size_t>(world));
  for (int global = 0; global < world; ++global) {
    RankInfo& info = topology.ranks_[static_cast<size_t>(global)];
    info.global = global;
    info.node = global / gpus_per_node;
    info.local = global % gpus_per_node;
    info.device = info.local;  // device = local by default (§5.4).
    int rest = global;
    // Outermost first: each axis consumes its radix before inner ones.
    for (int pos = 0; pos < kNumAxes; ++pos) {
      const Axis axis = cfg.axis_order[static_cast<size_t>(pos)];
      const int coord = rest / strides[static_cast<size_t>(pos)];
      rest %= strides[static_cast<size_t>(pos)];
      switch (axis) {
        case Axis::kDataParallel: info.c.dp = coord; break;
        case Axis::kPipelineParallel: info.c.pp = coord; break;
        case Axis::kContextParallel: info.c.cp = coord; break;
        case Axis::kTensorParallel: info.c.tp = coord; break;
        default: break;
      }
    }
  }

  // ---- groups (§5.2-5.3) ---------------------------------------------------
  // One pass per kind, in enum order (so the first kind to claim a member
  // set owns the desc's `kind` field; later kinds alias it, §5.3). Indices
  // are provisional until the canonical sort remaps them.
  std::vector<GroupDesc> descs;
  // member set -> provisional index.
  std::map<std::vector<int>, std::uint32_t> set_index;
  topology.group_index_.assign(static_cast<size_t>(world),
                               std::array<std::uint32_t, kNumKinds>{});

  const auto add_kind = [&](GroupKind kind,
                            const std::function<std::vector<int>(int)>& key_of) {
    std::map<std::vector<int>, std::vector<int>> grouped;
    for (int global = 0; global < world; ++global) {
      grouped[key_of(global)].push_back(global);
    }
    for (auto& [key, members] : grouped) {
      (void)key;
      const auto [it, inserted] = set_index.emplace(members, 0);
      if (inserted) {
        it->second = static_cast<std::uint32_t>(descs.size());
        descs.push_back(GroupDesc{0, kind, members});
      }
      const std::uint32_t index = it->second;
      for (const int member : members) {
        topology.group_index_[static_cast<size_t>(member)][static_cast<size_t>(kind)] = index;
      }
    }
  };

  // World: one group covering every rank.
  add_kind(GroupKind::kWorld, [](int) { return std::vector<int>{}; });

  // Single-axis groups: key on every axis except the varying one.
  const auto coord_key = [&](int global, Axis a, Axis b) {
    const RankCoord& c = topology.ranks_[static_cast<size_t>(global)].c;
    std::vector<int> key;
    const int coords[kNumAxes] = {c.dp, c.pp, c.cp, c.tp};
    for (int axis = 0; axis < kNumAxes; ++axis) {
      if (static_cast<Axis>(axis) != a && static_cast<Axis>(axis) != b) {
        key.push_back(coords[axis]);
      }
    }
    return key;
  };
  add_kind(GroupKind::kDp,
           [&](int g) { return coord_key(g, Axis::kDataParallel, Axis::kDataParallel); });
  add_kind(GroupKind::kPp,
           [&](int g) { return coord_key(g, Axis::kPipelineParallel, Axis::kPipelineParallel); });
  add_kind(GroupKind::kCp,
           [&](int g) { return coord_key(g, Axis::kContextParallel, Axis::kContextParallel); });
  add_kind(GroupKind::kTp,
           [&](int g) { return coord_key(g, Axis::kTensorParallel, Axis::kTensorParallel); });
  // Plane: same (pp, cp), varying (dp, tp) — the FFN plane (§5.3).
  add_kind(GroupKind::kPlane, [&](int g) { return coord_key(g, Axis::kDataParallel,
                                                                Axis::kTensorParallel); });

  // Ep / MoeTp: mixed-radix over the plane (§5.3). plane_index spans the
  // plane only: dp*tp when dp_attention, else tp (dp then joins the key).
  // ep_rank = plane_index / moe_tp, moe_tp_rank = plane_index % moe_tp with
  // moe_tp = plane / ep.
  {
    const int moe_tp = plane / ep;
    const int tp = cfg.tensor_parallel_size;
    const auto plane_index_of = [&](const RankCoord& c) {
      return cfg.dp_attention ? c.dp * tp + c.tp : c.tp;
    };
    add_kind(GroupKind::kEp, [&](int global) {
      const RankCoord& c = topology.ranks_[static_cast<size_t>(global)].c;
      std::vector<int> key = {c.pp, c.cp, plane_index_of(c) % moe_tp};
      if (!cfg.dp_attention) key.push_back(c.dp);
      return key;
    });
    add_kind(GroupKind::kMoeTp, [&](int global) {
      const RankCoord& c = topology.ranks_[static_cast<size_t>(global)].c;
      std::vector<int> key = {c.pp, c.cp, plane_index_of(c) / moe_tp};
      if (!cfg.dp_attention) key.push_back(c.dp);
      return key;
    });
  }

  // ---- canonical order + final ids (§5.4) ----------------------------------
  std::vector<std::uint32_t> old_to_new(descs.size());
  std::vector<size_t> order(descs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    const GroupDesc& da = descs[a];
    const GroupDesc& db = descs[b];
    if (da.members.front() != db.members.front()) return da.members.front() < db.members.front();
    if (da.kind != db.kind) return da.kind < db.kind;
    return da.members.size() < db.members.size();
  });
  topology.groups_.reserve(descs.size());
  for (size_t new_index = 0; new_index < order.size(); ++new_index) {
    const size_t old_index = order[new_index];
    old_to_new[old_index] = static_cast<std::uint32_t>(new_index);
    topology.groups_.push_back(std::move(descs[old_index]));
    topology.groups_.back().id = static_cast<std::uint32_t>(new_index);
  }
  for (auto& row : topology.group_index_) {
    for (std::uint32_t& index : row) index = old_to_new[index];
  }

  // ---- TP locality (§5.4): contiguous ranks are intra-node by default ------
  if (!cfg.allow_cross_node_tp) {
    for (const GroupDesc& group : topology.GetGroupDescs()) {
      if (group.kind != GroupKind::kTp) continue;
      const int node = topology.ranks_[static_cast<size_t>(group.members.front())].node;
      for (const int member : group.members) {
        if (topology.ranks_[static_cast<size_t>(member)].node != node) {
          return InvalidArgumentError(
              "invalid parallel topology:\n  - TP group of rank " +
              std::to_string(group.members.front()) +
              " spans nodes (ranks " + std::to_string(group.members.front()) + ".." +
              std::to_string(group.members.back()) + "); set allow_cross_node_tp to accept");
        }
      }
    }
  }

  return topology;
}

const GroupDesc& Topology::GroupOf(int global, GroupKind kind) const {
  return groups_[group_index_[static_cast<size_t>(global)][static_cast<size_t>(kind)]];
}

int Topology::RankInGroup(int global, GroupKind kind) const {
  const GroupDesc& group = GroupOf(global, kind);
  const auto it = std::find(group.members.begin(), group.members.end(), global);
  return static_cast<int>(it - group.members.begin());
}

std::pair<int, int> Topology::StageLayers(int pp, int num_layers) const {
  if (pp < 0 || pp >= cfg_.pipeline_parallel_size) return {0, 0};

  if (cfg_.pp_layer_split.empty()) {
    // Balanced: earlier stages take the remainder.
    const int stages = cfg_.pipeline_parallel_size;
    const int base = num_layers / stages;
    const int remainder = num_layers % stages;
    const int first = pp * base + std::min<int>(pp, remainder);
    const int count = base + (pp < remainder ? 1 : 0);
    return {first, first + count};
  }

  int first = 0;
  for (int stage = 0; stage < pp; ++stage) {
    first += cfg_.pp_layer_split[static_cast<size_t>(stage)];
  }
  return {first, first + cfg_.pp_layer_split[static_cast<size_t>(pp)]};
}

}  // namespace inferx
