#include "inferx/models/qwen3_moe/qwen3_moe.h"

#include <optional>
#include <utility>

#include "inferx/ops/elementwise.h"
#include "inferx/ops/moe.h"

namespace inferx {
namespace models {
namespace qwen3_moe {

StatusOr<Qwen3Moe> Qwen3Moe::Create(const Qwen3MoeConfig& config,
                                    const ParallelConfig& parallel, DataType dtype,
                                    Device device) {
  const int64_t tp = parallel.tensor_parallel_size;
  const int64_t hidden = config.base.hidden_size;

  INFERX_ASSIGN_OR_RETURN(Linear router,
                          Linear::Create(hidden, config.num_experts, dtype, device));
  INFERX_ASSIGN_OR_RETURN(
      FusedExperts experts,
      FusedExperts::Create(config.num_experts, hidden,
                                   config.base.intermediate_size, tp,
                                   parallel.tensor_parallel_rank,
                                   ops::Activation::kSilu, dtype, device));

  std::optional<MLP> shared;
  std::optional<Linear> shared_gate;
  if (config.shared_expert_intermediate > 0) {
    INFERX_ASSIGN_OR_RETURN(
        shared, MLP::Create(hidden, config.shared_expert_intermediate, tp,
                                    parallel.tensor_parallel_rank,
                                    ops::Activation::kSilu, dtype, device));
    if (config.shared_expert_gate) {
      INFERX_ASSIGN_OR_RETURN(shared_gate, Linear::Create(hidden, 1, dtype, device));
    }
  }

  return Qwen3Moe(std::move(router), std::move(experts), std::move(shared),
                  std::move(shared_gate), config.Routing(), hidden,
                  parallel.tensor_parallel_rank, config.num_experts,
                  config.base.intermediate_size, config.shared_expert_intermediate);
}

std::vector<ParamDecl> Qwen3Moe::DescribeParams(const std::string& prefix) const {
  std::vector<ParamDecl> params;

  // Router: [experts, hidden], replicated across the plane.
  ParamDecl router;
  router.name = prefix + ".gate.weight";
  router.global_shape = ShardShape{num_experts_, hidden_};
  router.rank = 2;
  router.dst = router_.Weight();
  params.push_back(std::move(router));

  // Experts: the checkpoint stores each expert under its own name, so the
  // Ep partition rides the name (a rank simply does not describe or load
  // another ep_rank's experts); each expert's gate/up rows and down
  // columns split over the TP group exactly like a dense MLP. Placement
  // into the flattened expert stacks stays model-side (BindWeights).
  const auto split_rows = [&](std::string_view name) {
    ParamDecl p;
    p.name = name;
    p.global_shape = ShardShape{global_intermediate_, hidden_};
    p.rank = 2;
    ShardSpec shard;
    shard.kind = ShardSpec::Kind::kSplit;
    shard.split = SplitSpec{Axis::kTensorParallel, 0, 1};
    p.shard = shard;
    return p;
  };
  for (int64_t expert = 0; expert < num_experts_; ++expert) {
    const std::string name = prefix + ".experts." + std::to_string(expert);
    params.push_back(split_rows(name + ".gate_proj.weight"));
    params.push_back(split_rows(name + ".up_proj.weight"));

    ParamDecl down;
    down.name = name + ".down_proj.weight";
    down.global_shape = ShardShape{hidden_, global_intermediate_};
    down.rank = 2;
    ShardSpec cols;
    cols.kind = ShardSpec::Kind::kSplit;
    cols.split = SplitSpec{Axis::kTensorParallel, 1, 1};
    down.shard = cols;
    params.push_back(std::move(down));
  }

  // Shared expert (when present): a dense MLP, TP-split like Qwen3MLP.
  if (shared_expert_.has_value()) {
    const std::string shared = prefix + ".shared_expert";
    params.push_back(split_rows(shared + ".gate_proj.weight"));
    // The row extent differs from the routed experts.
    params.back().global_shape = ShardShape{shared_global_intermediate_, hidden_};
    params.push_back(split_rows(shared + ".up_proj.weight"));
    params.back().global_shape = ShardShape{shared_global_intermediate_, hidden_};

    ParamDecl down;
    down.name = shared + ".down_proj.weight";
    down.global_shape = ShardShape{hidden_, shared_global_intermediate_};
    down.rank = 2;
    ShardSpec cols;
    cols.kind = ShardSpec::Kind::kSplit;
    cols.split = SplitSpec{Axis::kTensorParallel, 1, 1};
    down.shard = cols;
    down.dst = shared_expert_->Down().Weight();
    params.push_back(std::move(down));

    if (shared_expert_gate_.has_value()) {
      ParamDecl gate;
      gate.name = shared + "_gate.weight";
      gate.global_shape = ShardShape{1, hidden_};
      gate.rank = 2;
      gate.dst = shared_expert_gate_->Weight();
      params.push_back(std::move(gate));
    }
  }
  return params;
}

Status Qwen3Moe::Forward(ForwardContext& ctx, const Tensor& hidden, Tensor& out) const {
  const int64_t tokens = hidden.Dim(0);
  const int64_t topk = routing_.topk;

  INFERX_ASSIGN_OR_RETURN(
      Tensor topk_indices,
      Tensor::Empty(DataType::kInt32, Shape({tokens * topk}), &ctx.scratch));
  INFERX_ASSIGN_OR_RETURN(
      Tensor topk_weights,
      Tensor::Empty(DataType::kFloat32, Shape({tokens * topk}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(ops::RouteTokens(ctx.op, hidden, router_.Weight(),
                                          /*router_bias=*/nullptr,
                                          /*correction_bias=*/nullptr, routing_,
                                          topk_indices, topk_weights));
  INFERX_RETURN_IF_ERROR(
      experts_.Forward(ctx.op, ctx.scratch, ctx.comm, hidden, topk_indices,
                       topk_weights, out));

  if (!shared_expert_.has_value()) return OkStatus();
  INFERX_ASSIGN_OR_RETURN(
      Tensor shared_out,
      Tensor::Empty(hidden.GetDataType(), Shape({tokens, hidden_}), &ctx.scratch));
  INFERX_RETURN_IF_ERROR(
      shared_expert_->Forward(ctx.op, ctx.scratch, ctx.comm, hidden, shared_out));
  if (shared_expert_gate_.has_value()) {
    INFERX_ASSIGN_OR_RETURN(Tensor gate,
                            Tensor::Empty(hidden.GetDataType(), Shape({tokens, 1}),
                                          &ctx.scratch));
    INFERX_RETURN_IF_ERROR(shared_expert_gate_->Forward(ctx.op, hidden, gate));
    INFERX_RETURN_IF_ERROR(ops::MulSigmoidRowGate(ctx.op, shared_out, gate));
  }
  return ops::Add(ctx.op, out, shared_out, out);
}

Status Qwen3Moe::BindWeights(const std::string& prefix, WeightMap& weights) {
  INFERX_RETURN_IF_ERROR(weights.Add(prefix + ".gate.weight",
                                     WeightBinding::Replicate(router_.MutableWeight())));

  // Expert stacks are flattened [E * rows_per_expert, width]; each expert's
  // rank shard lands inside its own slot of the stack. The down shard is
  // column-sharded per expert, so its spans are spelled out: all source
  // rows of expert m into dest rows [m * hidden, (m + 1) * hidden).
  Tensor* gate_up = experts_.MutableGateUpWeight();
  Tensor* down = experts_.MutableDownWeight();
  const int64_t shard = experts_.IntermediateShard();
  const int64_t hidden = hidden_;
  for (int64_t expert = 0; expert < experts_.NumExperts(); ++expert) {
    const std::string name = prefix + ".experts." + std::to_string(expert);
    INFERX_RETURN_IF_ERROR(weights.Add(
        name + ".gate_proj.weight",
        WeightBinding::ShardRows(gate_up, shard, tp_rank_,
                                 expert * 2 * shard)));
    INFERX_RETURN_IF_ERROR(weights.Add(
        name + ".up_proj.weight",
        WeightBinding::ShardRows(gate_up, shard, tp_rank_,
                                 expert * 2 * shard + shard)));
    WeightBinding expert_down;
    expert_down.dest = down;
    expert_down.src_col = tp_rank_ * shard;
    expert_down.spans.push_back({0, expert * hidden, hidden});
    INFERX_RETURN_IF_ERROR(weights.Add(name + ".down_proj.weight",
                                       std::move(expert_down)));
  }

  if (!shared_expert_.has_value()) return OkStatus();
  Linear* shared_gate_up = shared_expert_->MutableGateUp();
  Linear* shared_down = shared_expert_->MutableDown();
  const int64_t shared_shard = shared_expert_->IntermediateShard();
  const std::string shared_name = prefix + ".shared_expert";
  INFERX_RETURN_IF_ERROR(weights.Add(
      shared_name + ".gate_proj.weight",
      WeightBinding::ShardRows(shared_gate_up->MutableWeight(), shared_shard, tp_rank_,
                               0)));
  INFERX_RETURN_IF_ERROR(weights.Add(
      shared_name + ".up_proj.weight",
      WeightBinding::ShardRows(shared_gate_up->MutableWeight(), shared_shard, tp_rank_,
                               shared_shard)));
  INFERX_RETURN_IF_ERROR(
      weights.Add(shared_name + ".down_proj.weight",
                  WeightBinding::ShardCols(shared_down->MutableWeight(), shared_shard,
                                           tp_rank_)));
  if (shared_expert_gate_.has_value()) {
    INFERX_RETURN_IF_ERROR(
        weights.Add(prefix + ".shared_expert_gate.weight",
                    WeightBinding::Replicate(shared_expert_gate_->MutableWeight())));
  }
  return OkStatus();
}


}  // namespace qwen3_moe
}  // namespace models
}  // namespace inferx
