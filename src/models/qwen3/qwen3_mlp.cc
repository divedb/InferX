#include "inferx/models/qwen3/qwen3_mlp.h"


#include "inferx/dist/shard.h"
#include <utility>

namespace inferx {
namespace models {
namespace qwen3 {

StatusOr<Qwen3MLP> Qwen3MLP::Create(const Qwen3Config& config, const ParallelConfig& parallel,
                                    DataType dtype, Device device) {
  INFERX_ASSIGN_OR_RETURN(
      MLP mlp,
      MLP::Create(config.hidden_size, config.intermediate_size,
                          parallel.tensor_parallel_size, parallel.tensor_parallel_rank,
                          ops::Activation::kSilu, dtype, device));
  return Qwen3MLP(std::move(mlp), parallel.tensor_parallel_rank, config.hidden_size,
                 config.intermediate_size);
}

std::vector<ParamDecl> Qwen3MLP::DescribeParams(const std::string& prefix) const {
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
  std::vector<ParamDecl> params;
  params.push_back(split_rows(prefix + ".gate_proj.weight"));
  params.push_back(split_rows(prefix + ".up_proj.weight"));

  // down_proj holds this rank's column block of [hidden, intermediate].
  ParamDecl down;
  down.name = prefix + ".down_proj.weight";
  down.global_shape = ShardShape{hidden_, global_intermediate_};
  down.rank = 2;
  ShardSpec cols;
  cols.kind = ShardSpec::Kind::kSplit;
  cols.split = SplitSpec{Axis::kTensorParallel, 1, 1};
  down.shard = cols;
  down.dst = mlp_.Down().Weight();
  params.push_back(std::move(down));
  return params;
}

Status Qwen3MLP::Forward(ForwardContext& ctx, const Tensor& hidden, Tensor& out) const {
  return mlp_.Forward(ctx.op, ctx.scratch, ctx.comm, hidden, out);
}

Status Qwen3MLP::BindWeights(const std::string& prefix, WeightMap& weights) {
  Linear* gate_up = mlp_.MutableGateUp();
  Linear* down = mlp_.MutableDown();
  const int64_t shard = mlp_.IntermediateShard();

  // The fused buffer packs [gate | up]; each projection's rank shard lands
  // at its half, and down takes this rank's column shard.
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".gate_proj.weight",
      WeightBinding::ShardRows(gate_up->MutableWeight(), shard, tp_rank_, 0)));
  INFERX_RETURN_IF_ERROR(weights.Add(
      prefix + ".up_proj.weight",
      WeightBinding::ShardRows(gate_up->MutableWeight(), shard, tp_rank_, shard)));
  return weights.Add(prefix + ".down_proj.weight",
                     WeightBinding::ShardCols(down->MutableWeight(), shard, tp_rank_));
}

}  // namespace qwen3
}  // namespace models
}  // namespace inferx
