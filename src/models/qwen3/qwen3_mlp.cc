#include "inferx/models/qwen3/qwen3_mlp.h"

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
  return Qwen3MLP(std::move(mlp), parallel.tensor_parallel_rank);
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
