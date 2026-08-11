#include "inferx/model/parallel/tp_layout.h"

#include <utility>

namespace inferx::model::parallel {
namespace {

int64_t DimensionValue(const ModelConfig& config, TpDimension dimension) {
  switch (dimension) {
    case TpDimension::kAttentionHeads:
      return config.num_attention_heads;
    case TpDimension::kKvHeads:
      return config.num_key_value_heads;
    case TpDimension::kIntermediate:
      return config.intermediate_size;
    case TpDimension::kMoeIntermediate:
      return config.moe_intermediate_size;
  }
  return 0;
}

const char* DimensionName(TpDimension dimension) {
  switch (dimension) {
    case TpDimension::kAttentionHeads:
      return "attention heads";
    case TpDimension::kKvHeads:
      return "KV heads";
    case TpDimension::kIntermediate:
      return "intermediate size";
    case TpDimension::kMoeIntermediate:
      return "MoE intermediate size";
  }
  return "unknown dimension";
}

}  // namespace

TpLayout::TpLayout(std::vector<TpRule> rules,
                   std::vector<TpConstraint> constraints,
                   KvSharding kv_sharding,
                   std::vector<CollectivePoint> collective_schedule)
    : rules_(std::move(rules)),
      constraints_(std::move(constraints)),
      kv_sharding_(kv_sharding),
      collective_schedule_(std::move(collective_schedule)) {}

ShardSpec TpLayout::SpecFor(std::string_view checkpoint_name) const {
  for (const TpRule& rule : rules_) {
    if (checkpoint_name == rule.tensor_suffix ||
        checkpoint_name.ends_with(rule.tensor_suffix)) {
      return rule.shard;
    }
  }
  return {};
}

Status TpLayout::Validate(const ModelConfig& config, int tp_size) const {
  if (tp_size <= 0) {
    return InvalidArgumentError("tensor parallel size must be positive, got ",
                                tp_size);
  }
  for (const TpConstraint& constraint : constraints_) {
    const int64_t global = DimensionValue(config, constraint.dimension);
    if (global <= 0) {
      return InvalidArgumentError(DimensionName(constraint.dimension),
                                  " must be positive for tensor parallelism");
    }
    if (global % tp_size != 0) {
      return InvalidArgumentError(
          "tensor parallel size ", tp_size, " must divide ",
          DimensionName(constraint.dimension), " ", global);
    }
    const int64_t local = global / tp_size;
    if (constraint.local_unit <= 0 || local % constraint.local_unit != 0) {
      return InvalidArgumentError("local ", DimensionName(constraint.dimension),
                                  " ", local, " must align to unit ",
                                  constraint.local_unit);
    }
  }
  return OkStatus();
}

bool TpLayout::Requires(TpDimension dimension) const {
  for (const TpConstraint& constraint : constraints_) {
    if (constraint.dimension == dimension) return true;
  }
  return false;
}

TpLayout Qwen2TpLayout(const ModelConfig& config) {
  const ShardSpec heads{Partition::kRows, config.head_dim};
  const ShardSpec rows{Partition::kRows, 1};
  const ShardSpec cols{Partition::kCols, 1};
  return TpLayout(
      {{"self_attn.q_proj.weight", heads},
       {"self_attn.k_proj.weight", heads},
       {"self_attn.v_proj.weight", heads},
       {"self_attn.q_proj.bias", heads},
       {"self_attn.k_proj.bias", heads},
       {"self_attn.v_proj.bias", heads},
       {"self_attn.o_proj.weight", {Partition::kCols, config.head_dim}},
       {"mlp.gate_proj.weight", rows},
       {"mlp.up_proj.weight", rows},
       {"mlp.down_proj.weight", cols}},
      {{TpDimension::kAttentionHeads, 1},
       {TpDimension::kKvHeads, 1},
       {TpDimension::kIntermediate, 1}},
      KvSharding::kHeads,
      {CollectivePoint::kAfterAttentionOutput,
       CollectivePoint::kAfterFfnOutput});
}

TpLayout GptOssTpLayout(const ModelConfig& config) {
  const ShardSpec attention_rows{Partition::kRows, config.head_dim};
  const ShardSpec attention_cols{Partition::kCols, config.head_dim};
  const ShardSpec expert_gate_up{Partition::kRows, 1, 1, 2};
  // MXFP4 stores two weights per byte and one scale per 32 weights. Requiring
  // 16 packed bytes per local down shard keeps both representations aligned.
  const ShardSpec expert_down_blocks{Partition::kCols, 16, 2};
  const ShardSpec expert_down_scales{Partition::kCols, 1, 2};
  return TpLayout({{"self_attn.q_proj.weight", attention_rows},
                   {"self_attn.k_proj.weight", attention_rows},
                   {"self_attn.v_proj.weight", attention_rows},
                   {"self_attn.q_proj.bias", attention_rows},
                   {"self_attn.k_proj.bias", attention_rows},
                   {"self_attn.v_proj.bias", attention_rows},
                   {"self_attn.o_proj.weight", attention_cols},
                   {"self_attn.sinks", {Partition::kRows, 1}},
                   {"mlp.experts.gate_up_proj_blocks", expert_gate_up},
                   {"mlp.experts.gate_up_proj_scales", expert_gate_up},
                   {"mlp.experts.gate_up_proj_bias", expert_gate_up},
                   {"mlp.experts.down_proj_blocks", expert_down_blocks},
                   {"mlp.experts.down_proj_scales", expert_down_scales}},
                  {{TpDimension::kAttentionHeads, 1},
                   {TpDimension::kKvHeads, 1},
                   {TpDimension::kMoeIntermediate, 32}},
                  KvSharding::kHeads,
                  {CollectivePoint::kAfterAttentionOutput,
                   CollectivePoint::kAfterFfnOutput});
}

}  // namespace inferx::model::parallel
