#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/model/config.h"
#include "inferx/model/parallel/partition.h"

namespace inferx::model::parallel {

enum class KvSharding : uint8_t { kHeads, kReplicated };

enum class CollectivePoint : uint8_t {
  kAfterAttentionOutput,
  kAfterFfnOutput,
};

enum class TpDimension : uint8_t {
  kAttentionHeads,
  kKvHeads,
  kIntermediate,
  kMoeIntermediate,
};

struct TpRule {
  // Rules match a complete checkpoint name or its suffix, allowing one rule
  // to describe the same tensor in every decoder layer.
  std::string tensor_suffix;
  ShardSpec shard;
};

struct TpConstraint {
  TpDimension dimension;
  int64_t local_unit = 1;
};

class TpLayout {
 public:
  TpLayout() = default;
  TpLayout(std::vector<TpRule> rules, std::vector<TpConstraint> constraints,
           KvSharding kv_sharding,
           std::vector<CollectivePoint> collective_schedule);

  ShardSpec SpecFor(std::string_view checkpoint_name) const;
  Status Validate(const ModelConfig& config, int tp_size) const;
  bool Requires(TpDimension dimension) const;

  KvSharding kv_sharding() const { return kv_sharding_; }
  const std::vector<CollectivePoint>& collective_schedule() const {
    return collective_schedule_;
  }

 private:
  std::vector<TpRule> rules_;
  std::vector<TpConstraint> constraints_;
  KvSharding kv_sharding_ = KvSharding::kReplicated;
  std::vector<CollectivePoint> collective_schedule_;
};

// Declarative form of the sharding already used by the dense Qwen2/Llama
// loader and forward pass.
TpLayout Qwen2TpLayout(const ModelConfig& config);

// GPT-OSS attention and expert-weight strategy. Packed MXFP4 expert tensors
// use explicit nested axes; fused gate/up tensors shard both halves together.
TpLayout GptOssTpLayout(const ModelConfig& config);

}  // namespace inferx::model::parallel
