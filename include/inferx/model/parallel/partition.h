#pragma once

#include <cstdint>

#include "inferx/core/status.h"

namespace inferx::model::parallel {

// Storage-axis partition for checkpoint tensors whose linear weights are
// consistently [out_features, in_features].
enum class Partition : uint8_t {
  kReplicated,
  kRows,
  kCols,
  kVocab,
};

struct ShardSpec {
  Partition partition = Partition::kReplicated;
  // Each local shard must contain a multiple of this many elements along the
  // partitioned axis. One means no alignment beyond divisibility.
  int64_t unit = 1;

  // Override the storage axis for nested tensors. -1 selects the partition's
  // ordinary matrix axis: rows/vocab = 0, cols = 1. Expert-stacked weights
  // use 1 for [experts, rows, cols] row shards and 2 for column shards.
  int axis = -1;

  // Independently shard this many concatenated regions along `axis`.
  // Fused gate/up tensors use 2 so every rank receives both local halves.
  int segments = 1;

  friend bool operator==(const ShardSpec&, const ShardSpec&) = default;
};

StatusOr<int> ResolveShardAxis(const ShardSpec& shard, int tensor_rank);

}  // namespace inferx::model::parallel
