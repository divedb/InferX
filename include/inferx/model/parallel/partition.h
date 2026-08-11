#pragma once

#include <cstdint>

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

  friend bool operator==(const ShardSpec&, const ShardSpec&) = default;
};

}  // namespace inferx::model::parallel
