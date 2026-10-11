#ifndef INFERX_MODELS_WEIGHT_LOADER_H_
#define INFERX_MODELS_WEIGHT_LOADER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor.h"
#include "inferx/models/safetensors.h"

namespace inferx {

class DeviceRuntime;

namespace models {

/// \brief Where one checkpoint parameter lands inside one rank's weights.
///
/// Destinations are rank-1 or rank-2 contiguous device tensors; expert
/// stacks are stored flattened ([experts * rows_per_expert, width]) so every
/// copy stays a plain 2-D row block. `spans` say which source rows land at
/// which destination rows -- that is the whole of fused-QKV, fused-gate/up,
/// and expert packing. Column windows express row-parallel sharding.
struct WeightBinding {
  /// One source-row block copied to one destination-row block.
  struct Span {
    int64_t src_row = 0;   ///< First source row (of the checkpoint tensor).
    int64_t dst_row = 0;   ///< First destination row (of `dest`).
    int64_t rows = 0;      ///< Row count; 0 copies nothing.
  };

  Tensor* dest = nullptr;  ///< Device tensor receiving the bytes.
  std::vector<Span> spans;
  int64_t src_col = 0;     ///< First checkpoint column (row-parallel shard).
  bool required = true;    ///< Missing parameter is an error when true.

  /// \brief Full copy: every row, every column.
  static WeightBinding Replicate(Tensor* dest);

  /// \brief This rank's contiguous row block of a column-parallel weight.
  ///
  /// Checkpoint rows [shard_index * shard, (shard_index + 1) * shard) land
  /// at dest rows [dest_row, dest_row + shard) -- the offset is how
  /// fused buffers (q|k|v, gate|up) place each part.
  static WeightBinding ShardRows(Tensor* dest, int64_t shard_rows, int64_t shard_index,
                                 int64_t dest_row = 0);

  /// \brief This rank's contiguous column block of a row-parallel weight
  ///        whose dest holds exactly the checkpoint's rows.
  ///
  /// Every row is copied; source columns [shard_index * shard,
  /// (shard_index + 1) * shard) land at dest columns [0, shard). Stacked
  /// destinations (expert banks) build their spans explicitly instead.
  static WeightBinding ShardCols(Tensor* dest, int64_t shard_cols, int64_t shard_index);
};

/// \brief The set of checkpoint names a model consumes, and where they go.
///
/// Built once at model construction, purely from the family's checkpoint
/// layout and this rank's parallel geometry. Nothing about safetensors or
/// sharding leaks into the layer modules: they own destination tensors, the
/// family owns the name map, the loader owns the bytes.
class WeightMap {
 public:
  /// \brief Registers one parameter. Duplicate names are an error.
  Status Add(const std::string& checkpoint_name, WeightBinding binding);

  /// \brief Marks a checkpoint name as present-but-unused (e.g. an untied
  ///        `lm_head.weight` saved by a tied model); not an error.
  void Skip(const std::string& checkpoint_name) { skipped_.insert(checkpoint_name); }

  WeightBinding* Find(const std::string& checkpoint_name);
  const absl::flat_hash_set<std::string>& Skipped() const { return skipped_; }

  /// \brief The full binding table, for load-time verification and tests.
  const absl::flat_hash_map<std::string, WeightBinding>& Bindings() const {
    return bindings_;
  }

 private:
  absl::flat_hash_map<std::string, WeightBinding> bindings_;
  absl::flat_hash_set<std::string> skipped_;
};

/// \brief Models that load from a checkpoint via the binding map.
///
/// Segregated from `Model`: the runner never binds weights, so the
/// execution interface stays four methods, while the load helpers reach
/// the map through this one.
class WeightBindable {
 public:
  virtual ~WeightBindable() = default;

  /// \brief Declares this rank's checkpoint bindings into `weights`.
  virtual Status BuildWeightMap(WeightMap& weights) = 0;
};

/// \brief Reads a checkpoint and fills every binding of a `WeightMap`.
///
/// Sources are borrowed CPU tensors from the `CheckpointReader`; the loader
/// converts dtypes on the host, stages each span, and enqueues one host-to-
/// device copy per span on `stream`. The stream is synchronized once at the
/// end, so all destination tensors are ready when this returns.
///
/// \param reader  The opened checkpoint.
/// \param weights The model's binding map; fully populated in-place.
/// \param runtime Runtime of the destination device.
/// \param stream  Copy stream on that device.
/// \return        OK, or InvalidArgument for missing/unexpected/mismatched
///                parameters (each error names the parameter).
Status LoadWeightsInto(const CheckpointReader& reader, WeightMap& weights,
                       DeviceRuntime& runtime, DeviceStream stream);

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_WEIGHT_LOADER_H_
