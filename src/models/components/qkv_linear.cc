#include "inferx/models/components/qkv_linear.h"

namespace inferx {
namespace components {

StatusOr<QkvParallelGeometry> ShardQkv(const AttentionConfig& total,
                                       const ParallelConfig& parallel) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  if (total.query_heads <= 0 || total.kv_heads <= 0 || total.head_dim <= 0 ||
      total.query_heads % total.kv_heads != 0) {
    return InvalidArgumentError("cannot shard invalid attention geometry: ",
                                total.query_heads, " query heads, ", total.kv_heads,
                                " kv heads, head dim ", total.head_dim);
  }

  const int64_t size = parallel.tensor_parallel_size;
  const int64_t rank = parallel.tensor_parallel_rank;
  QkvParallelGeometry g;
  g.head_dim = total.head_dim;

  if (total.query_heads % size != 0) {
    return InvalidArgumentError("query heads (", total.query_heads,
                                ") do not divide tensor parallel size (", size, ")");
  }
  g.query_heads = total.query_heads / size;

  if (total.kv_heads >= size) {
    if (total.kv_heads % size != 0) {
      return InvalidArgumentError("kv heads (", total.kv_heads,
                                  ") do not divide tensor parallel size (", size, ")");
    }
    g.kv_heads = total.kv_heads / size;
    g.kv_head_replicas = 1;
    g.kv_shard = rank;
  } else {
    if (size % total.kv_heads != 0) {
      return InvalidArgumentError("tensor parallel size (", size,
                                  ") does not divide kv heads (", total.kv_heads,
                                  ") for replication");
    }
    g.kv_heads = 1;
    g.kv_head_replicas = size / total.kv_heads;
    g.kv_shard = rank / g.kv_head_replicas;
  }

  const int64_t gate_rows = total.output_gate == OutputGate::kNone ? 1 : 2;
  g.query_rows = g.query_heads * total.head_dim * gate_rows;
  g.kv_rows = g.kv_heads * total.head_dim;
  return g;
}

}  // namespace inferx::components
}  // namespace inferx
