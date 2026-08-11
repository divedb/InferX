#include "kv_autosize.h"

#include <cuda_runtime_api.h>

#include <algorithm>

#include "absl/log/log.h"
#include "inferx/core/cuda_utils.h"

namespace inferx::engine {

StatusOr<int64_t> ResolveKvBlocks(const KvSizingSpec& spec, int64_t free_bytes,
                                  int64_t total_bytes) {
  if (spec.explicit_blocks > 0) return spec.explicit_blocks;
  if (spec.block_bytes <= 0) {
    return InvalidArgumentError("KV sizing needs positive block_bytes, got ",
                                spec.block_bytes);
  }

  if (spec.explicit_bytes > 0) {
    const int64_t blocks = spec.explicit_bytes / spec.block_bytes;
    if (blocks < spec.min_blocks) {
      return InvalidArgumentError(
          "--kv-cache-memory-bytes ", spec.explicit_bytes, " holds only ",
          blocks, " blocks of ", spec.block_bytes, " bytes but one ",
          "max-model-len sequence needs ", spec.min_blocks,
          "; raise the budget or lower --max-model-len");
    }
    return blocks;
  }

  const int64_t used_bytes = total_bytes - free_bytes;
  const int64_t budget = static_cast<int64_t>(
                             spec.gpu_memory_utilization *
                             static_cast<double>(total_bytes)) -
                         used_bytes - spec.headroom_bytes;
  const int64_t blocks = budget > 0 ? budget / spec.block_bytes : 0;
  if (blocks < spec.min_blocks) {
    return InvalidArgumentError(
        "the KV cache does not fit: --gpu-memory-utilization ",
        spec.gpu_memory_utilization, " of ", total_bytes, " bytes leaves ",
        std::max<int64_t>(budget, 0), " bytes after the ", used_bytes,
        " already allocated (weights, activations) and ", spec.headroom_bytes,
        " headroom, which holds ", blocks, " blocks of ", spec.block_bytes,
        " bytes but one max-model-len sequence needs ", spec.min_blocks,
        "; raise --gpu-memory-utilization or lower --max-model-len");
  }
  return blocks;
}

StatusOr<int64_t> AutosizeKvBlocksOnCurrentDevice(const KvSizingSpec& spec) {
  if (spec.explicit_blocks > 0) return spec.explicit_blocks;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  INFERX_RETURN_IF_ERROR(CudaErrorToStatus(
      cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", __FILE__,
      __LINE__));

  INFERX_ASSIGN_OR_RETURN(
      const int64_t blocks,
      ResolveKvBlocks(spec, static_cast<int64_t>(free_bytes),
                      static_cast<int64_t>(total_bytes)));

  LOG(INFO) << "KV autosize: total " << total_bytes << " bytes, already used "
            << (total_bytes - free_bytes) << ", utilization "
            << spec.gpu_memory_utilization << ", headroom "
            << spec.headroom_bytes << ", block " << spec.block_bytes
            << " bytes -> " << blocks << " blocks ("
            << blocks * spec.block_bytes << " bytes)";
  return blocks;
}

}  // namespace inferx::engine
