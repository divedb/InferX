// CacheConfig validation (see include/inferx/config/cache_config.h).
#include "inferx/config/cache_config.h"

#include <string>

namespace inferx {

Status CacheConfig::Validate() const {
  if (block_size < 1) {
    return InvalidArgumentError("CacheConfig: block_size must be >= 1, got ", block_size);
  }
  if (gpu_memory_utilization <= 0.0f || gpu_memory_utilization > 1.0f) {
    return InvalidArgumentError(
        "CacheConfig: gpu_memory_utilization must be in (0, 1], got ",
        gpu_memory_utilization);
  }
  if (kv_cache_memory_bytes < 0) {
    return InvalidArgumentError("CacheConfig: kv_cache_memory_bytes must be >= 0");
  }
  if (num_gpu_blocks_override < 0) {
    return InvalidArgumentError("CacheConfig: num_gpu_blocks_override must be >= 0");
  }
  if (cache_dtype != "auto" && cache_dtype != "fp8" && cache_dtype != "fp8_e4m3" &&
      cache_dtype != "fp8_e5m2") {
    return InvalidArgumentError(
        "CacheConfig: cache_dtype must be one of auto, fp8, fp8_e4m3, fp8_e5m2; got '",
        cache_dtype, "'");
  }
  if (prefix_caching_hash_algo != "sha256" && prefix_caching_hash_algo != "sha256_cbor") {
    return InvalidArgumentError(
        "CacheConfig: prefix_caching_hash_algo must be sha256 or sha256_cbor; got '",
        prefix_caching_hash_algo, "'");
  }
  if (sliding_window < 0) {
    return InvalidArgumentError("CacheConfig: sliding_window must be >= 0");
  }
  return OkStatus();
}

}  // namespace inferx
