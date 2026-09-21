#ifndef INFERX_CACHE_PROVIDER_MEMORY_H_
#define INFERX_CACHE_PROVIDER_MEMORY_H_

#include "inferx/core/storage.h"
#include "kvc/cache.h"

namespace inferx {

// Engine integration belongs here, outside the provider SPI. Retains owned
// Storage until the imported region and all accepted transfers release it.
// Borrowed Storage cannot establish allocation lifetime and is rejected.
kvc::Result<kvc::MemoryRegion> ImportProviderMemory(const kvc::KVCacheData& data,
                                                    StoragePtr storage,
                                                    uint32_t access = KVC_READ_WRITE);

}  // namespace inferx

#endif  // INFERX_CACHE_PROVIDER_MEMORY_H_
