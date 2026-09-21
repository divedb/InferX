#include "inferx/cache/provider_memory.h"

#include <string>

namespace inferx {
kvc::Result<kvc::MemoryRegion> ImportProviderMemory(const kvc::KVCacheData& data,
                                                    StoragePtr storage, uint32_t access) {
  if (!storage || storage->IsBorrowed()) {
    return std::unexpected(
        kvc::Status{KVC_INVALID_ARGUMENT, "Provider import requires owned Storage"});
  }
  const char* runtime;
  switch (storage->Device().kind) {
    case DeviceKind::kCpu:
      runtime = "cpu";
      break;
    case DeviceKind::kCuda:
      runtime = "cuda";
      break;
    case DeviceKind::kRocm:
      runtime = "hip";
      break;
    case DeviceKind::kAscend:
      runtime = "ascend";
      break;
    default:
      return std::unexpected(kvc::Status{KVC_UNSUPPORTED, "Unknown device runtime"});
  }
  std::string runtime_name(runtime);
  std::string identifier = std::to_string(storage->Device().index);
  KvcMemoryInfo info{};
  info.struct_size = sizeof(info);
  info.memory_type = storage->Device().IsCpu() ? KVC_HOST : KVC_DEVICE;
  info.access = access;
  info.locator_kind = KVC_LOCAL_ADDRESS;
  info.local_address = reinterpret_cast<uintptr_t>(storage->Data());
  info.byte_size = storage->Size();
  info.device_runtime = {runtime_name.data(), runtime_name.size()};
  info.device_identifier = {identifier.data(), identifier.size()};
  return data.ImportMemory(info, std::make_shared<StoragePtr>(std::move(storage)));
}
}  // namespace inferx
