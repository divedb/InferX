#include "inferx/core/device_runtime.h"

#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "inferx/core/status.h"

#ifdef INFERX_WITH_CUDA
#include "cuda_provider.h"
#endif

namespace inferx {
namespace {

struct Registry {
  std::mutex mu;
  std::map<DeviceType, std::unique_ptr<RuntimeProvider>> providers;
};

// Intentionally leaked: providers own device runtimes whose destructors call
// into the CUDA runtime, which may already be torn down during static
// destruction at process exit.
Registry& GetRegistry() {
  static Registry* registry = new Registry;
  return *registry;
}

// Returns null when `type` is not built into this binary.
std::unique_ptr<RuntimeProvider> MakeBuiltinProvider(DeviceType type) {
  switch (type) {
#ifdef INFERX_WITH_CUDA
    case DeviceType::kCuda:
      return MakeCudaProvider();
#endif
    default:
      return nullptr;
  }
}

// Returns the provider for `type`, creating a built-in one on first use, or
// null if the backend is not built. Providers are never removed, so the
// returned pointer stays valid after the lock is released.
RuntimeProvider* FindProvider(DeviceType type) {
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mu);

  auto it = registry.providers.find(type);

  if (it == registry.providers.end()) {
    std::unique_ptr<RuntimeProvider> built = MakeBuiltinProvider(type);

    if (built == nullptr) return nullptr;

    it = registry.providers.emplace(type, std::move(built)).first;
  }

  return it->second.get();
}

Status NotBuilt(DeviceType type) {
  return absl::NotFoundError(absl::StrCat("no runtime for device type '",
                                          DeviceTypeToString(type),
                                          "': the backend is not built into this binary"));
}

}  // namespace

Status RegisterProvider(std::unique_ptr<RuntimeProvider> provider) {
  if (provider == nullptr) {
    return absl::InvalidArgumentError("RegisterProvider: provider is null");
  }

  const DeviceType type = provider->GetDeviceType();
  Registry& registry = GetRegistry();
  std::lock_guard<std::mutex> lock(registry.mu);

  if (!registry.providers.emplace(type, std::move(provider)).second) {
    return absl::AlreadyExistsError(absl::StrCat(
        "a provider for device type '", DeviceTypeToString(type), "' is already registered"));
  }

  return absl::OkStatus();
}

StatusOr<RuntimeProvider*> ProviderFor(DeviceType type) {
  RuntimeProvider* provider = FindProvider(type);

  if (provider == nullptr) return NotBuilt(type);

  return provider;
}

StatusOr<DeviceRuntime*> RuntimeFor(Device device) {
  RuntimeProvider* provider = FindProvider(device.Type());

  if (provider == nullptr) return NotBuilt(device.Type());

  return provider->GetRuntime(device.Index());
}

}  // namespace inferx