#include "inferx/config/device_config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "inferx/core/device_runtime.h"

namespace inferx {
namespace {

constexpr std::array kAllTypes = {DeviceType::kCpu, DeviceType::kCuda, DeviceType::kHip};

// Order in which `auto` tries accelerators.
constexpr std::array kAcceleratorPreference = {DeviceType::kCuda, DeviceType::kHip};

// Device stores its ordinal in an int8_t, so larger values cannot be named.
constexpr int kMaxOrdinal = 127;

std::optional<DeviceType> ParseType(std::string_view name) {
  for (DeviceType type : kAllTypes) {
    if (name == DeviceTypeToString(type)) {
      return type;
    }
  }

  return std::nullopt;
}

std::string KnownTypes() {
  std::string out = "auto";

  for (DeviceType type : kAllTypes) {
    out += '/';
    out += DeviceTypeToString(type);
  }

  return out;
}

Status ValidateIds(const std::vector<int>& ids) {
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] < 0) {
      return InvalidArgumentError("device ids must be non-negative, got ", ids[i]);
    }

    if (ids[i] > kMaxOrdinal) {
      return InvalidArgumentError("device id ", ids[i], " is too large (maximum ", kMaxOrdinal,
                                  ")");
    }

    if (std::find(ids.begin(), ids.begin() + i, ids[i]) != ids.begin() + i) {
      return InvalidArgumentError("duplicate device id ", ids[i]);
    }
  }

  return OkStatus();
}

// CPU is always available. Drop the special case once a CPU provider exists.
Status CheckAvailable(Device device) {
  if (device.Type() == DeviceType::kCpu) return OkStatus();

  return RuntimeFor(device).status();
}

std::optional<Device> PickAccelerator() {
  for (DeviceType type : kAcceleratorPreference) {
    const StatusOr<RuntimeProvider*> provider = ProviderFor(type);
    if (!provider.ok()) continue;  // Backend not built; try the next one.

    const StatusOr<int> count = (*provider)->DeviceCount();
    if (!count.ok()) {
      LOG(WARNING) << "auto: could not probe " << DeviceTypeToString(type)
                   << " devices: " << count.status();
      continue;
    }
    if (*count == 0) continue;

    const Device candidate = Device::For(type, 0);
    StatusOr<DeviceRuntime*> runtime = RuntimeFor(candidate);
    if (!runtime.ok()) {
      LOG(WARNING) << "auto: skipping " << candidate.ToString() << ": " << runtime.status();
      continue;
    }
    const StatusOr<DeviceProperties> properties = (*runtime)->GetProperties();
    if (!properties.ok()) {
      LOG(WARNING) << "auto: skipping " << candidate.ToString() << ": " << properties.status();
      continue;
    }
    LOG(INFO) << "auto: selected " << candidate.ToString() << " (" << (*properties).name << ")";
    return candidate;
  }
  return std::nullopt;
}

}  // namespace

StatusOr<DeviceSpec> ParseDeviceSpec(std::string_view device,
                                     const std::vector<int>& device_ids) {
  const std::string name = absl::AsciiStrToLower(absl::StripAsciiWhitespace(device));

  if (name.empty()) {
    return InvalidArgumentError("--device is empty; want one of ", KnownTypes());
  }
  if (name.find(':') != std::string::npos) {
    return InvalidArgumentError("--device takes only a device type, got '", device,
                                "'; give ordinals with --device-ids");
  }

  if (name == "auto") {
    if (!device_ids.empty()) {
      return InvalidArgumentError(
          "--device-ids needs an explicit --device (got 'auto'); say which device type "
          "the ids refer to");
    }
    return DeviceSpec{};
  }

  const std::optional<DeviceType> type = ParseType(name);
  if (!type) {
    return InvalidArgumentError("unknown device type '", name, "' (want one of ", KnownTypes(),
                                ")");
  }

  if (Status status = ValidateIds(device_ids); !status.ok()) return status;

  if (*type == DeviceType::kCpu && !(device_ids.empty() || device_ids == std::vector<int>{0})) {
    return InvalidArgumentError("cpu has a single device; --device-ids must be empty or 0");
  }

  DeviceSpec spec;
  spec.automatic = false;
  if (device_ids.empty()) {
    spec.devices.push_back(Device::For(*type, 0));
  } else {
    for (int id : device_ids)
      spec.devices.push_back(Device::For(*type, static_cast<int8_t>(id)));
  }
  return spec;
}

StatusOr<std::vector<Device>> ResolveDevices(const DeviceSpec& spec) {
  if (spec.automatic) {
    if (std::optional<Device> accelerator = PickAccelerator()) {
      return std::vector<Device>{*accelerator};
    }
    LOG(INFO) << "auto: no usable accelerator found; selected cpu";
    return std::vector<Device>{Device::Cpu()};
  }

  for (const Device& device : spec.devices) {
    if (Status status = CheckAvailable(device); !status.ok()) return status;
  }
  return spec.devices;
}

Status DeviceConfig::Validate() const {
  const StatusOr<DeviceSpec> spec = ParseDeviceSpec(device, device_ids);
  if (!spec.ok()) return spec.status();

  for (const Device& d : (*spec).devices) {
    if (d.Type() == DeviceType::kCpu) continue;
    if (const StatusOr<RuntimeProvider*> provider = ProviderFor(d.Type()); !provider.ok()) {
      return provider.status();
    }
  }
  return OkStatus();
}

StatusOr<std::vector<Device>> DeviceConfig::Resolve() const {
  const StatusOr<DeviceSpec> spec = ParseDeviceSpec(device, device_ids);
  if (!spec.ok()) return spec.status();
  return ResolveDevices(*spec);
}

}  // namespace inferx