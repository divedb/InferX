#ifndef INFERX_CONFIG_DEVICE_CONFIG_H_
#define INFERX_CONFIG_DEVICE_CONFIG_H_

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/status.h"

namespace inferx {

struct BackendInfo {
  std::string_view name;
  bool built;
};

/// One source of truth: every backend inferx knows about, and whether
/// this build actually links its runtime.
static constexpr std::array<BackendInfo, 4> kBackends = {{
    {"cpu", true},
    {"cuda", kCudaBuilt},
    {"rocm", kRocmBuilt},
    {"ascend", kAscendBuilt},
}};

/// \brief Which device type and ordinals the engine executes on.
struct DeviceConfig {
  /// The device type to execute on; "cpu" or "cuda" or "rocm" or "ascend". Defaults to "cuda".
  /// EXAMPLE: --device cuda
  std::string device_type = "cuda";

  /// The logical device ordinals to execute on, after filtering by CUDA_VISIBLE_DEVICES.
  /// Empty means "default" (the first ordinal).
  /// EXAMPLE: --device-ids 0,1,2
  std::vector<int> device_ids;

  /// \brief The device the engine executes on.
  ///
  /// The first requested ordinal, or ordinal 0 when none is given. Only valid
  /// after Validate() has accepted the type; unbuilt backends still resolve
  /// here so the error stays with Validate().
  ///
  /// \return The primary DeviceId for this configuration.
  DeviceId PrimaryDevice() const {
    const int8_t ordinal = static_cast<int8_t>(device_ids.empty() ? 0 : device_ids.front());
    if (device_type == "cuda") return DeviceId::Cuda(ordinal);
    if (device_type == "rocm") return DeviceId::Rocm(ordinal);
    if (device_type == "ascend") return DeviceId::Ascend(ordinal);
    return DeviceId::Cpu();
  }

  /// \brief Structural validation: known type, backend compiled in,
  ///        non-negative unique ids.
  ///
  /// \return OK if the config is valid; otherwise an error status.
  Status Validate() const {
    auto it = std::find_if(kBackends.begin(), kBackends.end(),
                           [&](const BackendInfo& b) { return b.name == device_type; });
    if (it == kBackends.end()) {
      return InvalidArgumentError("unknown device type '", device_type,
                                  "' (want one of cpu/cuda/rocm/ascend)");
    }

    if (!it->built) {
      return UnimplementedError("device type '", device_type,
                                "' requested, but this build carries no ", device_type,
                                " runtime");
    }

    if (std::any_of(device_ids.begin(), device_ids.end(), [](int id) { return id < 0; })) {
      return InvalidArgumentError("device ids must be non-negative");
    }

    for (size_t i = 0; i < device_ids.size(); ++i) {
      for (size_t j = i + 1; j < device_ids.size(); ++j) {
        if (device_ids[i] == device_ids[j]) {
          return InvalidArgumentError("duplicate device id ", device_ids[i]);
        }
      }
    }

    return OkStatus();
  }
};

}  // namespace inferx

#endif  // INFERX_CONFIG_DEVICE_CONFIG_H_
