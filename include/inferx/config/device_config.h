#ifndef INFERX_CONFIG_DEVICE_CONFIG_H_
#define INFERX_CONFIG_DEVICE_CONFIG_H_

#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/status.h"

namespace inferx {

/// \brief The typed form of the device flags. Produced by ParseDeviceSpec().
///
/// Invariant: `automatic` is true exactly when `devices` is empty. When
/// explicit, `devices` is non-empty, duplicate-free, of one device type, and
/// every ordinal is in [0, 127] (CPU is always ordinal 0).
struct DeviceSpec {
  bool automatic = true;        ///< Let the engine pick; see ResolveDevices().
  std::vector<Device> devices;  ///< The requested devices, in the order given.
};

/// \brief Parses the `--device` and `--device-ids` values.
///
/// Rules:
///  - `device` is "auto" or one of the device type names (case-insensitive).
///  - `device_ids` lists logical ordinals; empty means "the first device".
///  - `auto` cannot be combined with ids, because the ids would have no
///    device type to refer to.
///  - `cpu` has a single device, so its ids must be empty or {0}.
///  - Ids must be unique, non-negative, and at most 127.
///
/// \param device     The `--device` value.
/// \param device_ids The `--device-ids` values.
/// \return           The parsed spec, or an error describing the problem.
StatusOr<DeviceSpec> ParseDeviceSpec(std::string_view device,
                                     const std::vector<int>& device_ids);

/// \brief Turns a spec into concrete devices that exist and can be used.
///
/// - Explicit spec: every device must have a built backend and a valid
///   ordinal. Fails otherwise; it never falls back to another device.
/// - Auto spec: the first accelerator (CUDA, ROCm, Ascend, in that order) that
///   is built, has hardware, and creates a working runtime; otherwise CPU.
///   The choice is logged.
///
/// \param spec The parsed spec.
/// \return     One or more devices, or an error status.
StatusOr<std::vector<Device>> ResolveDevices(const DeviceSpec& spec);

/// \brief Which devices the engine executes on.
///
/// The library never reads argv or the environment: the application fills
/// these from the flags, then INFERX_DEVICE / INFERX_DEVICE_IDS, then a config
/// file, in that order of precedence, and the library only sees the result.
struct DeviceConfig {
  /// The device type to execute on: "auto", "cpu", "cuda", "rocm" or "ascend".
  /// "auto" picks the best usable accelerator and falls back to CPU.
  /// EXAMPLE: --device cuda
  std::string device = "auto";

  /// The logical device ordinals to execute on, after filtering by
  /// CUDA_VISIBLE_DEVICES. Empty means the first device. Requires an explicit
  /// `device`. The engine may reject more than one id until multi-device
  /// execution is supported.
  /// EXAMPLE: --device-ids 0,1,2
  std::vector<int> device_ids;

  /// \brief Checks that the flags parse and that the backend is built in.
  ///
  /// Does not look for hardware; Resolve() does.
  ///
  /// \return OK if the config is valid; otherwise an error status.
  Status Validate() const;

  /// \brief Parses and resolves the flags. See ResolveDevices().
  ///
  /// \return The devices to execute on, or an error status.
  StatusOr<std::vector<Device>> Resolve() const;
};

}  // namespace inferx

#endif  // INFERX_CONFIG_DEVICE_CONFIG_H_