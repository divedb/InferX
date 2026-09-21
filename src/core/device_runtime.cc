#include "inferx/core/device_runtime.h"

#include "device_runtime_internal.h"

namespace inferx {

StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device) {
  if (device.index < 0 || (device.IsCpu() && device.index != 0)) {
    return InvalidArgumentError("invalid device ordinal");
  }
  if (device.IsCpu()) return internal::CpuDeviceRuntime();
  return internal::AcceleratorDeviceRuntime(device);
}

}  // namespace inferx
