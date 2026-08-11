#include "inferx/core/device_runtime.h"

#include "device_runtime_internal.h"

namespace inferx {

StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device) {
  if (device.IsCpu()) return internal::CpuDeviceRuntime();
  return internal::AcceleratorDeviceRuntime(device);
}

}  // namespace inferx
