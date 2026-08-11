#include "device_runtime_internal.h"

namespace inferx::internal {

StatusOr<DeviceRuntime*> AcceleratorDeviceRuntime(DeviceId device) {
  return FailedPreconditionError("no runtime is built for ", device.ToString());
}

}  // namespace inferx::internal
