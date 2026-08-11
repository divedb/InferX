#pragma once

#include "inferx/core/device_runtime.h"

namespace inferx::internal {

DeviceRuntime* CpuDeviceRuntime();
StatusOr<DeviceRuntime*> AcceleratorDeviceRuntime(DeviceId device);

}  // namespace inferx::internal
