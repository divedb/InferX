#pragma once

#include "inferx/core/device_runtime.h"

namespace inferx::internal {

/// \brief The process-wide CPU runtime singleton.
DeviceRuntime* CpuDeviceRuntime();

/// \brief The CUDA runtime bound to `device` (accelerator/device_runtime.cc).
StatusOr<DeviceRuntime*> AcceleratorDeviceRuntime(Device device);

}  // namespace inferx::internal
