#pragma once

#include "inferx/core/device_runtime.h"

namespace inferx::internal {

DeviceRuntime* CpuDeviceRuntime();

#ifdef INFERX_WITH_CUDA
DeviceRuntime* CudaDeviceRuntime();
#endif

}  // namespace inferx::internal
