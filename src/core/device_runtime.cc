#include "inferx/core/device_runtime.h"

#include "device_runtime_internal.h"

namespace inferx {

StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device) {
  if (device.IsCpu()) return internal::CpuDeviceRuntime();
#ifdef INFERX_WITH_CUDA
  if (device.IsCuda()) return internal::CudaDeviceRuntime();
#endif
  return FailedPreconditionError("no runtime is built for ", device.ToString());
}

}  // namespace inferx
