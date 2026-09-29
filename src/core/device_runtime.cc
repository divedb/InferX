#include "inferx/core/device_runtime.h"

namespace inferx {

StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device) {
  return UnimplementedError("no runtime for device kind ", static_cast<int>(device.kind));
}

}  // namespace inferx
