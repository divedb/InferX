#include "inferx/core/device_buffer.h"

#include "inferx/core/device_runtime.h"

namespace inferx {
StatusOr<DeviceBuffer> DeviceBuffer::Allocate(size_t bytes, DeviceId device) {
  if (bytes == 0) {
    return DeviceBuffer(nullptr, 0, device);
  }

  INFERX_ASSIGN_OR_RETURN(DeviceRuntime * runtime, RuntimeFor(device));
  INFERX_ASSIGN_OR_RETURN(void* p, runtime->Allocate(device, bytes));
  const size_t allocated_bytes =
      device.IsCpu()
          ? (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment
          : bytes;
  return DeviceBuffer(static_cast<std::byte*>(p), allocated_bytes, device);
}

void DeviceBuffer::Reset() {
  if (data_ == nullptr) return;

  auto runtime = RuntimeFor(device_);
  if (runtime.ok()) (void)(*runtime)->Free(device_, data_);
  data_ = nullptr;
  size_ = 0;
}

}  // namespace inferx
