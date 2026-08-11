#include <cstdlib>
#include <cstring>

#include "../../core/device_runtime_internal.h"

namespace inferx {
namespace {

class CpuRuntime final : public DeviceRuntime {
 public:
  DeviceKind kind() const override { return DeviceKind::kCpu; }
  int DeviceCount() const override { return 1; }
  Status SetDevice(DeviceId device) override {
    return device.IsCpu() ? OkStatus()
                          : InvalidArgumentError("CPU runtime cannot select ",
                                                 device.ToString());
  }
  StatusOr<DeviceMemoryInfo> GetMemoryInfo(DeviceId) override {
    return UnimplementedError("host memory discovery is not implemented");
  }
  StatusOr<void*> Allocate(DeviceId device, size_t bytes) override {
    INFERX_RETURN_IF_ERROR(SetDevice(device));
    if (bytes == 0) return static_cast<void*>(nullptr);
    const size_t padded =
        (bytes + kTensorAlignment - 1) / kTensorAlignment * kTensorAlignment;
    void* ptr = std::aligned_alloc(kTensorAlignment, padded);
    if (ptr == nullptr)
      return ResourceExhaustedError("failed to allocate ", bytes,
                                    " bytes of host memory");
    return ptr;
  }
  Status Free(DeviceId, void* ptr) override {
    std::free(ptr);
    return OkStatus();
  }
  StatusOr<void*> AllocatePinnedHost(size_t bytes) override {
    return Allocate(DeviceId::Cpu(), bytes);
  }
  Status FreePinnedHost(void* ptr) override {
    return Free(DeviceId::Cpu(), ptr);
  }
  Status Copy(void* dst, const void* src, size_t bytes, CopyKind) override {
    if (bytes != 0) std::memcpy(dst, src, bytes);
    return OkStatus();
  }
  Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                   Stream) override {
    return Copy(dst, src, bytes, kind);
  }
  StatusOr<Stream> CreateStream(DeviceId device) override {
    INFERX_RETURN_IF_ERROR(SetDevice(device));
    return Stream{};
  }
  Status DestroyStream(Stream) override { return OkStatus(); }
  Status SynchronizeStream(Stream) override { return OkStatus(); }
  StatusOr<DeviceEvent> CreateEvent(bool) override { return DeviceEvent{}; }
  Status DestroyEvent(DeviceEvent) override { return OkStatus(); }
  Status RecordEvent(DeviceEvent, Stream) override { return OkStatus(); }
  Status SynchronizeEvent(DeviceEvent) override { return OkStatus(); }
  StatusOr<bool> QueryEvent(DeviceEvent) override { return true; }
  StatusOr<float> ElapsedMs(DeviceEvent, DeviceEvent) override { return 0.0f; }
  StatusOr<bool> IsCapturing(Stream) override { return false; }
  Status BeginCapture(Stream) override {
    return UnimplementedError("CPU runtime does not support graph capture");
  }
  StatusOr<GraphExec> EndCaptureAndInstantiate(Stream) override {
    return UnimplementedError("CPU runtime does not support graph capture");
  }
  Status LaunchGraph(GraphExec, Stream) override {
    return UnimplementedError("CPU runtime does not support graphs");
  }
  Status DestroyGraph(GraphExec) override { return OkStatus(); }
  const RuntimeCapabilities& capabilities() const override { return caps_; }

 private:
  RuntimeCapabilities caps_{};
};

}  // namespace

namespace internal {

DeviceRuntime* CpuDeviceRuntime() {
  static DeviceRuntime* runtime = new CpuRuntime();
  return runtime;
}

}  // namespace internal
}  // namespace inferx
