#include <cuda_runtime_api.h>

#include "../../core/device_runtime_internal.h"
#include "inferx/core/cuda_utils.h"

namespace inferx {
namespace {

#define CUDA_STATUS(expr) CudaErrorToStatus((expr), #expr, __FILE__, __LINE__)
cudaStream_t Native(Stream stream) {
  return static_cast<cudaStream_t>(stream.handle);
}
cudaEvent_t Native(DeviceEvent event) {
  return static_cast<cudaEvent_t>(event.handle);
}

class CudaRuntime final : public DeviceRuntime {
 public:
  DeviceKind kind() const override { return DeviceKind::kCuda; }
  int DeviceCount() const override { return CudaDeviceCount(); }
  Status SetDevice(DeviceId device) override {
    if (!device.IsCuda())
      return InvalidArgumentError("CUDA runtime cannot select ",
                                  device.ToString());
    return CUDA_STATUS(cudaSetDevice(device.index));
  }
  StatusOr<DeviceMemoryInfo> GetMemoryInfo(DeviceId device) override {
    INFERX_RETURN_IF_ERROR(SetDevice(device));
    size_t free = 0;
    size_t total = 0;
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(cudaMemGetInfo(&free, &total)));
    return DeviceMemoryInfo{free, total};
  }
  StatusOr<void*> Allocate(DeviceId device, size_t bytes) override {
    INFERX_RETURN_IF_ERROR(SetDevice(device));
    void* ptr = nullptr;
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(cudaMalloc(&ptr, bytes)));
    return ptr;
  }
  Status Free(DeviceId device, void* ptr) override {
    if (!ptr) return OkStatus();
    INFERX_RETURN_IF_ERROR(SetDevice(device));
    return CUDA_STATUS(cudaFree(ptr));
  }
  StatusOr<void*> AllocatePinnedHost(size_t bytes) override {
    void* ptr = nullptr;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault)));
    return ptr;
  }
  Status FreePinnedHost(void* ptr) override {
    return CUDA_STATUS(cudaFreeHost(ptr));
  }
  Status Copy(void* dst, const void* src, size_t bytes,
              CopyKind kind) override {
    return CUDA_STATUS(cudaMemcpy(dst, src, bytes, CopyType(kind)));
  }
  Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                   Stream stream) override {
    return CUDA_STATUS(
        cudaMemcpyAsync(dst, src, bytes, CopyType(kind), Native(stream)));
  }
  StatusOr<Stream> CreateStream(DeviceId device) override {
    INFERX_RETURN_IF_ERROR(SetDevice(device));
    cudaStream_t stream;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)));
    return Stream{static_cast<void*>(stream)};
  }
  Status DestroyStream(Stream stream) override {
    return CUDA_STATUS(cudaStreamDestroy(Native(stream)));
  }
  Status SynchronizeStream(Stream stream) override {
    return CUDA_STATUS(cudaStreamSynchronize(Native(stream)));
  }
  StatusOr<DeviceEvent> CreateEvent(bool timing) override {
    cudaEvent_t event;
    unsigned flags = timing ? cudaEventDefault : cudaEventDisableTiming;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaEventCreateWithFlags(&event, flags)));
    return DeviceEvent{static_cast<void*>(event)};
  }
  Status DestroyEvent(DeviceEvent event) override {
    return CUDA_STATUS(cudaEventDestroy(Native(event)));
  }
  Status RecordEvent(DeviceEvent event, Stream stream) override {
    return CUDA_STATUS(cudaEventRecord(Native(event), Native(stream)));
  }
  Status SynchronizeEvent(DeviceEvent event) override {
    return CUDA_STATUS(cudaEventSynchronize(Native(event)));
  }
  StatusOr<bool> QueryEvent(DeviceEvent event) override {
    auto result = cudaEventQuery(Native(event));
    if (result == cudaSuccess) return true;
    if (result == cudaErrorNotReady) return false;
    return CUDA_STATUS(result);
  }
  StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) override {
    float milliseconds;
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(
        cudaEventElapsedTime(&milliseconds, Native(start), Native(end))));
    return milliseconds;
  }
  StatusOr<bool> IsCapturing(Stream stream) override {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaStreamIsCapturing(Native(stream), &capture)));
    return capture != cudaStreamCaptureStatusNone;
  }
  Status BeginCapture(Stream stream) override {
    return CUDA_STATUS(cudaStreamBeginCapture(
        Native(stream), cudaStreamCaptureModeThreadLocal));
  }
  StatusOr<GraphExec> EndCaptureAndInstantiate(Stream stream) override {
    cudaGraph_t graph = nullptr;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaStreamEndCapture(Native(stream), &graph)));
    cudaGraphExec_t executable = nullptr;
    const cudaError_t instantiated =
        cudaGraphInstantiate(&executable, graph, 0);
    const cudaError_t destroyed = cudaGraphDestroy(graph);
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(instantiated));
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(destroyed));
    return GraphExec{static_cast<void*>(executable)};
  }
  Status LaunchGraph(GraphExec graph, Stream stream) override {
    return CUDA_STATUS(cudaGraphLaunch(
        static_cast<cudaGraphExec_t>(graph.handle), Native(stream)));
  }
  Status DestroyGraph(GraphExec graph) override {
    if (graph.handle == nullptr) return OkStatus();
    return CUDA_STATUS(
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph.handle)));
  }
  const RuntimeCapabilities& capabilities() const override { return caps_; }

 private:
  static cudaMemcpyKind CopyType(CopyKind kind) {
    switch (kind) {
      case CopyKind::kHostToDevice:
        return cudaMemcpyHostToDevice;
      case CopyKind::kDeviceToHost:
        return cudaMemcpyDeviceToHost;
      case CopyKind::kDeviceToDevice:
        return cudaMemcpyDeviceToDevice;
    }
    return cudaMemcpyDefault;
  }

  RuntimeCapabilities caps_{true, true, true};
};
#undef CUDA_STATUS

}  // namespace

namespace internal {

DeviceRuntime* CudaDeviceRuntime() {
  static DeviceRuntime* runtime = new CudaRuntime();
  return runtime;
}

}  // namespace internal
}  // namespace inferx
