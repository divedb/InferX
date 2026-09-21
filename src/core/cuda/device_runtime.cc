// CUDA DeviceRuntime: allocation, copies, streams, events, and graph capture.
//
// Compiled only when the CUDA toolkit is available. The runtime exposes the
// same mechanical services as the CPU runtime so the ops and tensor layers
// remain device-agnostic.

#include "inferx/core/device_runtime.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

#include "../device_runtime_internal.h"
#include "absl/status/status.h"
#include "inferx/core/device.h"

namespace inferx::internal {
namespace {

Status CudaError(cudaError_t code, const char* what) {
  if (code == cudaSuccess) return OkStatus();
  return InternalError(what, " failed: ", cudaGetErrorString(code));
}

class CudaDeviceRuntime final : public DeviceRuntime {
 public:
  explicit CudaDeviceRuntime(DeviceId device) : DeviceRuntime(device) {}

  int DeviceCount() const override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
  }

  Status Activate() override {
    return CudaError(cudaSetDevice(device().index), "cudaSetDevice");
  }

  StatusOr<DeviceMemoryInfo> GetMemoryInfo() override {
    INFERX_RETURN_IF_ERROR(Activate());
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo"));
    return DeviceMemoryInfo{free_bytes, total_bytes};
  }

  StatusOr<void*> Allocate(size_t bytes) override {
    if (bytes == 0) return static_cast<void*>(nullptr);
    INFERX_RETURN_IF_ERROR(Activate());
    void* ptr = nullptr;
    INFERX_RETURN_IF_ERROR(CudaError(cudaMalloc(&ptr, bytes), "cudaMalloc"));
    return ptr;
  }

  Status Free(void* ptr) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaFree(ptr), "cudaFree");
  }

  StatusOr<void*> AllocatePinnedHost(size_t bytes) override {
    if (bytes == 0) return static_cast<void*>(nullptr);
    INFERX_RETURN_IF_ERROR(Activate());
    void* ptr = nullptr;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault), "cudaHostAlloc"));
    return ptr;
  }

  Status FreePinnedHost(void* ptr) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaFreeHost(ptr), "cudaFreeHost");
  }

  Status Copy(void* dst, const void* src, size_t bytes, CopyKind kind) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaMemcpy(dst, src, bytes, ToCudaKind(kind)), "cudaMemcpy");
  }

  Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                   Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(
        cudaMemcpyAsync(dst, src, bytes, ToCudaKind(kind), static_cast<cudaStream_t>(stream)),
        "cudaMemcpyAsync");
  }

  StatusOr<Stream> CreateStream() override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaStream_t stream = nullptr;
    INFERX_RETURN_IF_ERROR(CudaError(cudaStreamCreate(&stream), "cudaStreamCreate"));
    return Stream(stream);
  }

  Status DestroyStream(Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaStreamDestroy(static_cast<cudaStream_t>(stream)), "cudaStreamDestroy");
  }

  Status SynchronizeStream(Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
                     "cudaStreamSynchronize");
  }

  StatusOr<DeviceEvent> CreateEvent(bool timing) override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaEvent_t event = nullptr;
    const unsigned int flags = timing ? cudaEventDefault : cudaEventDisableTiming;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaEventCreateWithFlags(&event, flags), "cudaEventCreate"));
    return DeviceEvent{event};
  }

  Status DestroyEvent(DeviceEvent event) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaEventDestroy(static_cast<cudaEvent_t>(event.handle)),
                     "cudaEventDestroy");
  }

  Status RecordEvent(DeviceEvent event, Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaEventRecord(static_cast<cudaEvent_t>(event.handle),
                                     static_cast<cudaStream_t>(stream)),
                     "cudaEventRecord");
  }

  Status SynchronizeEvent(DeviceEvent event) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaEventSynchronize(static_cast<cudaEvent_t>(event.handle)),
                     "cudaEventSynchronize");
  }

  StatusOr<bool> QueryEvent(DeviceEvent event) override {
    INFERX_RETURN_IF_ERROR(Activate());
    const cudaError_t code = cudaEventQuery(static_cast<cudaEvent_t>(event.handle));
    if (code == cudaSuccess) return true;
    if (code == cudaErrorNotReady) return false;
    return CudaError(code, "cudaEventQuery");
  }

  StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) override {
    INFERX_RETURN_IF_ERROR(Activate());
    float ms = 0.0f;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start.handle),
                                       static_cast<cudaEvent_t>(end.handle)),
                  "cudaEventElapsedTime"));
    return ms;
  }

  StatusOr<bool> IsCapturing(Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream), &status),
                  "cudaStreamIsCapturing"));
    return status != cudaStreamCaptureStatusNone;
  }

  Status BeginCapture(Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaStreamBeginCapture(static_cast<cudaStream_t>(stream),
                                            cudaStreamCaptureModeThreadLocal),
                     "cudaStreamBeginCapture");
  }

  StatusOr<GraphExec> EndCaptureAndInstantiate(Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaGraph_t graph = nullptr;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaStreamEndCapture(static_cast<cudaStream_t>(stream), &graph),
                  "cudaStreamEndCapture"));
    cudaGraphExec_t exec = nullptr;
    const cudaError_t code = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    INFERX_RETURN_IF_ERROR(CudaError(code, "cudaGraphInstantiate"));
    return GraphExec{exec};
  }

  Status LaunchGraph(GraphExec graph, Stream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph.handle),
                                     static_cast<cudaStream_t>(stream)),
                     "cudaGraphLaunch");
  }

  Status DestroyGraph(GraphExec graph) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph.handle)),
                     "cudaGraphExecDestroy");
  }

  const RuntimeCapabilities& capabilities() const override { return capabilities_; }

 private:
  static cudaMemcpyKind ToCudaKind(CopyKind kind) {
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

  RuntimeCapabilities capabilities_{true, true, false};
};

}  // namespace

StatusOr<DeviceRuntime*> AcceleratorDeviceRuntime(DeviceId device) {
  if (!device.IsCuda()) {
    return FailedPreconditionError("no runtime is built for ", device.ToString());
  }
  static std::mutex mu;
  static std::map<int, std::unique_ptr<CudaDeviceRuntime>> runtimes;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = runtimes.find(device.index);
  if (it != runtimes.end()) return it->second.get();

  int count = 0;
  INFERX_RETURN_IF_ERROR(CudaError(cudaGetDeviceCount(&count), "cudaGetDeviceCount"));
  if (device.index < 0 || device.index >= count) {
    return InvalidArgumentError("invalid device ordinal for ", device.ToString());
  }
  auto runtime = std::make_unique<CudaDeviceRuntime>(device);
  auto* result = runtime.get();
  runtimes.emplace(device.index, std::move(runtime));
  return result;
}

}  // namespace inferx::internal
