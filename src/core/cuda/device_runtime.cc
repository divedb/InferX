#include "inferx/core/device_runtime.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

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
  explicit CudaDeviceRuntime(Device device) : DeviceRuntime(device) {}

  int DeviceCount() const override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return 0;
    return count;
  }

  Status Activate() override {
    return CudaError(cudaSetDevice(device().index()), "cudaSetDevice");
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
                   DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(
        cudaMemcpyAsync(dst, src, bytes, ToCudaKind(kind), stream.As<cudaStream_t>()),
        "cudaMemcpyAsync");
  }

  Status MemsetAsync(void* dst, size_t bytes, DeviceStream stream) override {
    const cudaError_t err = cudaMemsetAsync(dst, 0, bytes, stream.As<cudaStream_t>());
    return err == cudaSuccess
               ? OkStatus()
               : InternalError("cudaMemsetAsync failed: ", cudaGetErrorString(err));
  }

  StatusOr<DeviceStream> CreateStream() override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaStream_t stream = nullptr;
    INFERX_RETURN_IF_ERROR(CudaError(cudaStreamCreate(&stream), "cudaStreamCreate"));
    return DeviceStream(stream);
  }

  Status DestroyStream(DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaStreamDestroy(stream.As<cudaStream_t>()), "cudaStreamDestroy");
  }

  Status SynchronizeStream(DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaStreamSynchronize(stream.As<cudaStream_t>()),
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
    return CudaError(cudaEventDestroy(event.As<cudaEvent_t>()),
                     "cudaEventDestroy");
  }

  Status RecordEvent(DeviceEvent event, DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaEventRecord(event.As<cudaEvent_t>(),
                                     stream.As<cudaStream_t>()),
                     "cudaEventRecord");
  }

  Status SynchronizeEvent(DeviceEvent event) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaEventSynchronize(event.As<cudaEvent_t>()),
                     "cudaEventSynchronize");
  }

  StatusOr<bool> QueryEvent(DeviceEvent event) override {
    INFERX_RETURN_IF_ERROR(Activate());
    const cudaError_t code = cudaEventQuery(event.As<cudaEvent_t>());
    if (code == cudaSuccess) return true;
    if (code == cudaErrorNotReady) return false;
    return CudaError(code, "cudaEventQuery");
  }

  StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) override {
    INFERX_RETURN_IF_ERROR(Activate());
    float ms = 0.0f;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaEventElapsedTime(&ms, start.As<cudaEvent_t>(),
                                       end.As<cudaEvent_t>()),
                  "cudaEventElapsedTime"));
    return ms;
  }

  StatusOr<bool> IsCapturing(DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaStreamIsCapturing(stream.As<cudaStream_t>(), &status),
                  "cudaStreamIsCapturing"));
    return status != cudaStreamCaptureStatusNone;
  }

  Status BeginCapture(DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaStreamBeginCapture(stream.As<cudaStream_t>(),
                                            cudaStreamCaptureModeThreadLocal),
                     "cudaStreamBeginCapture");
  }

  StatusOr<DeviceGraphExec> EndCaptureAndInstantiate(DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    cudaGraph_t graph = nullptr;
    INFERX_RETURN_IF_ERROR(
        CudaError(cudaStreamEndCapture(stream.As<cudaStream_t>(), &graph),
                  "cudaStreamEndCapture"));
    cudaGraphExec_t exec = nullptr;
    const cudaError_t code = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);
    INFERX_RETURN_IF_ERROR(CudaError(code, "cudaGraphInstantiate"));
    return DeviceGraphExec{exec};
  }

  Status LaunchGraph(DeviceGraphExec graph, DeviceStream stream) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaGraphLaunch(graph.As<cudaGraphExec_t>(),
                                     stream.As<cudaStream_t>()),
                     "cudaGraphLaunch");
  }

  Status DestroyGraph(DeviceGraphExec graph) override {
    INFERX_RETURN_IF_ERROR(Activate());
    return CudaError(cudaGraphExecDestroy(graph.As<cudaGraphExec_t>()),
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

StatusOr<DeviceRuntime*> AcceleratorDeviceRuntime(Device device) {
  if (!device.IsCuda()) {
    return FailedPreconditionError("no runtime is built for ", device.ToString());
  }
  static std::mutex mu;
  static std::map<int, std::unique_ptr<CudaDeviceRuntime>> runtimes;
  std::lock_guard<std::mutex> lock(mu);
  const auto it = runtimes.find(device.index());
  if (it != runtimes.end()) return it->second.get();

  int count = 0;
  INFERX_RETURN_IF_ERROR(CudaError(cudaGetDeviceCount(&count), "cudaGetDeviceCount"));
  if (device.index() < 0 || device.index() >= count) {
    return InvalidArgumentError("invalid device ordinal for ", device.ToString());
  }
  auto runtime = std::make_unique<CudaDeviceRuntime>(device);
  auto* result = runtime.get();
  runtimes.emplace(device.index(), std::move(runtime));
  return result;
}

}  // namespace inferx::internal
