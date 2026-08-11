#include "inferx/core/device_runtime.h"

#include <cstdlib>
#include <cstring>

#ifdef INFERX_WITH_CUDA
#include <cuda_runtime_api.h>

#include "inferx/core/cuda_utils.h"
#endif

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

#ifdef INFERX_WITH_CUDA
#define CUDA_STATUS(expr) CudaErrorToStatus((expr), #expr, __FILE__, __LINE__)
cudaStream_t Native(Stream s) { return static_cast<cudaStream_t>(s.handle); }
cudaEvent_t Native(DeviceEvent e) { return static_cast<cudaEvent_t>(e.handle); }

class CudaRuntime final : public DeviceRuntime {
 public:
  DeviceKind kind() const override { return DeviceKind::kCuda; }
  int DeviceCount() const override { return CudaDeviceCount(); }
  Status SetDevice(DeviceId d) override {
    if (!d.IsCuda())
      return InvalidArgumentError("CUDA runtime cannot select ", d.ToString());
    return CUDA_STATUS(cudaSetDevice(d.index));
  }
  StatusOr<DeviceMemoryInfo> GetMemoryInfo(DeviceId d) override {
    INFERX_RETURN_IF_ERROR(SetDevice(d));
    size_t free = 0, total = 0;
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(cudaMemGetInfo(&free, &total)));
    return DeviceMemoryInfo{free, total};
  }
  StatusOr<void*> Allocate(DeviceId d, size_t n) override {
    INFERX_RETURN_IF_ERROR(SetDevice(d));
    void* p = nullptr;
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(cudaMalloc(&p, n)));
    return p;
  }
  Status Free(DeviceId d, void* p) override {
    if (!p) return OkStatus();
    INFERX_RETURN_IF_ERROR(SetDevice(d));
    return CUDA_STATUS(cudaFree(p));
  }
  StatusOr<void*> AllocatePinnedHost(size_t n) override {
    void* p = nullptr;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaHostAlloc(&p, n, cudaHostAllocDefault)));
    return p;
  }
  Status FreePinnedHost(void* p) override {
    return CUDA_STATUS(cudaFreeHost(p));
  }
  Status Copy(void* d, const void* s, size_t n, CopyKind k) override {
    return CUDA_STATUS(cudaMemcpy(d, s, n, CopyType(k)));
  }
  Status CopyAsync(void* d, const void* s, size_t n, CopyKind k,
                   Stream st) override {
    return CUDA_STATUS(cudaMemcpyAsync(d, s, n, CopyType(k), Native(st)));
  }
  StatusOr<Stream> CreateStream(DeviceId d) override {
    INFERX_RETURN_IF_ERROR(SetDevice(d));
    cudaStream_t s;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking)));
    return Stream{static_cast<void*>(s)};
  }
  Status DestroyStream(Stream s) override {
    return CUDA_STATUS(cudaStreamDestroy(Native(s)));
  }
  Status SynchronizeStream(Stream s) override {
    return CUDA_STATUS(cudaStreamSynchronize(Native(s)));
  }
  StatusOr<DeviceEvent> CreateEvent(bool timing) override {
    cudaEvent_t e;
    unsigned flags = timing ? cudaEventDefault : cudaEventDisableTiming;
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(cudaEventCreateWithFlags(&e, flags)));
    return DeviceEvent{static_cast<void*>(e)};
  }
  Status DestroyEvent(DeviceEvent e) override {
    return CUDA_STATUS(cudaEventDestroy(Native(e)));
  }
  Status RecordEvent(DeviceEvent e, Stream s) override {
    return CUDA_STATUS(cudaEventRecord(Native(e), Native(s)));
  }
  Status SynchronizeEvent(DeviceEvent e) override {
    return CUDA_STATUS(cudaEventSynchronize(Native(e)));
  }
  StatusOr<bool> QueryEvent(DeviceEvent e) override {
    auto r = cudaEventQuery(Native(e));
    if (r == cudaSuccess) return true;
    if (r == cudaErrorNotReady) return false;
    return CUDA_STATUS(r);
  }
  StatusOr<float> ElapsedMs(DeviceEvent a, DeviceEvent b) override {
    float ms;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaEventElapsedTime(&ms, Native(a), Native(b))));
    return ms;
  }
  StatusOr<bool> IsCapturing(Stream s) override {
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaStreamIsCapturing(Native(s), &capture)));
    return capture != cudaStreamCaptureStatusNone;
  }
  Status BeginCapture(Stream s) override {
    return CUDA_STATUS(
        cudaStreamBeginCapture(Native(s), cudaStreamCaptureModeThreadLocal));
  }
  StatusOr<GraphExec> EndCaptureAndInstantiate(Stream s) override {
    cudaGraph_t graph = nullptr;
    INFERX_RETURN_IF_ERROR(
        CUDA_STATUS(cudaStreamEndCapture(Native(s), &graph)));
    cudaGraphExec_t executable = nullptr;
    const cudaError_t instantiated =
        cudaGraphInstantiate(&executable, graph, 0);
    const cudaError_t destroyed = cudaGraphDestroy(graph);
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(instantiated));
    INFERX_RETURN_IF_ERROR(CUDA_STATUS(destroyed));
    return GraphExec{static_cast<void*>(executable)};
  }
  Status LaunchGraph(GraphExec graph, Stream s) override {
    return CUDA_STATUS(
        cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph.handle), Native(s)));
  }
  Status DestroyGraph(GraphExec graph) override {
    if (graph.handle == nullptr) return OkStatus();
    return CUDA_STATUS(
        cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph.handle)));
  }
  const RuntimeCapabilities& capabilities() const override { return caps_; }

 private:
  static cudaMemcpyKind CopyType(CopyKind k) {
    switch (k) {
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
#endif

}  // namespace

StatusOr<DeviceRuntime*> RuntimeFor(DeviceId device) {
  static DeviceRuntime* cpu = new CpuRuntime();
  if (device.IsCpu()) return cpu;
#ifdef INFERX_WITH_CUDA
  static DeviceRuntime* cuda = new CudaRuntime();
  if (device.IsCuda()) return cuda;
#endif
  return FailedPreconditionError("no runtime is built for ", device.ToString());
}

}  // namespace inferx
