#include "inferx/core/device_runtime.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"

namespace inferx {
namespace {

// cudaMalloc and cudaHostAlloc guarantee at least 256 B alignment. Stronger
// requests would need over-allocation plus bookkeeping in Free(), so they are
// rejected instead of silently under-aligned.
constexpr size_t kMaxCudaAlignment = 256;

Status CudaError(cudaError_t err, const char* call) {
  return absl::InternalError(
      absl::StrCat(call, " failed: ", cudaGetErrorName(err), ": ", cudaGetErrorString(err)));
}

#define INFERX_CUDA_RETURN_IF_ERROR(expr)        \
  do {                                           \
    const cudaError_t inferx_cuda_err_ = (expr); \
    if (inferx_cuda_err_ != cudaSuccess) {       \
      return CudaError(inferx_cuda_err_, #expr); \
    }                                            \
  } while (false)

// Used where a destructor-like path cannot report an error.
void LogIfFailed(cudaError_t err, const char* call) {
  // At process exit the runtime may already be unloaded; that is not a bug.
  if (err == cudaSuccess || err == cudaErrorCudartUnloading) return;
  LOG(ERROR) << call << " failed: " << cudaGetErrorName(err) << ": " << cudaGetErrorString(err);
}

/// Makes `ordinal` current on this thread and restores the previous device on
/// scope exit. Skips the switch when the device is already current.
class DeviceScope {
 public:
  explicit DeviceScope(int ordinal) {
    err_ = cudaGetDevice(&previous_);
    if (err_ == cudaSuccess && previous_ != ordinal) {
      err_ = cudaSetDevice(ordinal);
      restore_ = (err_ == cudaSuccess);
    }
  }
  ~DeviceScope() {
    if (restore_) (void)cudaSetDevice(previous_);
  }
  DeviceScope(const DeviceScope&) = delete;
  DeviceScope& operator=(const DeviceScope&) = delete;

  bool ok() const { return err_ == cudaSuccess; }
  Status status() const { return ok() ? absl::OkStatus() : CudaError(err_, "cudaSetDevice"); }

 private:
  int previous_ = 0;
  bool restore_ = false;
  cudaError_t err_ = cudaSuccess;
};

// Every API method starts by making its device current for the call.
#define INFERX_SCOPE_DEVICE(ordinal)         \
  DeviceScope inferx_device_scope_(ordinal); \
  if (!inferx_device_scope_.ok()) return inferx_device_scope_.status()

// The device header's ordinal accessor; adapt here if yours is named differently.
int OrdinalOf(Device device) { return device.Index(); }

Status ValidateAlignment(size_t alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("alignment must be a power of two, got ", alignment));
  }
  if (alignment > kMaxCudaAlignment) {
    return absl::InvalidArgumentError(absl::StrCat("alignment ", alignment, " exceeds the ",
                                                   kMaxCudaAlignment,
                                                   " B the CUDA allocator guarantees"));
  }
  return absl::OkStatus();
}

cudaStream_t ToNative(DeviceStream s) { return s.As<cudaStream_t>(); }
cudaEvent_t ToNative(DeviceEvent e) { return e.As<cudaEvent_t>(); }
cudaGraphExec_t ToNative(DeviceGraphExec g) { return g.As<cudaGraphExec_t>(); }

cudaMemcpyKind ToNative(CopyKind kind) {
  switch (kind) {
    case CopyKind::kHostToHost:
      return cudaMemcpyHostToHost;
    case CopyKind::kHostToDevice:
      return cudaMemcpyHostToDevice;
    case CopyKind::kDeviceToHost:
      return cudaMemcpyDeviceToHost;
    case CopyKind::kDeviceToDevice:
      return cudaMemcpyDeviceToDevice;
  }
  return cudaMemcpyDefault;
}

cudaStreamCaptureMode ToNative(CaptureMode mode) {
  switch (mode) {
    case CaptureMode::kGlobal:
      return cudaStreamCaptureModeGlobal;
    case CaptureMode::kThreadLocal:
      return cudaStreamCaptureModeThreadLocal;
    case CaptureMode::kRelaxed:
      return cudaStreamCaptureModeRelaxed;
  }
  return cudaStreamCaptureModeGlobal;
}

class CudaMemory final : public MemoryApi {
 public:
  explicit CudaMemory(int ordinal) : ordinal_(ordinal) {}

  StatusOr<DeviceMemoryInfo> GetMemoryInfo() override {
    INFERX_SCOPE_DEVICE(ordinal_);
    size_t free_bytes = 0, total_bytes = 0;
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemGetInfo(&free_bytes, &total_bytes));
    return DeviceMemoryInfo{free_bytes, total_bytes};
  }

  StatusOr<void*> Allocate(size_t bytes, size_t alignment) override {
    if (Status s = ValidateAlignment(alignment); !s.ok()) return s;
    if (bytes == 0) return static_cast<void*>(nullptr);
    INFERX_SCOPE_DEVICE(ordinal_);
    void* ptr = nullptr;
    INFERX_CUDA_RETURN_IF_ERROR(cudaMalloc(&ptr, bytes));
    return ptr;
  }

  // Note: cudaFree implicitly synchronizes the device. Callers must not rely on
  // that; the contract is that they synchronize first.
  Status Free(void* ptr) override {
    if (ptr == nullptr) return absl::OkStatus();
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaFree(ptr));
    return absl::OkStatus();
  }

  // Portable, so the pinned memory is usable from every device's streams.
  StatusOr<void*> AllocatePinnedHost(size_t bytes, size_t alignment) override {
    if (Status s = ValidateAlignment(alignment); !s.ok()) return s;
    if (bytes == 0) return static_cast<void*>(nullptr);
    INFERX_SCOPE_DEVICE(ordinal_);
    void* ptr = nullptr;
    INFERX_CUDA_RETURN_IF_ERROR(cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable));
    return ptr;
  }

  Status FreePinnedHost(void* ptr) override {
    if (ptr == nullptr) return absl::OkStatus();
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaFreeHost(ptr));
    return absl::OkStatus();
  }

  Status RegisterHost(void* ptr, size_t bytes) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaHostRegister(ptr, bytes, cudaHostRegisterPortable));
    return absl::OkStatus();
  }

  Status UnregisterHost(void* ptr) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaHostUnregister(ptr));
    return absl::OkStatus();
  }

  Status Copy(void* dst, const void* src, size_t bytes, CopyKind kind) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpy(dst, src, bytes, ToNative(kind)));
    return absl::OkStatus();
  }

  Status CopyAsync(void* dst, const void* src, size_t bytes, CopyKind kind,
                   DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpyAsync(dst, src, bytes, ToNative(kind), ToNative(stream)));
    return absl::OkStatus();
  }

  Status MemsetAsync(void* dst, uint8_t value, size_t bytes, DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemsetAsync(dst, value, bytes, ToNative(stream)));
    return absl::OkStatus();
  }

 private:
  const int ordinal_;
};

class CudaStreams final : public StreamApi {
 public:
  explicit CudaStreams(int ordinal) : ordinal_(ordinal) {}

  StatusOr<OwnedStream> CreateStream(StreamPriority priority) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    int least = 0, greatest = 0;
    INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceGetStreamPriorityRange(&least, &greatest));
    const int native_priority = (priority == StreamPriority::kHigh) ? greatest : least;

    cudaStream_t stream = nullptr;
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, native_priority));
    return OwnedStream(DeviceStream(stream), this);
  }

  Status SynchronizeStream(DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(ToNative(stream)));
    return absl::OkStatus();
  }

  StatusOr<bool> QueryStream(DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    const cudaError_t err = cudaStreamQuery(ToNative(stream));
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    return CudaError(err, "cudaStreamQuery");
  }

  Status WaitEvent(DeviceStream stream, DeviceEvent event) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaStreamWaitEvent(ToNative(stream), ToNative(event), 0));
    return absl::OkStatus();
  }

  Status SynchronizeDevice() override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceSynchronize());
    return absl::OkStatus();
  }

  StatusOr<OwnedEvent> CreateEvent(bool timing) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    cudaEvent_t event = nullptr;
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaEventCreateWithFlags(&event, timing ? cudaEventDefault : cudaEventDisableTiming));
    return OwnedEvent(DeviceEvent(event), this);
  }

  Status RecordEvent(DeviceEvent event, DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaEventRecord(ToNative(event), ToNative(stream)));
    return absl::OkStatus();
  }

  Status SynchronizeEvent(DeviceEvent event) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaEventSynchronize(ToNative(event)));
    return absl::OkStatus();
  }

  StatusOr<bool> QueryEvent(DeviceEvent event) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    const cudaError_t err = cudaEventQuery(ToNative(event));
    if (err == cudaSuccess) return true;
    if (err == cudaErrorNotReady) return false;
    return CudaError(err, "cudaEventQuery");
  }

  StatusOr<float> ElapsedMs(DeviceEvent start, DeviceEvent end) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    float ms = 0.0f;
    INFERX_CUDA_RETURN_IF_ERROR(cudaEventElapsedTime(&ms, ToNative(start), ToNative(end)));
    return ms;
  }

 private:
  void Destroy(DeviceStream stream) noexcept override {
    DeviceScope scope(ordinal_);
    LogIfFailed(cudaStreamDestroy(ToNative(stream)), "cudaStreamDestroy");
  }

  void Destroy(DeviceEvent event) noexcept override {
    DeviceScope scope(ordinal_);
    LogIfFailed(cudaEventDestroy(ToNative(event)), "cudaEventDestroy");
  }

  const int ordinal_;
};

// -----------------------------------------------------------------------------
// GraphApi
// -----------------------------------------------------------------------------

// Destroys the capture's template graph once it has been instantiated or
// abandoned; only the executable graph outlives a capture.
class ScopedGraph {
 public:
  explicit ScopedGraph(cudaGraph_t graph) : graph_(graph) {}
  ~ScopedGraph() {
    if (graph_ != nullptr) LogIfFailed(cudaGraphDestroy(graph_), "cudaGraphDestroy");
  }
  ScopedGraph(const ScopedGraph&) = delete;
  ScopedGraph& operator=(const ScopedGraph&) = delete;
  cudaGraph_t get() const { return graph_; }

 private:
  cudaGraph_t graph_;
};

class CudaGraphs final : public GraphApi {
 public:
  explicit CudaGraphs(int ordinal) : ordinal_(ordinal) {}

  StatusOr<bool> IsCapturing(DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    INFERX_CUDA_RETURN_IF_ERROR(cudaStreamIsCapturing(ToNative(stream), &status));
    return status != cudaStreamCaptureStatusNone;
  }

  Status BeginCapture(DeviceStream stream, CaptureMode mode) override {
    // The null handle is the legacy default stream, which cannot be captured.
    if (stream == DeviceStream{}) {
      return absl::InvalidArgumentError(
          "BeginCapture: the default stream cannot be captured; create a stream");
    }
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaStreamBeginCapture(ToNative(stream), ToNative(mode)));
    return absl::OkStatus();
  }

  StatusOr<OwnedGraphExec> EndCaptureAndInstantiate(DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    cudaGraph_t captured = nullptr;
    // Capture ends here whether or not it succeeded; on an invalidated capture
    // this returns an error and no graph.
    INFERX_CUDA_RETURN_IF_ERROR(cudaStreamEndCapture(ToNative(stream), &captured));
    ScopedGraph guard(captured);

    cudaGraphExec_t exec = nullptr;
    INFERX_CUDA_RETURN_IF_ERROR(cudaGraphInstantiateWithFlags(&exec, guard.get(), 0));
    return OwnedGraphExec(DeviceGraphExec(exec), this);
  }

  Status EndCaptureAndUpdate(DeviceStream stream, DeviceGraphExec graph) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    cudaGraph_t captured = nullptr;
    INFERX_CUDA_RETURN_IF_ERROR(cudaStreamEndCapture(ToNative(stream), &captured));
    ScopedGraph guard(captured);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo info{};
    const cudaError_t err = cudaGraphExecUpdate(ToNative(graph), guard.get(), &info);
#else
    cudaGraphNode_t error_node = nullptr;
    cudaGraphExecUpdateResult result = cudaGraphExecUpdateSuccess;
    const cudaError_t err =
        cudaGraphExecUpdate(ToNative(graph), guard.get(), &error_node, &result);
#endif
    if (err == cudaErrorGraphExecUpdateFailure) {
      (void)cudaGetLastError();  // The failure is recoverable; clear it.
      return absl::FailedPreconditionError(
          "graph topology changed; re-instantiate with EndCaptureAndInstantiate");
    }
    INFERX_CUDA_RETURN_IF_ERROR(err);
    return absl::OkStatus();
  }

  Status LaunchGraph(DeviceGraphExec graph, DeviceStream stream) override {
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(cudaGraphLaunch(ToNative(graph), ToNative(stream)));
    return absl::OkStatus();
  }

 private:
  void Destroy(DeviceGraphExec graph) noexcept override {
    DeviceScope scope(ordinal_);
    LogIfFailed(cudaGraphExecDestroy(ToNative(graph)), "cudaGraphExecDestroy");
  }

  const int ordinal_;
};

// -----------------------------------------------------------------------------
// PeerApi
// -----------------------------------------------------------------------------

class CudaPeer final : public PeerApi {
 public:
  explicit CudaPeer(int ordinal) : ordinal_(ordinal) {}

  StatusOr<bool> CanAccess(Device peer) override {
    if (peer.Type() != DeviceType::kCuda) {
      return absl::InvalidArgumentError("CanAccess: peer is not a CUDA device");
    }
    const int peer_ordinal = OrdinalOf(peer);
    if (peer_ordinal == ordinal_) return true;
    int can_access = 0;
    INFERX_CUDA_RETURN_IF_ERROR(cudaDeviceCanAccessPeer(&can_access, ordinal_, peer_ordinal));
    return can_access != 0;
  }

  // cudaMemcpyPeerAsync works whether or not direct access is available; it
  // stages through the host when it is not.
  Status CopyAsync(void* dst, Device dst_device, const void* src, Device src_device,
                   size_t bytes, DeviceStream stream) override {
    if (dst_device.Type() != DeviceType::kCuda || src_device.Type() != DeviceType::kCuda) {
      return absl::InvalidArgumentError("peer copy endpoints must be CUDA devices");
    }
    const int dst_ordinal = OrdinalOf(dst_device);
    const int src_ordinal = OrdinalOf(src_device);
    if (dst_ordinal != ordinal_ && src_ordinal != ordinal_) {
      return absl::InvalidArgumentError(
          "peer copy must have this runtime's device as one endpoint");
    }
    INFERX_SCOPE_DEVICE(ordinal_);
    INFERX_CUDA_RETURN_IF_ERROR(
        cudaMemcpyPeerAsync(dst, dst_ordinal, src, src_ordinal, bytes, ToNative(stream)));
    return absl::OkStatus();
  }

 private:
  const int ordinal_;
};

class CudaRuntime final : public DeviceRuntime {
 public:
  explicit CudaRuntime(int ordinal)
      : DeviceRuntime(Device::Cuda(static_cast<int8_t>(ordinal))),
        ordinal_(ordinal),
        memory_(ordinal),
        streams_(ordinal),
        graphs_(ordinal),
        peer_(ordinal) {}

  // cudaGetDeviceProperties is slow; callers should cache what they need.
  StatusOr<DeviceProperties> GetProperties() override {
    cudaDeviceProp prop{};
    INFERX_CUDA_RETURN_IF_ERROR(cudaGetDeviceProperties(&prop, ordinal_));
    DeviceProperties out;
    out.name = prop.name;
    out.arch_major = prop.major;
    out.arch_minor = prop.minor;
    out.multiprocessor_count = prop.multiProcessorCount;
    out.total_memory_bytes = prop.totalGlobalMem;
    return out;
  }

  Status Activate() override {
    INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(ordinal_));
    return absl::OkStatus();
  }

  MemoryApi& Memory() override { return memory_; }
  StreamApi& Streams() override { return streams_; }
  GraphApi* Graphs() override { return &graphs_; }
  PeerApi* Peer() override { return &peer_; }

 private:
  const int ordinal_;
  CudaMemory memory_;
  CudaStreams streams_;
  CudaGraphs graphs_;
  CudaPeer peer_;
};

// 0 means the runtime is linked but no GPU is attached. Any other failure (for
// example a driver older than the runtime) is reported, not hidden as 0.
StatusOr<int> QueryDeviceCount() {
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorNoDevice) {
    (void)cudaGetLastError();
    return 0;
  }
  if (err != cudaSuccess) return CudaError(err, "cudaGetDeviceCount");
  return count;
}

class CudaProvider final : public RuntimeProvider {
 public:
  DeviceType GetDeviceType() const override { return DeviceType::kCuda; }

  StatusOr<int> DeviceCount() override { return QueryDeviceCount(); }

  StatusOr<DeviceRuntime*> GetRuntime(int ordinal) override {
    StatusOr<int> count = QueryDeviceCount();
    if (!count.ok()) return count.status();
    if (ordinal < 0 || ordinal >= *count) {
      return absl::InvalidArgumentError(absl::StrCat("CUDA device ordinal ", ordinal,
                                                     " is out of range; ", *count,
                                                     " device(s) available"));
    }

    std::lock_guard<std::mutex> lock(mu_);
    if (runtimes_.size() < static_cast<size_t>(*count)) runtimes_.resize(*count);
    if (runtimes_[ordinal] == nullptr) {
      runtimes_[ordinal] = std::make_unique<CudaRuntime>(ordinal);
    }
    return runtimes_[ordinal].get();
  }

 private:
  std::mutex mu_;
  std::vector<std::unique_ptr<CudaRuntime>> runtimes_;
};

}  // namespace

std::unique_ptr<RuntimeProvider> MakeCudaProvider() { return std::make_unique<CudaProvider>(); }

}  // namespace inferx