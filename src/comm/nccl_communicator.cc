#include "inferx/comm/nccl_communicator.h"

#include <cstring>
#include <memory>
#include <utility>

#if defined(INFERX_WITH_NCCL)
#include <cuda_runtime_api.h>
#include <nccl.h>

#include "inferx/backends/cuda/cuda_utils.h"
#endif

namespace inferx::comm {

#if defined(INFERX_WITH_NCCL)
namespace {

Status NcclStatus(ncclResult_t result, const char* operation) {
  if (result == ncclSuccess) return OkStatus();
  return InternalError(operation, " failed: ", ncclGetErrorString(result));
}

StatusOr<ncclDataType_t> NcclDataType(DataType dtype) {
  switch (dtype) {
    case DataType::kBFloat16:
      return ncclBfloat16;
    case DataType::kFloat:
      return ncclFloat;
    case DataType::kDouble:
      return ncclDouble;
    case DataType::kInt32:
      return ncclInt32;
    case DataType::kInt64:
      return ncclInt64;
    default:
      return UnimplementedError("NCCL all-reduce does not support dtype ",
                                DataTypeName(dtype));
  }
}

class NcclComm final : public Communicator {
 public:
  NcclComm(int rank, int size, DeviceId device, ncclComm_t comm)
      : rank_(rank), size_(size), device_(device), comm_(comm) {}

  ~NcclComm() override {
    if (comm_ == nullptr) return;
    (void)cudaSetDevice(static_cast<int>(device_.index));
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 14, 0)
    (void)ncclCommFinalize(comm_);
#endif
    (void)ncclCommDestroy(comm_);
  }

  int rank() const override { return rank_; }
  int size() const override { return size_; }
  DeviceId device() const override { return device_; }
  CommBackend backend() const override { return CommBackend::kNccl; }
  CommCapabilities capabilities() const override {
    return {.cuda_graph_capture = true, .device_collectives = true};
  }

  Status AllReduceSum(const TensorView& tensor, Stream stream) override {
    if (!tensor.IsDefined()) {
      return InvalidArgumentError("NCCL AllReduceSum: tensor is undefined");
    }
    if (!tensor.IsCuda() || tensor.Device() != device_) {
      return InvalidArgumentError("NCCL AllReduceSum: tensor is on ",
                                  tensor.Device().ToString(),
                                  ", communicator is on ", device_.ToString());
    }
    if (tensor.Numel() == 0) return OkStatus();

    INFERX_ASSIGN_OR_RETURN(const ncclDataType_t dtype,
                            NcclDataType(tensor.GetDataType()));
    auto cuda_stream = static_cast<cudaStream_t>(stream);
    return NcclStatus(ncclAllReduce(tensor.Data(), tensor.Data(),
                                    static_cast<size_t>(tensor.Numel()), dtype,
                                    ncclSum, comm_, cuda_stream),
                      "ncclAllReduce");
  }

  Status Abort() override {
    if (comm_ == nullptr) return OkStatus();
    ncclComm_t comm = std::exchange(comm_, nullptr);
    return NcclStatus(ncclCommAbort(comm), "ncclCommAbort");
  }

 private:
  int rank_;
  int size_;
  DeviceId device_;
  ncclComm_t comm_ = nullptr;
};

}  // namespace
#endif

bool NcclBackendAvailable() {
#if defined(INFERX_WITH_NCCL)
  return true;
#else
  return false;
#endif
}

StatusOr<NcclUniqueIdBytes> CreateNcclUniqueId() {
#if defined(INFERX_WITH_NCCL)
  static_assert(sizeof(ncclUniqueId) == kNcclUniqueIdBytes);
  ncclUniqueId id;
  INFERX_RETURN_IF_ERROR(NcclStatus(ncclGetUniqueId(&id), "ncclGetUniqueId"));
  NcclUniqueIdBytes bytes;
  std::memcpy(bytes.data(), &id, sizeof(id));
  return bytes;
#else
  return UnimplementedError("InferX was built without NCCL");
#endif
}

StatusOr<std::unique_ptr<Communicator>> CreateNcclCommunicator(
    const NcclCommConfig& config) {
#if defined(INFERX_WITH_NCCL)
  if (config.world_size <= 1) {
    return InvalidArgumentError("NCCL world_size must be greater than 1, got ",
                                config.world_size);
  }
  if (config.rank < 0 || config.rank >= config.world_size) {
    return InvalidArgumentError("NCCL rank ", config.rank,
                                " is outside world size ", config.world_size);
  }
  int device_count = 0;
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetDeviceCount(&device_count));
  if (config.local_device < 0 || config.local_device >= device_count) {
    return InvalidArgumentError("CUDA device ", config.local_device,
                                " is outside available device count ",
                                device_count);
  }
  INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(config.local_device));

  ncclUniqueId id;
  std::memcpy(&id, config.unique_id.data(), sizeof(id));
  ncclComm_t raw = nullptr;
  INFERX_RETURN_IF_ERROR(
      NcclStatus(ncclCommInitRank(&raw, config.world_size, id, config.rank),
                 "ncclCommInitRank"));
  return std::unique_ptr<Communicator>(new NcclComm(
      config.rank, config.world_size,
      DeviceId::Cuda(static_cast<int8_t>(config.local_device)), raw));
#else
  (void)config;
  return UnimplementedError("InferX was built without NCCL");
#endif
}

}  // namespace inferx::comm
