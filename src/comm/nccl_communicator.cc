#include "inferx/comm/nccl_communicator.h"

#include "nccl_backend.h"

namespace inferx::comm {

bool NcclBackendAvailable() {
#ifdef INFERX_WITH_NCCL
  return true;
#else
  return false;
#endif
}

StatusOr<NcclUniqueIdBytes> CreateNcclUniqueId() {
#ifdef INFERX_WITH_NCCL
  return internal::CreateCudaNcclUniqueId();
#else
  return UnimplementedError("InferX was built without NCCL");
#endif
}

StatusOr<std::unique_ptr<Communicator>> CreateNcclCommunicator(
    const NcclCommConfig& config) {
#ifdef INFERX_WITH_NCCL
  return internal::CreateCudaNcclCommunicator(config);
#else
  (void)config;
  return UnimplementedError("InferX was built without NCCL");
#endif
}

}  // namespace inferx::comm
