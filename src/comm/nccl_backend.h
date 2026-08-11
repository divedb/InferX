#pragma once

#include <memory>

#include "inferx/comm/nccl_communicator.h"

namespace inferx::comm::internal {

StatusOr<NcclUniqueIdBytes> CreateCudaNcclUniqueId();
StatusOr<std::unique_ptr<Communicator>> CreateCudaNcclCommunicator(
    const NcclCommConfig& config);

}  // namespace inferx::comm::internal
