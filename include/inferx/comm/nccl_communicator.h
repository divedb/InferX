#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "inferx/comm/communicator.h"
#include "inferx/core/status.h"

namespace inferx::comm {

// NCCL's bootstrap identifier is an opaque 128-byte token. Keeping the bytes
// here prevents nccl.h from leaking through InferX's public/model interfaces.
inline constexpr size_t kNcclUniqueIdBytes = 128;
using NcclUniqueIdBytes = std::array<std::byte, kNcclUniqueIdBytes>;

struct NcclCommConfig {
  int rank = 0;
  int world_size = 1;
  int local_device = 0;
  NcclUniqueIdBytes unique_id{};
};

/// True when this binary was linked with the optional NCCL backend.
bool NcclBackendAvailable();

/// Creates the bootstrap token once on the coordinator and copies it to every
/// rank configuration. Returns Unimplemented in a build without NCCL.
StatusOr<NcclUniqueIdBytes> CreateNcclUniqueId();

/// Creates one communicator rank bound permanently to `local_device`.
StatusOr<std::unique_ptr<Communicator>> CreateNcclCommunicator(
    const NcclCommConfig& config);

}  // namespace inferx::comm
