# Collectives sit behind dist/comm.h: SingleRankComm (identity), LoopbackComm
# (in-process virtual ranks, the sharded-execution reference), and NcclComm
# (dlopen'd libnccl for multi-process worlds).
inferx_add_library(
  NAME comm
  SRCS
    loopback_comm.cc
    nccl_comm.cc
  DEPS inferx::ops
  PRIVATE_DEPS ${CMAKE_DL_LIBS}
)

# inferx_dist -- tensor-parallel worker topology: the controller-side
# process pool (posix_spawn, one worker per device), the IPC wire protocol
# over Boost.InterProcess message queues, and the worker-side engine loop.
inferx_add_library(
  NAME dist
  SRCS
    worker_ipc.cc
    worker_launch.cc
    worker_main.cc
  DEPS
    inferx::engine
    inferx::boost
    absl::status absl::statusor absl::strings
  PRIVATE_DEPS ${CMAKE_DL_LIBS}
)
