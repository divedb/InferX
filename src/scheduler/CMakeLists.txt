# inferx_scheduler -- request lifecycle and batching policy.
#
# Host-only by construction, and that is the point: §3.1 makes the scheduler the
# only component with policy and the only one that never touches CUDA, so its
# whole lifecycle is testable on a machine with no GPU. Nothing here may gain a
# device dependency.

add_library(inferx_scheduler
  scheduler.cc
)
add_library(inferx::scheduler ALIAS inferx_scheduler)

target_link_libraries(inferx_scheduler
  PUBLIC
    inferx::core
    inferx::model
    inferx::flags
)
