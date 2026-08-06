add_library(inferx_comm
  communicator.cc
  nccl_communicator.cc
  tensor_parallel.cc
)
add_library(inferx::comm ALIAS inferx_comm)

target_include_directories(inferx_comm PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(inferx_comm PUBLIC
  inferx::core
  inferx::flags
)

set(INFERX_HAVE_NCCL OFF)
if(INFERX_ENABLE_NCCL AND INFERX_ENABLE_CUDA AND INFERX_CUDA_MEETS_FLOOR)
  find_path(INFERX_NCCL_INCLUDE_DIR nccl.h)
  find_library(INFERX_NCCL_LIBRARY NAMES nccl)
  if(INFERX_NCCL_INCLUDE_DIR AND INFERX_NCCL_LIBRARY)
    set(INFERX_HAVE_NCCL ON)
    target_compile_definitions(inferx_comm PRIVATE INFERX_WITH_NCCL=1)
    target_include_directories(inferx_comm PRIVATE ${INFERX_NCCL_INCLUDE_DIR})
    target_link_libraries(inferx_comm PRIVATE
      CUDA::cudart
      ${INFERX_NCCL_LIBRARY}
    )
  endif()
endif()

message(STATUS "  NCCL backend    : ${INFERX_HAVE_NCCL}")
