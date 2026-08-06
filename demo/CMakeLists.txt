# Standalone demos that exercise FlashInfer kernel templates directly (no
# TVM-FFI / PyTorch binding layer).
#
# Built only when both CUDA device code and the FlashInfer headers are present,
# since every demo .cu instantiates templates from include/flashinfer/.

if(NOT INFERX_CUDA_MEETS_FLOOR)
  return()
endif()

if(NOT INFERX_HAVE_FLASHINFER)
  message(STATUS "demo: FlashInfer not found -- demos skipped")
  return()
endif()

function(inferx_add_demo name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::flags
    inferx::flashinfer
    CUDA::cudart
  )
  set_target_properties(${name} PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/demo"
  )
endfunction()

inferx_add_demo(single_prefill_demo single_prefill_demo.cu)
