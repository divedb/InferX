# inferx_model -- checkpoint loading and model definitions.
#
# The loader half is host-only and GPU-free on purpose: it reads files and
# produces host tensors, so loader conformance tests run without a device, the
# same way the scheduler will (§3.1).
#
# The model half needs the kernels and therefore the device, so it is added only
# above the CUDA floor. A --host-only build still gets the loader and the config
# parser, which is most of what there is to get wrong about a checkpoint.

add_library(inferx_model
  config.cc
  forward_batch.cc
  safetensors.cc
)
add_library(inferx::model ALIAS inferx_model)

target_link_libraries(inferx_model
  PUBLIC
    inferx::core
    inferx::common
    inferx::flags
)

if(INFERX_CUDA_MEETS_FLOOR)
  target_sources(inferx_model PRIVATE qwen2.cc moe_ffn.cc mla.cc gpt_oss.cc)
  target_link_libraries(inferx_model PUBLIC inferx::kernels)
endif()
