set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/googletest
                 ${CMAKE_BINARY_DIR}/third_party/googletest
                 EXCLUDE_FROM_ALL)

include(GoogleTest)

# Every test here must run to completion on a machine with no GPU. Tests that
# need a device gate themselves on inferx::CudaAvailable() and skip.
function(inferx_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::support
    inferx::model
    inferx::scheduler
    inferx::tokenizer
    inferx::api
    inferx::flags
    GTest::gtest_main
    GTest::gmock
  )
  gtest_discover_tests(${name}
    PROPERTIES LABELS unit
    DISCOVERY_TIMEOUT 30
  )
endfunction()

# Kernel tests link the device layer and are labelled separately, so that
# `ctest -L unit` stays the no-GPU-required suite. Like every GPU test they
# skip at runtime when no device is present -- the label is about what they
# link, the skip is about what is plugged in.
function(inferx_add_kernel_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE
    inferx::core
    inferx::support
    inferx::model
    inferx::scheduler
    inferx::kernels
    inferx::flags
    GTest::gtest_main
  )
  gtest_discover_tests(${name}
    PROPERTIES LABELS kernel
    DISCOVERY_TIMEOUT 30
  )
endfunction()

inferx_add_test(chat_template_test     unit/chat_template_test.cc)
inferx_add_test(safetensors_test       unit/safetensors_test.cc)
inferx_add_test(config_test            unit/config_test.cc)
inferx_add_test(scheduler_test         unit/scheduler_test.cc)
inferx_add_test(prefix_cache_test      unit/prefix_cache_test.cc)
inferx_add_test(log_test               unit/log_test.cc)
inferx_add_test(crash_test             unit/support/crash_test.cc)

if(INFERX_CUDA_MEETS_FLOOR)
  inferx_add_kernel_test(weight_loader_test kernel/weight_loader_test.cc)
  inferx_add_kernel_test(moe_test kernel/moe_test.cc)
  inferx_add_kernel_test(mla_test kernel/mla_test.cc)
  inferx_add_kernel_test(deepseek_v2_model_test kernel/deepseek_v2_model_test.cc)
  inferx_add_kernel_test(deepseek_v2_reference_test
                         kernel/deepseek_v2_reference_test.cc)
endif()

add_executable(inferx_transport_test
  unit/transport/transport_test.cc)
target_link_libraries(inferx_transport_test PRIVATE
  inferx::http_transport
  inferx::server_protocol
  inferx::admission
  inferx::auth
  inferx::server_config
  inferx::model_registry
  inferx::tokenization_service
  inferx::request_runtime
  inferx::managed_request_service
  inferx::streaming
  inferx::server_middleware
  inferx::request_handlers
  GTest::gtest_main)

gtest_discover_tests(inferx_transport_test
  PROPERTIES LABELS unit
  DISCOVERY_TIMEOUT 30)

add_executable(inferx_generation_test
  unit/server/generation_test.cc)
target_link_libraries(inferx_generation_test PRIVATE
  inferx::generation
  GTest::gtest_main)
gtest_discover_tests(inferx_generation_test
  PROPERTIES LABELS unit
  DISCOVERY_TIMEOUT 30)

add_executable(inferx_remote_scheduler_client_test
  unit/scheduler_client/remote_scheduler_client_test.cc)
target_link_libraries(inferx_remote_scheduler_client_test PRIVATE
  inferx::scheduler_client
  GTest::gtest_main)
gtest_discover_tests(inferx_remote_scheduler_client_test
  PROPERTIES LABELS unit
  DISCOVERY_TIMEOUT 30)

if(TARGET inferx_grpc_scheduler_transport)
  add_executable(inferx_scheduler_contract_compatibility_test
    unit/scheduler_client/scheduler_contract_compatibility_test.cc)
  target_link_libraries(inferx_scheduler_contract_compatibility_test PRIVATE
    inferx::grpc_scheduler_transport
    GTest::gtest_main)
  gtest_discover_tests(inferx_scheduler_contract_compatibility_test
    PROPERTIES LABELS unit
    DISCOVERY_TIMEOUT 30)

  add_executable(inferx_grpc_scheduler_transport_test
    integration/scheduler_client/grpc_scheduler_transport_test.cc)
  target_link_libraries(inferx_grpc_scheduler_transport_test PRIVATE
    inferx::grpc_scheduler_transport
    GTest::gtest_main)
  gtest_discover_tests(inferx_grpc_scheduler_transport_test
    PROPERTIES LABELS integration
    DISCOVERY_TIMEOUT 30)
endif()

if(TARGET inferx_gateway_server)
  add_executable(inferx_gateway_config_test
    unit/gateway/gateway_config_test.cc)
  target_link_libraries(inferx_gateway_config_test PRIVATE
    inferx::gateway_server
    GTest::gtest_main)
  gtest_discover_tests(inferx_gateway_config_test
    PROPERTIES LABELS unit
    DISCOVERY_TIMEOUT 30)
endif()
