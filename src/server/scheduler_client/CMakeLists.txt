add_library(inferx_scheduler_client
  in_process_scheduler_client.cc
  remote_scheduler_client.cc)
add_library(inferx::scheduler_client ALIAS inferx_scheduler_client)
target_include_directories(inferx_scheduler_client PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_scheduler_client PUBLIC inferx::request_runtime
  inferx::folly_coro inferx::flags)

if(INFERX_ENABLE_GRPC)
  find_package(Protobuf QUIET)
  find_package(gRPC CONFIG QUIET)
  if(Protobuf_FOUND AND gRPC_FOUND)
    set(_inferx_scheduler_proto
        ${PROJECT_SOURCE_DIR}/proto/inference/scheduler/v1/scheduler.proto)
    set(_inferx_scheduler_generated_dir
        ${CMAKE_CURRENT_BINARY_DIR}/generated)
    set(_inferx_scheduler_generated_base
        ${_inferx_scheduler_generated_dir}/inference/scheduler/v1/scheduler)
    add_custom_command(
      OUTPUT
        ${_inferx_scheduler_generated_base}.pb.cc
        ${_inferx_scheduler_generated_base}.pb.h
        ${_inferx_scheduler_generated_base}.grpc.pb.cc
        ${_inferx_scheduler_generated_base}.grpc.pb.h
      COMMAND ${CMAKE_COMMAND} -E make_directory
              ${_inferx_scheduler_generated_dir}
      COMMAND protobuf::protoc
              --proto_path=${PROJECT_SOURCE_DIR}/proto
              --cpp_out=${_inferx_scheduler_generated_dir}
              --grpc_out=${_inferx_scheduler_generated_dir}
              --plugin=protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
              ${_inferx_scheduler_proto}
      DEPENDS ${_inferx_scheduler_proto} protobuf::protoc
              gRPC::grpc_cpp_plugin
      VERBATIM)

    add_library(inferx_grpc_scheduler_transport
      grpc_scheduler_transport.cc
      ${_inferx_scheduler_generated_base}.pb.cc
      ${_inferx_scheduler_generated_base}.grpc.pb.cc)
    add_library(inferx::grpc_scheduler_transport ALIAS
                inferx_grpc_scheduler_transport)
    target_include_directories(inferx_grpc_scheduler_transport PUBLIC
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
      $<BUILD_INTERFACE:${_inferx_scheduler_generated_dir}>)
    target_link_libraries(inferx_grpc_scheduler_transport PUBLIC
      inferx::scheduler_client
      protobuf::libprotobuf
      gRPC::grpc++)
    target_compile_definitions(inferx_grpc_scheduler_transport PUBLIC
      INFERX_WITH_GRPC_SCHEDULER=1)
    message(STATUS "gRPC scheduler transport: enabled")
  else()
    message(STATUS "gRPC scheduler transport: disabled (protobuf/gRPC not found)")
  endif()
endif()
