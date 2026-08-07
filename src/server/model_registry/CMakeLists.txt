add_library(inferx_model_registry registry.cc)
add_library(inferx::model_registry ALIAS inferx_model_registry)
target_include_directories(inferx_model_registry PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_model_registry PUBLIC inferx::core inferx::flags)
