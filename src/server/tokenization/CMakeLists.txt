add_library(inferx_tokenization_service INTERFACE)
add_library(inferx::tokenization_service ALIAS inferx_tokenization_service)
target_include_directories(inferx_tokenization_service INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_tokenization_service INTERFACE inferx::tokenizer
  inferx::model_registry inferx::flags)
