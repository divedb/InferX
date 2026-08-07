add_library(inferx_tokenization_service tokenization_service.cc)
add_library(inferx::tokenization_service ALIAS inferx_tokenization_service)
target_include_directories(inferx_tokenization_service PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
target_link_libraries(inferx_tokenization_service PUBLIC inferx::tokenizer
  inferx::model_registry inferx::folly_coro inferx::flags)
