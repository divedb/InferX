add_library(inferx_cli_presentation cli/presentation.cc cli/terminal.cc)
add_library(inferx::cli_presentation ALIAS inferx_cli_presentation)
target_include_directories(inferx_cli_presentation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(inferx_cli_presentation PUBLIC CLI11::CLI11)
target_compile_features(inferx_cli_presentation PUBLIC cxx_std_23)

add_library(inferx_cli
  cli/app.cc
  cli/bench/bench.cc 
  cli/bench/latency.cc 
  cli/bench/throughput.cc
  cli/bench/workload.cc
  cli/chat.cc 
  cli/collect_env.cc 
  cli/complete.cc
  cli/diagnostic/diagnostic.cc 
  cli/diagnostic/replay_logits.cc
  cli/launch/launch.cc 
  cli/launch/render.cc
  cli/run_batch.cc 
  cli/serve.cc
)

add_library(inferx::cli ALIAS inferx_cli)
target_include_directories(inferx_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(inferx_cli
  PUBLIC inferx::cli_presentation inferx::bench inferx::diagnostic inferx::server)
target_compile_options(inferx_cli PRIVATE -Wall -Wextra -Wpedantic)

add_executable(inferx main.cc)
target_compile_definitions(inferx PRIVATE INFERX_VERSION="${PROJECT_VERSION}")
target_link_libraries(inferx PRIVATE inferx::cli)
target_compile_options(inferx PRIVATE -Wall -Wextra -Wpedantic)
