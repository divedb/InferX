# Shared CLI console: help rendering, error translation, styled output.
inferx_add_library(
  NAME cli_presentation
  HDRS
    cli/console.h
    cli/terminal.h
  SRCS
    cli/console.cc
    cli/terminal.cc
  INCLUDE_DIRS
    .
  DEPS CLI11::CLI11 fmt::fmt rapidfuzz::rapidfuzz
)

# Every subcommand registrar plus the app shell. Business errors and command
# execution stay out of the presentation layer.
inferx_add_library(
  NAME cli
  HDRS
    cli/app.h
    cli/commands.h
  SRCS
    cli/app.cc
    cli/cmd/bench/bench.cc
    cli/cmd/bench/latency.cc
    cli/cmd/bench/throughput.cc
    cli/cmd/bench/workload.cc
    cli/cmd/chat.cc
    cli/cmd/collect_env.cc
    cli/cmd/complete.cc
    cli/diagnostic/diagnostic.cc
    cli/diagnostic/replay_logits.cc
    cli/cmd/help.cc
    cli/cmd/launch/launch.cc
    cli/cmd/launch/render.cc
    cli/cmd/run_batch.cc
    cli/cmd/serve.cc
    cli/cmd/version.cc
  DEPS inferx::cli_presentation inferx::bench inferx::diagnostic inferx::server
)

add_executable(inferx main.cc)
target_compile_definitions(inferx PRIVATE INFERX_VERSION="${PROJECT_VERSION}")
target_link_libraries(inferx PRIVATE inferx::cli)
target_compile_options(inferx PRIVATE -Wall -Wextra -Wpedantic)
