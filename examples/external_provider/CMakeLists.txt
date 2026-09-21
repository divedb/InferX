cmake_minimum_required(VERSION 3.24)
project(ExternalKvCacheProvider LANGUAGES C CXX)

# This project uses only the installed SDK. No InferX source/build paths.
find_package(KvCache 0.1 CONFIG REQUIRED)
add_library(example_provider MODULE provider.c)
target_link_libraries(example_provider PRIVATE kvc::abi)
set_target_properties(example_provider PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
add_executable(example_provider_smoke smoke.cc)
target_link_libraries(example_provider_smoke PRIVATE kvc::runtime)
enable_testing()
add_test(NAME external_provider_smoke COMMAND example_provider_smoke $<TARGET_FILE:example_provider>)
