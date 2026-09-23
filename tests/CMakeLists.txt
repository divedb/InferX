include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(inferx_scheduler_output_test scheduler_output_test.cc)
target_link_libraries(inferx_scheduler_output_test PRIVATE inferx::engine
                      GTest::gtest_main)
target_compile_options(inferx_scheduler_output_test PRIVATE -Wall -Wextra
                       -Wpedantic)
add_test(NAME scheduler_output_test COMMAND inferx_scheduler_output_test)

add_executable(inferx_scheduler_test scheduler_test.cc)
target_link_libraries(inferx_scheduler_test PRIVATE inferx::engine
                      GTest::gtest_main)
target_compile_options(inferx_scheduler_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME scheduler_test COMMAND inferx_scheduler_test)

add_executable(inferx_model_test model_test.cc)
target_link_libraries(inferx_model_test PRIVATE inferx::models GTest::gtest_main)
target_compile_options(inferx_model_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME model_test COMMAND inferx_model_test
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(inferx_model_runner_test model_runner_test.cc)
target_link_libraries(inferx_model_runner_test PRIVATE inferx::engine
                      GTest::gtest_main)
target_compile_options(inferx_model_runner_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME model_runner_test COMMAND inferx_model_runner_test)

add_executable(inferx_rms_norm_test rms_norm_test.cc)
target_link_libraries(inferx_rms_norm_test PRIVATE inferx::ops GTest::gtest_main)
target_compile_options(inferx_rms_norm_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME rms_norm_test COMMAND inferx_rms_norm_test)

add_executable(inferx_ops_test ops_test.cc)
target_link_libraries(inferx_ops_test PRIVATE inferx::ops GTest::gtest_main)
target_compile_options(inferx_ops_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME ops_test COMMAND inferx_ops_test)
if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
  # WSL2 ships two libcuda copies; only the WSL one can see the GPU.
  set_tests_properties(rms_norm_test model_test ops_test PROPERTIES ENVIRONMENT
                       "LD_LIBRARY_PATH=/usr/lib/wsl/lib")
endif()

add_executable(inferx_cli_args_test cli_args_test.cc)
target_link_libraries(inferx_cli_args_test PRIVATE inferx::cli GTest::gtest_main)
target_compile_options(inferx_cli_args_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME cli_args_test COMMAND inferx_cli_args_test)

add_executable(inferx_cli_presentation_test cli_presentation_test.cc)
target_link_libraries(inferx_cli_presentation_test
  PRIVATE inferx::cli_presentation GTest::gtest_main)
add_test(NAME cli_presentation_test COMMAND inferx_cli_presentation_test)

add_executable(inferx_server_api_test server_api_test.cc)
target_link_libraries(inferx_server_api_test
  PRIVATE inferx::server nlohmann_json::nlohmann_json GTest::gtest_main)
target_compile_options(inferx_server_api_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME server_api_test COMMAND inferx_server_api_test)

add_executable(inferx_event_queue_test event_queue_test.cc)
target_link_libraries(inferx_event_queue_test
  PRIVATE inferx::server GTest::gtest_main)
target_compile_options(inferx_event_queue_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME event_queue_test COMMAND inferx_event_queue_test)
set_tests_properties(event_queue_test PROPERTIES TIMEOUT 30)

add_executable(inferx_text_delta_test text_delta_test.cc)
target_link_libraries(inferx_text_delta_test
  PRIVATE inferx::server GTest::gtest_main)
target_compile_options(inferx_text_delta_test PRIVATE -Wall -Wextra
                       -Wpedantic)
add_test(NAME text_delta_test COMMAND inferx_text_delta_test)

if(INFERX_ENABLE_TOKENIZERS)
  # Integration tests for the prompt-preparation worker pool (token-for-token
  # parity with the facade, admission bounds, cancellation, timeouts, drain).
  add_executable(inferx_tokenizer_pool_test tokenizer_pool_test.cc)
  target_link_libraries(inferx_tokenizer_pool_test
    PRIVATE inferx::server GTest::gtest_main)
  target_compile_options(inferx_tokenizer_pool_test PRIVATE -Wall -Wextra
                         -Wpedantic)
  add_test(NAME tokenizer_pool_test COMMAND inferx_tokenizer_pool_test
           WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
  set_tests_properties(tokenizer_pool_test PROPERTIES TIMEOUT 120)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME cli_output_test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/cli_output_test.py
          $<TARGET_FILE:inferx>)

add_executable(inferx_sampling_test sampling_test.cc)
target_link_libraries(inferx_sampling_test PRIVATE inferx::sampling
                      GTest::gtest_main)
target_compile_options(inferx_sampling_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME sampling_test COMMAND inferx_sampling_test)

add_executable(inferx_attention_test attention_test.cc)
target_link_libraries(inferx_attention_test PRIVATE inferx::ops GTest::gtest_main)
target_compile_options(inferx_attention_test PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME attention_test COMMAND inferx_attention_test)
if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
  set_tests_properties(attention_test PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib")
endif()
