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
if(EXISTS "/usr/lib/wsl/lib/libcuda.so.1")
  # WSL2 ships two libcuda copies; only the WSL one can see the GPU.
  set_tests_properties(rms_norm_test model_test PROPERTIES ENVIRONMENT
                       "LD_LIBRARY_PATH=/usr/lib/wsl/lib")
endif()
