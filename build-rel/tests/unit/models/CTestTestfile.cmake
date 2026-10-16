# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/unit/models
# Build directory: /home/gc/InferX/build-rel/tests/unit/models
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[safe_tensors_view_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_safe_tensors_view_test")
set_tests_properties([=[safe_tensors_view_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;7;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
add_test([=[safe_tensors_reader_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_safe_tensors_reader_test")
set_tests_properties([=[safe_tensors_reader_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;14;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
add_test([=[scratch_arena_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_scratch_arena_test")
set_tests_properties([=[scratch_arena_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;21;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
add_test([=[checkpoint_reader_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_checkpoint_reader_test")
set_tests_properties([=[checkpoint_reader_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;29;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
add_test([=[model_config_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_model_config_test")
set_tests_properties([=[model_config_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;36;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
add_test([=[model_registry_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_model_registry_test")
set_tests_properties([=[model_registry_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;43;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
add_test([=[qwen3_forward_test]=] "/home/gc/InferX/build-rel/tests/unit/models/inferx_qwen3_forward_test")
set_tests_properties([=[qwen3_forward_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;50;inferx_add_test;/home/gc/InferX/tests/unit/models/CMakeLists.txt;0;")
