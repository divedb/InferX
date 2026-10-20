# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/unit/cache
# Build directory: /home/gc/InferX/build-dbg/tests/unit/cache
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[cache_spec_test]=] "/home/gc/InferX/build-dbg/tests/unit/cache/inferx_cache_spec_test")
set_tests_properties([=[cache_spec_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;1;inferx_add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;0;")
add_test([=[kv_block_pool_test]=] "/home/gc/InferX/build-dbg/tests/unit/cache/inferx_kv_block_pool_test")
set_tests_properties([=[kv_block_pool_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;7;inferx_add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;0;")
add_test([=[cache_group_manager_test]=] "/home/gc/InferX/build-dbg/tests/unit/cache/inferx_cache_group_manager_test")
set_tests_properties([=[cache_group_manager_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;13;inferx_add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;0;")
add_test([=[cache_manager_test]=] "/home/gc/InferX/build-dbg/tests/unit/cache/inferx_cache_manager_test")
set_tests_properties([=[cache_manager_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;19;inferx_add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;0;")
add_test([=[sha256_test]=] "/home/gc/InferX/build-dbg/tests/unit/cache/inferx_sha256_test")
set_tests_properties([=[sha256_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;25;inferx_add_test;/home/gc/InferX/tests/unit/cache/CMakeLists.txt;0;")
