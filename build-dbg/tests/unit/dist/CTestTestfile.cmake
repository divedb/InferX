# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/unit/dist
# Build directory: /home/gc/InferX/build-dbg/tests/unit/dist
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[comm_test]=] "/home/gc/InferX/build-dbg/tests/unit/dist/inferx_comm_test")
set_tests_properties([=[comm_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "unit" TIMEOUT "30" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/dist/CMakeLists.txt;2;inferx_add_test;/home/gc/InferX/tests/unit/dist/CMakeLists.txt;0;")
add_test([=[topology_test]=] "/home/gc/InferX/build-dbg/tests/unit/dist/inferx_topology_test")
set_tests_properties([=[topology_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/dist/CMakeLists.txt;10;inferx_add_test;/home/gc/InferX/tests/unit/dist/CMakeLists.txt;0;")
add_test([=[shard_test]=] "/home/gc/InferX/build-dbg/tests/unit/dist/inferx_shard_test")
set_tests_properties([=[shard_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/dist/CMakeLists.txt;16;inferx_add_test;/home/gc/InferX/tests/unit/dist/CMakeLists.txt;0;")
