# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/unit/core
# Build directory: /home/gc/InferX/build-rel/tests/unit/core
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[tensor_test]=] "/home/gc/InferX/build-rel/tests/unit/core/inferx_tensor_test")
set_tests_properties([=[tensor_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/core/CMakeLists.txt;1;inferx_add_test;/home/gc/InferX/tests/unit/core/CMakeLists.txt;0;")
add_test([=[logging_test]=] "/home/gc/InferX/build-rel/tests/unit/core/inferx_logging_test")
set_tests_properties([=[logging_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/core/CMakeLists.txt;9;inferx_add_test;/home/gc/InferX/tests/unit/core/CMakeLists.txt;0;")
