# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/integration/models
# Build directory: /home/gc/InferX/build-dbg/tests/integration/models
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[model_test]=] "/home/gc/InferX/build-dbg/tests/integration/models/inferx_model_test")
set_tests_properties([=[model_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "integration" WORKING_DIRECTORY "/home/gc/InferX" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/integration/models/CMakeLists.txt;4;inferx_add_test;/home/gc/InferX/tests/integration/models/CMakeLists.txt;0;")
add_test([=[tensor_parallel_test]=] "/home/gc/InferX/build-dbg/tests/integration/models/inferx_tensor_parallel_test")
set_tests_properties([=[tensor_parallel_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "integration" TIMEOUT "300" WORKING_DIRECTORY "/home/gc/InferX" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/integration/models/CMakeLists.txt;13;inferx_add_test;/home/gc/InferX/tests/integration/models/CMakeLists.txt;0;")
