# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/regression
# Build directory: /home/gc/InferX/build-dbg/tests/regression
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[argmax_tie_test]=] "/home/gc/InferX/build-dbg/tests/regression/inferx_argmax_tie_test")
set_tests_properties([=[argmax_tie_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "regression" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/regression/CMakeLists.txt;5;inferx_add_test;/home/gc/InferX/tests/regression/CMakeLists.txt;0;")
add_test([=[vllm_parity_probe_test]=] "/home/gc/InferX/build-dbg/tests/regression/inferx_vllm_parity_probe_test")
set_tests_properties([=[vllm_parity_probe_test]=] PROPERTIES  ENVIRONMENT "LD_LIBRARY_PATH=/usr/lib/wsl/lib" LABELS "regression" TIMEOUT "300" WORKING_DIRECTORY "/home/gc/InferX" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/regression/CMakeLists.txt;13;inferx_add_test;/home/gc/InferX/tests/regression/CMakeLists.txt;0;")
