# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/unit/engine
# Build directory: /home/gc/InferX/build-dbg/tests/unit/engine
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[request_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_request_test")
set_tests_properties([=[request_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;1;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
add_test([=[scheduler_output_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_scheduler_output_test")
set_tests_properties([=[scheduler_output_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;8;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
add_test([=[scheduler_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_scheduler_test")
set_tests_properties([=[scheduler_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;15;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
add_test([=[scheduler_prefix_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_scheduler_prefix_test")
set_tests_properties([=[scheduler_prefix_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;21;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
add_test([=[scheduler_preempt_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_scheduler_preempt_test")
set_tests_properties([=[scheduler_preempt_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;27;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
add_test([=[input_batch_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_input_batch_test")
set_tests_properties([=[input_batch_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;34;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
add_test([=[worker_test]=] "/home/gc/InferX/build-dbg/tests/unit/engine/inferx_worker_test")
set_tests_properties([=[worker_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;41;inferx_add_test;/home/gc/InferX/tests/unit/engine/CMakeLists.txt;0;")
