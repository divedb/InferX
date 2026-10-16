# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/unit/server
# Build directory: /home/gc/InferX/build-rel/tests/unit/server
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[server_api_test]=] "/home/gc/InferX/build-rel/tests/unit/server/inferx_server_api_test")
set_tests_properties([=[server_api_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/server/CMakeLists.txt;1;inferx_add_test;/home/gc/InferX/tests/unit/server/CMakeLists.txt;0;")
add_test([=[event_queue_test]=] "/home/gc/InferX/build-rel/tests/unit/server/inferx_event_queue_test")
set_tests_properties([=[event_queue_test]=] PROPERTIES  LABELS "unit" TIMEOUT "30" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/server/CMakeLists.txt;8;inferx_add_test;/home/gc/InferX/tests/unit/server/CMakeLists.txt;0;")
add_test([=[text_delta_test]=] "/home/gc/InferX/build-rel/tests/unit/server/inferx_text_delta_test")
set_tests_properties([=[text_delta_test]=] PROPERTIES  LABELS "unit" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/unit/server/CMakeLists.txt;16;inferx_add_test;/home/gc/InferX/tests/unit/server/CMakeLists.txt;0;")
