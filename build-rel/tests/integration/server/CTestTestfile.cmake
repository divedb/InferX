# CMake generated Testfile for 
# Source directory: /home/gc/InferX/tests/integration/server
# Build directory: /home/gc/InferX/build-rel/tests/integration/server
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[tokenizer_pool_test]=] "/home/gc/InferX/build-rel/tests/integration/server/inferx_tokenizer_pool_test")
set_tests_properties([=[tokenizer_pool_test]=] PROPERTIES  LABELS "integration" TIMEOUT "120" WORKING_DIRECTORY "/home/gc/InferX" _BACKTRACE_TRIPLES "/home/gc/InferX/cmake/InferxHelpers.cmake;169;add_test;/home/gc/InferX/tests/integration/server/CMakeLists.txt;4;inferx_add_test;/home/gc/InferX/tests/integration/server/CMakeLists.txt;0;")
