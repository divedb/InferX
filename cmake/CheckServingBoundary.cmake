# CheckServingBoundary.cmake
#
# CTest script: fails when the serving boundary header (§16.1) grows
# transport or server-side includes. Scans #include lines only, so prose in
# comments mentioning Asio/Beast/HTTP stays legal.
#
# Usage: cmake -DHEADER=<path> -P CheckServingBoundary.cmake

if(NOT DEFINED HEADER)
  message(FATAL_ERROR "CheckServingBoundary: -DHEADER=<path> is required")
endif()

if(NOT EXISTS "${HEADER}")
  message(FATAL_ERROR "CheckServingBoundary: header not found: ${HEADER}")
endif()

file(READ "${HEADER}" content)

string(REGEX MATCHALL
  "#[ \t]*include[ \t]*[\"<][^\">\r\n]*(boost|asio|beast|http|inferx/server)[\">]"
  banned_includes "${content}")

if(banned_includes)
  list(JOIN banned_includes " " joined)
  message(FATAL_ERROR
    "serving boundary header must not include transport/server headers "
    "(docs/server_design.md §9.1): ${joined}")
endif()

string(REGEX MATCHALL "namespace[ \t]+(asio|beast|boost)[ \t:]*" banned_namespaces "${content}")
if(banned_namespaces)
  message(FATAL_ERROR
    "serving boundary header must not use transport namespaces: ${banned_namespaces}")
endif()

message(STATUS "serving boundary purity: OK (${HEADER})")
