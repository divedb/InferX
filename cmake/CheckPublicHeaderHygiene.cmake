if(NOT DEFINED INFERX_PUBLIC_INCLUDE_DIR)
  message(FATAL_ERROR "INFERX_PUBLIC_INCLUDE_DIR is required")
endif()

file(GLOB_RECURSE inferx_public_headers
  "${INFERX_PUBLIC_INCLUDE_DIR}/inferx/*.h")

foreach(header IN LISTS inferx_public_headers)
  if(header MATCHES "/inferx/backends/")
    continue()
  endif()

  file(READ "${header}" contents)
  if(contents MATCHES
     "#[ \t]*include[ \t]*[<\"](cuda|cublas|nccl|hip|acl)[^>\"]*[>\"]")
    message(FATAL_ERROR "Vendor header leaked into public API: ${header}")
  endif()
  if(contents MATCHES
     "(cuda(Stream|Event|Graph|Error)_t|hip(Stream|Event|Graph|Error)_t|aclrt(Stream|Event))")
    message(FATAL_ERROR "Vendor runtime type leaked into public API: ${header}")
  endif()
endforeach()
