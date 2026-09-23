# Build only the HuggingFace JSON tokenizer path. The upstream CMake project
# also builds SentencePiece and its bundled Abseil, conflicting with ours.
find_program(CARGO_EXECUTABLE cargo REQUIRED)
find_package(Threads REQUIRED)
set(TK_SRC "${PROJECT_SOURCE_DIR}/third_party/tokenizers-cpp")
set(TK_CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/rust")
if(MSVC)
  set(TK_RUST_LIB "${TK_CARGO_TARGET_DIR}/release/tokenizers_c.lib")
else()
  set(TK_RUST_LIB "${TK_CARGO_TARGET_DIR}/release/libtokenizers_c.a")
endif()

# Let Cargo track Rust sources, manifests, lockfiles, and build scripts itself.
# Its incremental build avoids recompilation when inputs have not changed.
# Retain the release Rust profile for all CMake configurations.
add_custom_target(tokenizers_rust
  COMMAND "${CMAKE_COMMAND}" -E env "CARGO_TARGET_DIR=${TK_CARGO_TARGET_DIR}"
          "${CARGO_EXECUTABLE}" build --release
          --manifest-path "${TK_SRC}/rust/Cargo.toml"
  BYPRODUCTS "${TK_RUST_LIB}"
  WORKING_DIRECTORY "${TK_SRC}/rust"
  COMMENT "Building tokenizers-cpp Rust core"
  VERBATIM
)

add_library(inferx_tokenizers_rust STATIC IMPORTED)
set_target_properties(inferx_tokenizers_rust PROPERTIES
  IMPORTED_LOCATION "${TK_RUST_LIB}")
add_dependencies(inferx_tokenizers_rust tokenizers_rust)

add_library(inferx_tokenizers STATIC "${TK_SRC}/src/huggingface_tokenizer.cc")
add_library(inferx::tokenizers ALIAS inferx_tokenizers)
target_include_directories(inferx_tokenizers PUBLIC "${TK_SRC}/include")
target_link_libraries(inferx_tokenizers
  PRIVATE inferx_tokenizers_rust Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(inferx_tokenizers PUBLIC INFERX_HAVE_TOKENIZERS)
if(NOT MSVC)
  target_compile_options(inferx_tokenizers PRIVATE -Wall -Wextra)
endif()
