#!/usr/bin/env bash
#
# Fetches submodules and configures a build tree.
#
#   ./scripts/bootstrap.sh              # default build/, CUDA required
#   ./scripts/bootstrap.sh --host-only  # no CUDA, for machines with no GPU
#   ./scripts/bootstrap.sh --debug      # Debug instead of RelWithDebInfo

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUILD_DIR="build"
BUILD_TYPE="RelWithDebInfo"
CMAKE_ARGS=()
WANT_ASAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host-only) CMAKE_ARGS+=(-DINFERX_ENABLE_CUDA=OFF); BUILD_DIR="build-host" ;;
    --debug)     BUILD_TYPE="Debug" ;;
    --asan)      CMAKE_ARGS+=(-DINFERX_ENABLE_ASAN=ON); BUILD_TYPE="Debug"; WANT_ASAN=1 ;;
    --dir)       shift; BUILD_DIR="$1" ;;
    *)           CMAKE_ARGS+=("$1") ;;
  esac
  shift
done

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

# GCC 13 cannot compile Abseil's hash containers under -fsanitize: its constexpr
# evaluator rejects absl::container_internal::TypeErasedApplyToSlotFn. Clang
# handles it, so prefer clang for sanitizer builds unless told otherwise.
if [[ $WANT_ASAN -eq 1 && ! "${CMAKE_ARGS[*]}" =~ CMAKE_CXX_COMPILER ]]; then
  for candidate in clang++-18 clang++; do
    if command -v "$candidate" >/dev/null 2>&1; then
      log "using $candidate for the sanitizer build (GCC + Abseil is broken here)"
      CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$candidate")
      break
    fi
  done
fi

log "syncing submodules"
git submodule update --init --recursive

log "configuring $BUILD_DIR ($BUILD_TYPE)"
cmake -S . -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  "${CMAKE_ARGS[@]}"

log "building"
cmake --build "$BUILD_DIR" -j "$(nproc)"

log "done -- build tree is $BUILD_DIR/"
