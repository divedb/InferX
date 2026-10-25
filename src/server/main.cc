/// \file
/// \brief `inferx-server` binary stub (Phase 0 scaffold).
///
/// The real server main lands with the Phase 1 HTTP core; until then this
/// keeps the `inferx_server_main` target (§3.5) buildable and prints the
/// Phase status so operators get a clear answer instead of a missing
/// binary.

#include <cstdio>

int main(int argc, char** argv) {
  if (argc > 1) {
    std::fprintf(stderr, "inferx-server: takes no arguments (Phase 0 stub)\n");
    return 2;
  }
  std::printf("inferx-server: serving stack not implemented yet "
              "(docs/server_design.md §14, Phase 1); use `inferx serve` "
              "once Phase 1 lands\n");
  return 0;
}
