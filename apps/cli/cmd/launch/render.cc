// `inferx launch render` — GPU-less rendering server; not implemented yet.
#include <CLI/CLI.hpp>

#include "cli/commands.h"
#include "cli/error.h"

namespace inferx::cli {

void RegisterRenderLaunch(CLI::App& parent) {
  CLI::App* sub = parent.add_subcommand(
      "render", "Launch a GPU-less rendering server (preprocessing and postprocessing only).");
  sub->callback([] { ThrowIfError(UnimplementedError("launch render is not implemented yet")); });
}

}  // namespace inferx::cli
