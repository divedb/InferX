// `inferx launch` — pure wiring; children live in launch/*.cc.
#include "cli/commands.h"

namespace inferx::cli {

void RegisterLaunch(CLI::App& root) {
  CLI::App* launch = root.add_subcommand("launch", "Launch individual InferX components.");
  launch->require_subcommand(1);
  RegisterRenderLaunch(*launch);
}

}  // namespace inferx::cli
