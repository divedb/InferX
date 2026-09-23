// `inferx diagnostic` — pure wiring; children live in diagnostic/*.cc.
#include "cli/commands.h"

namespace inferx::cli {

void RegisterDiagnostic(CLI::App& root) {
  CLI::App* diagnostic = root.add_subcommand("diagnostic", "Inspect model execution and numerical results.");
  diagnostic->require_subcommand(1);
  RegisterReplayLogits(*diagnostic);
}

}  // namespace inferx::cli
