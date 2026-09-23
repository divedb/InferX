#include "cli/app.h"

#include <cstdio>
#include <exception>
#include <string>

#include "cli/commands.h"
#include "cli/error.h"
#include "cli/presentation.h"

namespace inferx::cli {

void InferxCli::RegisterCommands() {
  RegisterServe(app_);
  RegisterComplete(app_);
  RegisterChat(app_);
  RegisterRunBatch(app_);
  RegisterCollectEnv(app_);
  RegisterLaunch(app_);
  RegisterDiagnostic(app_);
  RegisterBench(app_);
}

int InferxCli::Run(int argc, char** argv) {
  Presentation presentation(app_, std::cout, std::cerr,
                            term::IsTerminal(stdout), term::IsTerminal(stderr));
  try {
    return presentation.Parse(argc, argv);
  } catch (const CommandError& e) {
    presentation.Error(e.what());
    return 1;
  } catch (const std::exception& e) {
    presentation.Error(std::string("internal error: ") + e.what());
    return 1;
  }
}

}  // namespace inferx::cli
