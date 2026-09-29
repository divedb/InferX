#include "cli/app.h"

#include <exception>
#include <string>

#include "cli/commands.h"
#include "cli/console.h"
#include "cli/error.h"

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
  RegisterHelp(app_);
  RegisterVersion(app_);
}

int InferxCli::ParseCommandLine(int argc, const char* const argv[]) { return 0; }

int InferxCli::Run(int argc, char** argv) {
  Console console(app_, std::cout, std::cerr);
  try {
    return console.Parse(argc, argv);
  } catch (const CommandError& e) {
    console.Error(e.what());
    return 1;
  } catch (const std::exception& e) {
    console.Error(std::string("internal error: ") + e.what());
    return 1;
  }
}

}  // namespace inferx::cli
