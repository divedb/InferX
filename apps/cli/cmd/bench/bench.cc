// `inferx bench` — pure wiring: the group owns no state and has no callback;
// CLI11 runs the selected child's callback because callbacks fire bottom-up
// along the parsed chain. RegisterBenchServe / Startup / Sweep / MMProcessor
// join here as they are implemented.
#include "cli/commands.h"

namespace inferx::cli {

void RegisterBench(CLI::App& root) {
  CLI::App* bench = root.add_subcommand("bench", "Run performance benchmarks.");
  bench->require_subcommand(1);  // exactly one child; bare `inferx bench` is a
                                 // usage error listing the children
  RegisterBenchLatency(*bench);
  RegisterBenchThroughput(*bench);
  RegisterBenchWorkload(*bench);  // InferX extension, not in vLLM's tree
}

}  // namespace inferx::cli
