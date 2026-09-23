// Declarations of every subcommand registrar. One function per subcommand;
// group registrars (bench, diagnostic, launch) call their children. Each is
// defined in the file named after the command, and root.cc calls the root
// ones. A registrar creates the subcommand, binds its options, and installs
// its callback; it must not bind options to stack locals (see throughput.cc).
#pragma once
#include <CLI/CLI.hpp>

namespace inferx::cli {

// Root commands.
void RegisterServe(CLI::App& root);
void RegisterChat(CLI::App& root);
void RegisterComplete(CLI::App& root);
void RegisterRunBatch(CLI::App& root);
void RegisterCollectEnv(CLI::App& root);
void RegisterLaunch(CLI::App& root);
void RegisterDiagnostic(CLI::App& root);
void RegisterBench(CLI::App& root);

// `bench` children (apps/cli/bench/*.cc).
void RegisterBenchLatency(CLI::App& parent);
void RegisterBenchThroughput(CLI::App& parent);
void RegisterBenchWorkload(CLI::App& parent);

// `diagnostic` children (apps/cli/diagnostic/*.cc).
void RegisterReplayLogits(CLI::App& parent);

// `launch` children (apps/cli/launch/*.cc).
void RegisterRenderLaunch(CLI::App& parent);

}  // namespace inferx::cli
