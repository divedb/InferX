#pragma once

namespace inferx {

/// \brief Installs the process-wide fatal-signal handler.
///
/// Call this once, near the start of main(), before starting worker threads.
/// On SIGSEGV, SIGABRT, SIGBUS, SIGFPE, or SIGILL, the handler writes the
/// signal and a best-effort symbolized stack trace to stderr before terminating
/// the process. This also gives CHECK and LOG(FATAL) failures symbolized frames.
/// SIGINT and SIGTERM are deliberately not handled here; use
/// BlockShutdownSignals() and ShutdownSignalWaiter for graceful shutdown.
///
/// \param argv0 The executable name from main()'s argv[0], used to initialize
///              symbolization.
void InstallCrashHandler(const char* argv0);

}  // namespace inferx
