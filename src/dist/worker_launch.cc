#include "inferx/dist/worker_launch.h"

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <csignal>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <utility>

#include "inferx/core/logging.h"

extern char** environ;

namespace inferx {
namespace dist {

StatusOr<std::string> SelfExePath() {
  char path[4096] = {0};
  const ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
  if (len <= 0) {
    return InternalError("could not resolve /proc/self/exe");
  }
  return std::string(path, static_cast<size_t>(len));
}

WorkerPool::~WorkerPool() {
  if (!waited_ && !pids_.empty()) {
    KillAll();
    (void)WaitAll();
  }
}

StatusOr<WorkerPool> WorkerPool::Launch(absl::Span<const WorkerSpec> specs,
                                        const std::string& worker_exe,
                                        absl::Span<const std::string> passthrough,
                                        const IpcChannelNames& channels) {
  if (specs.empty()) {
    return InvalidArgumentError("worker pool needs at least one spec");
  }
  for (const auto& spec : specs) {
    if (spec.rank < 0 || spec.rank >= spec.world || spec.world < 1) {
      return InvalidArgumentError("invalid worker topology: rank ", spec.rank,
                                  " of world ", spec.world);
    }
  }

  // argv: exe worker --rank r --world w --device-ids d
  //       --ipc-commands NAME --ipc-events NAME <passthrough...>
  std::vector<pid_t> pids;
  for (const auto& spec : specs) {
    std::vector<std::string> owned;
    owned.push_back("worker");
    owned.push_back("--rank");
    owned.push_back(std::to_string(spec.rank));
    owned.push_back("--world");
    owned.push_back(std::to_string(spec.world));
    owned.push_back("--device-ids");
    owned.push_back(std::to_string(spec.device_id));
    owned.push_back("--ipc-commands");
    owned.push_back(channels.commands);
    owned.push_back("--ipc-events");
    owned.push_back(channels.events);
    for (const auto& arg : passthrough) owned.push_back(arg);

    std::vector<char*> argv;
    argv.push_back(const_cast<char*>(worker_exe.c_str()));
    for (auto& arg : owned) argv.push_back(arg.data());
    argv.push_back(nullptr);

    // posix_spawn: fresh child, immediate exec -- nothing from this process
    // (CUDA state included) leaks into the worker.
    pid_t pid = 0;
    const int rc =
        posix_spawn(&pid, worker_exe.c_str(), nullptr, nullptr, argv.data(), environ);
    if (rc != 0) {
      // `pids` already holds earlier children; let this pool's teardown kill
      // and reap them.
      WorkerPool partial(std::move(pids));
      return InternalError("posix_spawn of worker rank ", spec.rank,
                           " failed: ", std::strerror(rc));
    }
    pids.push_back(pid);
    INFERX_LOG(INFO) << "launched worker rank=" << spec.rank << " of " << spec.world
                     << " on device " << spec.device_id << " (pid " << pid << ")";
  }
  return WorkerPool(std::move(pids));
}

void WorkerPool::KillAll() {
  for (const pid_t pid : pids_) {
    if (pid > 0) kill(pid, SIGTERM);
  }
}

Status WorkerPool::WaitAll() {
  waited_ = true;
  Status overall = OkStatus();
  for (const pid_t pid : pids_) {
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
      if (errno != EINTR) {
        overall = InternalError("waitpid for worker pid ", pid, ": ",
                                std::strerror(errno));
        break;
      }
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) != 0 && overall.ok()) {
      overall = InternalError("worker pid ", pid, " exited with code ",
                              WEXITSTATUS(status));
    } else if (WIFSIGNALED(status) && overall.ok()) {
      overall = InternalError("worker pid ", pid, " killed by signal ",
                              WTERMSIG(status));
    }
  }
  return overall;
}

}  // namespace inferx::dist
}  // namespace inferx
