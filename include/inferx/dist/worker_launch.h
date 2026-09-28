/// \file
/// \brief Worker process management: one process per device.
///
/// Deliberately plain POSIX fork/exec rather than a library: the child
/// execs immediately (resetting any inherited runtime state), Linux is the
/// only target, and the vendored Boost subset carries no Boost.Process.
/// The controller is the `inferx` binary itself, so workers re-execute
/// `inferx worker` with rank, world, device, and channel names appended.

#ifndef INFERX_DIST_WORKER_LAUNCH_H_
#define INFERX_DIST_WORKER_LAUNCH_H_

#include <sys/types.h>

#include <string>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/status.h"
#include "inferx/dist/worker_ipc.h"

namespace inferx {
namespace dist {

/// \brief One worker's position in the tensor-parallel world.
struct WorkerSpec {
  int rank = 0;
  int world = 1;
  int device_id = 0;  ///< Logical ordinal (after CUDA_VISIBLE_DEVICES).
};

/// \brief The controller-side handle over the launched worker processes.
///
/// Launch never waits for readiness -- read the events channel for the
/// workers' Ready messages. Teardown kills and reaps every child.
class WorkerPool {
 public:
  /// \brief Forks and execs one worker process per spec.
  ///
  /// \param specs         One entry per rank; worker i gets device_ids[i].
  /// \param worker_exe    Executable to run (the controller binary).
  /// \param passthrough   Extra worker arguments, ahead of the rank/device
  ///                      args (model and engine flags).
  /// \param channels      IPC kernel-object names the workers open.
  static StatusOr<WorkerPool> Launch(absl::Span<const WorkerSpec> specs,
                                     const std::string& worker_exe,
                                     absl::Span<const std::string> passthrough,
                                     const IpcChannelNames& channels);

  WorkerPool() = default;
  WorkerPool(WorkerPool&&) = default;
  WorkerPool& operator=(WorkerPool&&) = default;
  ~WorkerPool();

  /// \brief Waits for every worker to exit; returns the first failure.
  Status WaitAll();

 private:
  explicit WorkerPool(std::vector<pid_t> pids) : pids_(std::move(pids)) {}
  void KillAll();

  std::vector<pid_t> pids_;
  bool waited_ = false;
};

/// \brief Resolves this process's own executable path (workers re-exec it).
StatusOr<std::string> SelfExePath();

}  // namespace inferx::dist
}  // namespace inferx

#endif  // INFERX_DIST_WORKER_LAUNCH_H_
