/// \file
/// \brief The worker-side engine loop: one process owns one rank-local
///        engine and serves generation requests from the IPC channel.

#ifndef INFERX_DIST_WORKER_MAIN_H_
#define INFERX_DIST_WORKER_MAIN_H_

#include "inferx/config/cache_config.h"
#include "inferx/config/execution_config.h"
#include "inferx/config/model_config.h"
#include "inferx/config/parallel_config.h"
#include "inferx/config/scheduler_config.h"
#include "inferx/core/status.h"
#include "inferx/dist/worker_ipc.h"

namespace inferx {
namespace dist {

/// \brief Everything a worker needs to build its engine.
struct WorkerOptions {
  ModelConfig model;
  CacheConfig cache;
  SchedulerConfig scheduler;
  ExecutionConfig execution;
  ParallelConfig parallel;  ///< This worker's rank within the world.
};

/// \brief Runs the worker until a shutdown message: builds the rank-local
///        engine (scheduler + ModelRunner on this rank's device), posts
///        Ready, then alternates between draining generation requests and
///        stepping the scheduler, streaming generated token ids and
///        terminal events back over the events channel.
///
/// The loop mirrors EngineGateway's step contract (samples are real output
/// only once the whole prompt has KV; PopFinished delivers terminals), with
/// raw token events in place of detokenized deltas.
Status RunWorker(const WorkerOptions& options, const IpcChannelNames& channels);

}  // namespace inferx::dist
}  // namespace inferx

#endif  // INFERX_DIST_WORKER_MAIN_H_
