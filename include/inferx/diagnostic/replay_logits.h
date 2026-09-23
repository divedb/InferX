#ifndef INFERX_DIAGNOSTIC_REPLAY_LOGITS_H_
#define INFERX_DIAGNOSTIC_REPLAY_LOGITS_H_

#include <string>

#include "inferx/core/status.h"

namespace inferx::diagnostic {

/// \brief Inputs for the teacher-forced numerical diagnostic
///        (`inferx diagnostic replay-logits`). Never use this path for
///        throughput timing.
struct ReplayLogitsParams {
  std::string model_dir;
  std::string fixture_path;              ///< Replay fixture JSON.
  std::string output;                    ///< FP32 logits dump path.
  std::string page_order = "reverse";    ///< identity | reverse | shuffle.
  std::string cache_dir;                 ///< Optional KV cache dump directory.
  int chunk_size = 4096;                 ///< Prefill tokens per forward pass.
};

/// \brief Replays identical prefixes and exports final full-vocabulary FP32
///        logits (plus optional per-layer KV cache dumps).
Status RunReplayLogits(const ReplayLogitsParams& params);

}  // namespace inferx::diagnostic

#endif  // INFERX_DIAGNOSTIC_REPLAY_LOGITS_H_
