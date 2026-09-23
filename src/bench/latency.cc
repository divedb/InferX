// Batch-latency benchmark entry point (`inferx bench latency`).
// The execution is not implemented yet; the CLI contract is in place.
#include "inferx/bench/latency.h"

namespace inferx::bench {

Status RunLatency(const LatencyParams& params) {
  return UnimplementedError("bench latency is not implemented yet");
}

}  // namespace inferx::bench
