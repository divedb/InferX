// Offline-throughput benchmark entry point (`inferx bench throughput`).
// The execution is not implemented yet; the CLI contract is in place.
#include "inferx/bench/throughput.h"

namespace inferx::bench {

Status RunThroughput(const ThroughputParams& params) {
  return UnimplementedError("bench throughput is not implemented yet");
}

}  // namespace inferx::bench
