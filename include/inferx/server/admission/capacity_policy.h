#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>

#include "inferx/server/admission/admission_controller.h"

namespace inferx::server::admission {

/// A point-in-time, externally supplied view of capacity. The policy does not
/// own scheduler or model state, which keeps admission evaluation host-only
/// and deterministic.
struct CapacitySnapshot {
  bool scheduler_healthy = false;
  bool model_ready = false;
  uint32_t queued_requests = 0;
  size_t event_buffer_bytes = 0;
};

struct CapacityPolicyConfig {
  uint32_t max_queued_requests = 4096;
  size_t max_event_buffer_bytes = 256u << 20;
  std::chrono::milliseconds overload_retry_after{1000};
  std::chrono::milliseconds unavailable_retry_after{1000};
};

class CapacityPolicy {
 public:
  explicit CapacityPolicy(CapacityPolicyConfig config);

  AdmissionDecision Evaluate(const CapacitySnapshot& snapshot) const;

 private:
  CapacityPolicyConfig config_;
};

}  // namespace inferx::server::admission
