#include "inferx/server/admission/capacity_policy.h"

namespace inferx::server::admission {

CapacityPolicy::CapacityPolicy(CapacityPolicyConfig config) : config_(config) {
  if (config_.overload_retry_after.count() < 0) {
    config_.overload_retry_after = std::chrono::milliseconds(0);
  }
  if (config_.unavailable_retry_after.count() < 0) {
    config_.unavailable_retry_after = std::chrono::milliseconds(0);
  }
}

AdmissionDecision CapacityPolicy::Evaluate(
    const CapacitySnapshot& snapshot) const {
  if (!snapshot.scheduler_healthy) {
    return {AdmissionReason::kSchedulerUnavailable,
            "scheduler is unavailable", config_.unavailable_retry_after};
  }
  if (!snapshot.model_ready) {
    return {AdmissionReason::kModelUnavailable, "model is unavailable",
            config_.unavailable_retry_after};
  }
  if (snapshot.queued_requests >= config_.max_queued_requests) {
    return {AdmissionReason::kQueueCapacity,
            "model queue capacity is exhausted",
            config_.overload_retry_after};
  }
  if (snapshot.event_buffer_bytes >= config_.max_event_buffer_bytes) {
    return {AdmissionReason::kEventBufferCapacity,
            "streaming buffer capacity is exhausted",
            config_.overload_retry_after};
  }
  return {AdmissionReason::kAdmitted, "capacity available"};
}

}  // namespace inferx::server::admission
