#include "inferx/server/admission/admission_controller.h"

namespace inferx::server::admission {

AdmissionController::AdmissionController(AdmissionConfig config)
    : config_(config) {}

AdmissionDecision AdmissionController::TryAdmit(
    const AdmissionRequest& request) {
  if (request.tenant_id.empty() || request.reserved_tokens == 0) {
    return {AdmissionReason::kInvalidRequest,
            "tenant ID and reserved tokens are required"};
  }
  std::lock_guard lock(mutex_);
  TenantUsage& tenant = tenants_[request.tenant_id];
  if (active_ >= config_.max_active_requests) {
    return {AdmissionReason::kGlobalConcurrency,
            "global request capacity is exhausted"};
  }
  if (tenant.active >= config_.max_active_per_tenant) {
    return {AdmissionReason::kTenantConcurrency,
            "tenant request capacity is exhausted"};
  }
  if (request.reserved_tokens > config_.max_reserved_tokens -
                                   (tokens_ > config_.max_reserved_tokens
                                        ? config_.max_reserved_tokens
                                        : tokens_)) {
    return {AdmissionReason::kTokenCapacity,
            "reserved token capacity is exhausted"};
  }
  ++active_;
  tokens_ += request.reserved_tokens;
  ++tenant.active;
  tenant.tokens += request.reserved_tokens;
  return {AdmissionReason::kAdmitted, "admitted"};
}

void AdmissionController::Release(const AdmissionRequest& request) {
  std::lock_guard lock(mutex_);
  const auto it = tenants_.find(request.tenant_id);
  if (it == tenants_.end() || it->second.active == 0) return;
  --it->second.active;
  it->second.tokens = it->second.tokens >= request.reserved_tokens
                          ? it->second.tokens - request.reserved_tokens
                          : 0;
  active_ = active_ == 0 ? 0 : active_ - 1;
  tokens_ = tokens_ >= request.reserved_tokens
                ? tokens_ - request.reserved_tokens
                : 0;
  if (it->second.active == 0) tenants_.erase(it);
}

uint32_t AdmissionController::active_requests() const {
  std::lock_guard lock(mutex_);
  return active_;
}

uint64_t AdmissionController::reserved_tokens() const {
  std::lock_guard lock(mutex_);
  return tokens_;
}

}  // namespace inferx::server::admission
