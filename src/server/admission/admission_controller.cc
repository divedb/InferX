#include "inferx/server/admission/admission_controller.h"

#include <algorithm>
#include <cmath>

namespace inferx::server::admission {

RateLimiter::RateLimiter(RateLimitConfig config) : config_(config) {}

RateLimitDecision RateLimiter::Consume(
    const request::TenantId& tenant, uint32_t requested_tokens,
    std::chrono::steady_clock::time_point now) {
  if (tenant.empty() || requested_tokens == 0 ||
      config_.request_capacity <= 0 || config_.token_capacity <= 0 ||
      config_.request_refill_per_second <= 0 ||
      config_.token_refill_per_second <= 0) {
    return {};
  }
  std::lock_guard lock(mutex_);
  auto [it, inserted] = buckets_.try_emplace(
      tenant, Bucket{config_.request_capacity, config_.token_capacity, now});
  Bucket& bucket = it->second;
  const double elapsed = std::max(
      0.0, std::chrono::duration<double>(now - bucket.updated).count());
  bucket.updated = now;
  bucket.requests = std::min(config_.request_capacity,
                             bucket.requests + elapsed * config_.request_refill_per_second);
  bucket.tokens = std::min(config_.token_capacity,
                           bucket.tokens + elapsed * config_.token_refill_per_second);
  const bool request_ok = bucket.requests >= 1.0;
  const bool token_ok = bucket.tokens >= requested_tokens;
  if (request_ok && token_ok) {
    bucket.requests -= 1.0;
    bucket.tokens -= requested_tokens;
    return {true, std::chrono::milliseconds(0)};
  }
  const double request_wait = request_ok
                                  ? 0.0
                                  : (1.0 - bucket.requests) /
                                        config_.request_refill_per_second;
  const double token_wait = token_ok
                                ? 0.0
                                : (requested_tokens - bucket.tokens) /
                                      config_.token_refill_per_second;
  const double wait = std::max(request_wait, token_wait);
  return {false, std::chrono::milliseconds(
                    static_cast<int64_t>(std::ceil(wait * 1000.0)))};
}

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
