#include "inferx/server/admission/admission_controller.h"

#include <algorithm>
#include <cmath>

namespace inferx::server::admission {
namespace {

void Refill(double request_capacity, double request_refill,
            double token_capacity, double token_refill,
            std::chrono::steady_clock::time_point now, double* requests,
            double* tokens, std::chrono::steady_clock::time_point* updated) {
  const double elapsed = std::max(
      0.0, std::chrono::duration<double>(now - *updated).count());
  *updated = now;
  *requests = std::min(request_capacity, *requests + elapsed * request_refill);
  *tokens = std::min(token_capacity, *tokens + elapsed * token_refill);
}

std::chrono::milliseconds RetryAfter(double requests, double tokens,
                                     uint32_t requested_tokens,
                                     double request_refill,
                                     double token_refill) {
  const double request_wait =
      requests >= 1.0 ? 0.0 : (1.0 - requests) / request_refill;
  const double token_wait = tokens >= requested_tokens
                                ? 0.0
                                : (requested_tokens - tokens) / token_refill;
  return std::chrono::milliseconds(
      static_cast<int64_t>(std::ceil(std::max(request_wait, token_wait) * 1000.0)));
}

}  // namespace

RateLimiter::RateLimiter(RateLimitConfig config) : config_(config) {}

RateLimitDecision RateLimiter::Consume(
    const request::TenantId& tenant, uint32_t requested_tokens,
    std::chrono::steady_clock::time_point now) {
  return Consume(tenant, {}, requested_tokens, now);
}

RateLimitDecision RateLimiter::Consume(
    const request::TenantId& tenant, std::string_view api_key_id,
    uint32_t requested_tokens, std::chrono::steady_clock::time_point now) {
  if (tenant.empty() || requested_tokens == 0 ||
      config_.request_capacity <= 0 || config_.token_capacity <= 0 ||
      config_.request_refill_per_second <= 0 ||
      config_.token_refill_per_second <= 0 ||
      (!api_key_id.empty() &&
       (config_.api_key_request_capacity <= 0 ||
        config_.api_key_token_capacity <= 0 ||
        config_.api_key_request_refill_per_second <= 0 ||
        config_.api_key_token_refill_per_second <= 0))) {
    return {};
  }
  std::lock_guard lock(mutex_);
  auto it = tenant_buckets_
                .try_emplace(
                    tenant, Bucket{config_.request_capacity,
                                   config_.token_capacity, now})
                .first;
  Bucket& tenant_bucket = it->second;
  Refill(config_.request_capacity, config_.request_refill_per_second,
         config_.token_capacity, config_.token_refill_per_second, now,
         &tenant_bucket.requests, &tenant_bucket.tokens, &tenant_bucket.updated);

  Bucket* key_bucket = nullptr;
  if (!api_key_id.empty()) {
    auto key_it = api_key_buckets_[tenant]
                      .try_emplace(
                          std::string(api_key_id),
                          Bucket{config_.api_key_request_capacity,
                                 config_.api_key_token_capacity, now})
                      .first;
    key_bucket = &key_it->second;
    Refill(config_.api_key_request_capacity,
           config_.api_key_request_refill_per_second,
           config_.api_key_token_capacity,
           config_.api_key_token_refill_per_second, now, &key_bucket->requests,
           &key_bucket->tokens, &key_bucket->updated);
  }

  if (tenant_bucket.requests < 1.0 || tenant_bucket.tokens < requested_tokens) {
    return {false,
            RetryAfter(tenant_bucket.requests, tenant_bucket.tokens,
                       requested_tokens, config_.request_refill_per_second,
                       config_.token_refill_per_second),
            RateLimitDecision::Reason::kTenantLimit};
  }
  if (key_bucket != nullptr &&
      (key_bucket->requests < 1.0 || key_bucket->tokens < requested_tokens)) {
    return {false,
            RetryAfter(key_bucket->requests, key_bucket->tokens,
                       requested_tokens,
                       config_.api_key_request_refill_per_second,
                       config_.api_key_token_refill_per_second),
            RateLimitDecision::Reason::kApiKeyLimit};
  }

  tenant_bucket.requests -= 1.0;
  tenant_bucket.tokens -= requested_tokens;
  if (key_bucket != nullptr) {
    key_bucket->requests -= 1.0;
    key_bucket->tokens -= requested_tokens;
  }
  return {true, std::chrono::milliseconds(0),
          RateLimitDecision::Reason::kAllowed};
}

Status RateLimiter::Reconcile(
    const request::TenantId& tenant, uint32_t reserved_tokens,
    uint32_t actual_tokens, std::chrono::steady_clock::time_point now) {
  return Reconcile(tenant, {}, reserved_tokens, actual_tokens, now);
}

Status RateLimiter::Reconcile(
    const request::TenantId& tenant, std::string_view api_key_id,
    uint32_t reserved_tokens, uint32_t actual_tokens,
    std::chrono::steady_clock::time_point now) {
  if (tenant.empty() || reserved_tokens == 0) {
    return InvalidArgumentError(
        "tenant ID and reserved tokens are required for reconciliation");
  }

  std::lock_guard lock(mutex_);
  const auto it = tenant_buckets_.find(tenant);
  if (it == tenant_buckets_.end()) {
    return FailedPreconditionError(
        "tenant has no token reservation to reconcile");
  }
  Bucket* key_bucket = nullptr;
  if (!api_key_id.empty()) {
    const auto tenant_it = api_key_buckets_.find(tenant);
    if (tenant_it == api_key_buckets_.end()) {
      return FailedPreconditionError(
          "API key has no token reservation to reconcile");
    }
    const auto key_it = tenant_it->second.find(std::string(api_key_id));
    if (key_it == tenant_it->second.end()) {
      return FailedPreconditionError(
          "API key has no token reservation to reconcile");
    }
    key_bucket = &key_it->second;
  }

  Bucket& bucket = it->second;
  Refill(config_.request_capacity, config_.request_refill_per_second,
         config_.token_capacity, config_.token_refill_per_second, now,
         &bucket.requests, &bucket.tokens, &bucket.updated);

  const double adjustment = static_cast<double>(reserved_tokens) -
                            static_cast<double>(actual_tokens);
  bucket.tokens = std::min(config_.token_capacity, bucket.tokens + adjustment);
  if (key_bucket != nullptr) {
    Refill(config_.api_key_request_capacity,
           config_.api_key_request_refill_per_second,
           config_.api_key_token_capacity,
           config_.api_key_token_refill_per_second, now, &key_bucket->requests,
           &key_bucket->tokens, &key_bucket->updated);
    key_bucket->tokens = std::min(config_.api_key_token_capacity,
                                  key_bucket->tokens + adjustment);
  }
  return OkStatus();
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
  uint64_t reservation_id = 0;
  do {
    reservation_id = next_reservation_id_++;
  } while (reservation_id == 0 || reservations_.contains(reservation_id));
  reservations_.emplace(reservation_id, request);
  AdmissionDecision decision{AdmissionReason::kAdmitted, "admitted"};
  decision.reservation_id = reservation_id;
  return decision;
}

void AdmissionController::ReleaseLocked(const AdmissionRequest& request) {
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

void AdmissionController::Release(uint64_t reservation_id) {
  if (reservation_id == 0) return;
  std::lock_guard lock(mutex_);
  const auto reservation = reservations_.find(reservation_id);
  if (reservation == reservations_.end()) return;
  ReleaseLocked(reservation->second);
  reservations_.erase(reservation);
}

void AdmissionController::Release(const AdmissionRequest& request) {
  std::lock_guard lock(mutex_);
  const auto reservation = std::find_if(
      reservations_.begin(), reservations_.end(), [&](const auto& entry) {
        return entry.second.tenant_id == request.tenant_id &&
               entry.second.reserved_tokens == request.reserved_tokens;
      });
  if (reservation == reservations_.end()) return;
  ReleaseLocked(reservation->second);
  reservations_.erase(reservation);
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
