#pragma once

#include <cstdint>
#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

#include "inferx/core/status.h"
#include "inferx/server/request/request_context.h"

namespace inferx::server::admission {

enum class AdmissionReason {
  kAdmitted,
  kGlobalConcurrency,
  kTenantConcurrency,
  kTokenCapacity,
  kInvalidRequest,
};

struct AdmissionDecision {
  AdmissionReason reason = AdmissionReason::kInvalidRequest;
  std::string message;
  bool admitted() const { return reason == AdmissionReason::kAdmitted; }
};

struct AdmissionRequest {
  request::TenantId tenant_id;
  uint32_t reserved_tokens = 0;
};

struct AdmissionConfig {
  uint32_t max_active_requests = 1024;
  uint64_t max_reserved_tokens = 1u << 20;
  uint32_t max_active_per_tenant = 128;
};

struct RateLimitDecision {
  bool allowed = false;
  std::chrono::milliseconds retry_after{0};
};

struct RateLimitConfig {
  double request_capacity = 100.0;
  double request_refill_per_second = 100.0;
  double token_capacity = 100000.0;
  double token_refill_per_second = 100000.0;
};

class RateLimiter {
 public:
  explicit RateLimiter(RateLimitConfig config);
  RateLimitDecision Consume(const request::TenantId& tenant,
                            uint32_t requested_tokens,
                            std::chrono::steady_clock::time_point now =
                                std::chrono::steady_clock::now());
  /// Reconciles a prior maximum-token reservation with actual usage. Unused
  /// tokens are returned to the tenant bucket; usage above the reservation is
  /// charged as debt so a caller cannot bypass the token rate by under-
  /// reserving. The request-rate token is deliberately not refunded.
  Status Reconcile(const request::TenantId& tenant, uint32_t reserved_tokens,
                   uint32_t actual_tokens,
                   std::chrono::steady_clock::time_point now =
                       std::chrono::steady_clock::now());

 private:
  struct Bucket {
    double requests;
    double tokens;
    std::chrono::steady_clock::time_point updated;
  };
  RateLimitConfig config_;
  std::mutex mutex_;
  std::unordered_map<request::TenantId, Bucket> buckets_;
};

class AdmissionController {
 public:
  explicit AdmissionController(AdmissionConfig config);

  AdmissionDecision TryAdmit(const AdmissionRequest& request);
  void Release(const AdmissionRequest& request);
  uint32_t active_requests() const;
  uint64_t reserved_tokens() const;

 private:
  struct TenantUsage {
    uint32_t active = 0;
    uint64_t tokens = 0;
  };

  AdmissionConfig config_;
  mutable std::mutex mutex_;
  uint32_t active_ = 0;
  uint64_t tokens_ = 0;
  std::unordered_map<request::TenantId, TenantUsage> tenants_;
};

}  // namespace inferx::server::admission
