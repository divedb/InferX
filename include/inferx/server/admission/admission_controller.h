#pragma once

#include <cstdint>
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
