#pragma once

#include <folly/CancellationToken.h>

#include <chrono>
#include <cstdint>
#include <string>

namespace inferx::server::request {

using RequestId = std::string;
using TenantId = std::string;
using PrincipalId = std::string;
using ModelId = std::string;
using ModelVersion = std::string;

enum class CancellationReason {
  kClientDisconnected,
  kDeadline,
  kAdministrative,
  kTenantShutdown,
  kModelUnavailable,
  kInternal,
};

enum class RequestState {
  kReceived,
  kValidating,
  kAuthenticated,
  kAdmissionPending,
  kQueued,
  kPrefilling,
  kDecoding,
  kFinishing,
  kCompleted,
  kRejected,
  kTimedOut,
  kCancelled,
  kFailed,
};

const char* RequestStateName(RequestState state);
bool IsTerminal(RequestState state);
bool CanTransition(RequestState from, RequestState to);

struct RequestContext {
  RequestId request_id;
  std::string client_request_id;
  TenantId tenant_id;
  PrincipalId principal_id;
  ModelId requested_model;
  ModelVersion resolved_version;

  std::chrono::steady_clock::time_point received_at;
  std::chrono::steady_clock::time_point deadline;
  std::chrono::steady_clock::time_point admitted_at;
  std::chrono::steady_clock::time_point first_token_at;
  std::chrono::steady_clock::time_point completed_at;

  RequestState state = RequestState::kReceived;
  folly::CancellationSource cancellation;

  bool TransitionTo(RequestState next);
  void Cancel(CancellationReason reason);
  CancellationReason cancellation_reason() const { return reason_; }

 private:
  CancellationReason reason_ = CancellationReason::kInternal;
};

}  // namespace inferx::server::request
