#pragma once

#include <folly/CancellationToken.h>
#include <folly/coro/AsyncGenerator.h>
#include <folly/coro/Task.h>

#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/server/request/request_context.h"

namespace inferx::server::scheduler_client {

using RequestId = request::RequestId;

enum class WorkloadClass { kGeneration, kEmbedding };
enum class PriorityClass { kInteractive, kBatch, kBackground };

struct SamplingParams {
  int32_t max_tokens = 0;
  float temperature = 0;
  float top_p = 1;
  uint64_t seed = 0;
  std::vector<std::string> stop;
};

struct ScheduledRequest {
  request::RequestId request_id;
  request::TenantId tenant_id;
  request::ModelVersion model_version;
  WorkloadClass workload = WorkloadClass::kGeneration;
  PriorityClass priority = PriorityClass::kInteractive;
  std::vector<int32_t> prompt_tokens;
  SamplingParams sampling;
  std::chrono::steady_clock::time_point deadline;
};

struct SubmitResult {
  request::RequestId request_id;
  uint32_t attempt = 0;
};

struct GenerationEvent {
  request::RequestId request_id;
  uint64_t sequence_number = 0;
  std::string text_delta;
  uint32_t generated_tokens = 0;
  bool terminal = false;
  std::string finish_reason;
  Status error = OkStatus();
};

class SchedulerClient {
 public:
  virtual ~SchedulerClient() = default;
  virtual folly::coro::Task<StatusOr<SubmitResult>> Submit(
      ScheduledRequest request, folly::CancellationToken cancellation) = 0;
  virtual folly::coro::AsyncGenerator<GenerationEvent&&> Events(
      request::RequestId request_id,
      folly::CancellationToken cancellation) = 0;
  virtual folly::coro::Task<Status> Cancel(
      request::RequestId request_id, request::CancellationReason reason) = 0;
};

}  // namespace inferx::server::scheduler_client
