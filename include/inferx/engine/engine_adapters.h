#pragma once

#include "inferx/engine/engine.h"
#include "inferx/server/observability/metrics_snapshot.h"
#include "inferx/server/scheduler_client/in_process_scheduler_client.h"

namespace inferx::engine {

namespace scheduler_client = server::scheduler_client;
namespace observability = server::observability;

class EngineSchedulerBackend final
    : public scheduler_client::InProcessEngineBackend {
 public:
  explicit EngineSchedulerBackend(Engine* engine) : engine_(engine) {}

  StatusOr<std::shared_ptr<scheduler_client::InProcessGeneration>> Submit(
      const scheduler_client::ScheduledRequest& request) override;

 private:
  Engine* engine_;
};

class EngineMetricsSource final : public observability::MetricsSource {
 public:
  explicit EngineMetricsSource(const Engine* engine) : engine_(engine) {}
  observability::MetricsSnapshot Snapshot() const override;

 private:
  const Engine* engine_;
};

}  // namespace inferx::engine
