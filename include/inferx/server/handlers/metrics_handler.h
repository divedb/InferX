#pragma once

#include "inferx/server/observability/metrics_snapshot.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::handlers {

enum class MetricsPresentation { kPrometheus, kLegacyJson };

class MetricsHandler final : public transport::RequestHandler {
 public:
  MetricsHandler(const observability::MetricsSource* source,
                 MetricsPresentation presentation)
      : source_(source), presentation_(presentation) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const observability::MetricsSource* source_;
  MetricsPresentation presentation_;
};

}  // namespace inferx::server::handlers
