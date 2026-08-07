#include "inferx/server/handlers/metrics_handler.h"

namespace inferx::server::handlers {

folly::coro::Task<void> MetricsHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  (void)context;
  transport::HttpResponse result{
      source_ == nullptr ? boost::beast::http::status::service_unavailable
                         : boost::beast::http::status::ok,
      request.version()};
  result.keep_alive(request.keep_alive());
  if (source_ == nullptr) {
    result.set(boost::beast::http::field::content_type, "application/json");
    result.body() =
        "{\"error\":{\"message\":\"metrics source is unavailable\","
        "\"type\":\"server_error\",\"code\":\"service_unavailable\"}}";
  } else if (presentation_ == MetricsPresentation::kPrometheus) {
    result.set(boost::beast::http::field::content_type,
               "text/plain; version=0.0.4; charset=utf-8");
    result.body() = observability::RenderPrometheus(source_->Snapshot());
  } else {
    result.set(boost::beast::http::field::content_type, "application/json");
    result.body() =
        observability::RenderLegacyStatsJson(source_->Snapshot());
  }
  result.prepare_payload();
  (void)co_await response.WriteResponse(std::move(result), cancellation);
}

}  // namespace inferx::server::handlers
