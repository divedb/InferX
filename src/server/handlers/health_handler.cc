#include "inferx/server/handlers/health_handler.h"

#include <string_view>

namespace inferx::server::handlers {

bool HealthState::live() const {
  return event_loop_healthy_.load() && critical_threads_healthy_.load();
}

bool HealthState::startup() const {
  return configuration_loaded_.load() && dependencies_loaded_.load();
}

bool HealthState::ready() const {
  return live() && startup() && listener_accepting_.load() &&
         scheduler_connected_.load() && ready_model_available_.load();
}

folly::coro::Task<void> HealthHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  (void)context;
  bool healthy = false;
  std::string_view probe;
  switch (probe_) {
    case HealthProbe::kLive:
      healthy = state_->live();
      probe = "live";
      break;
    case HealthProbe::kReady:
      healthy = state_->ready();
      probe = "ready";
      break;
    case HealthProbe::kStartup:
      healthy = state_->startup();
      probe = "startup";
      break;
  }

  transport::HttpResponse result{
      healthy ? boost::beast::http::status::ok
              : boost::beast::http::status::service_unavailable,
      request.version()};
  result.keep_alive(request.keep_alive());
  result.set(boost::beast::http::field::content_type, "application/json");
  result.body() = std::string("{\"status\":\"") +
                  (healthy ? "ok" : "unavailable") + "\",\"probe\":\"" +
                  std::string(probe) + "\"}";
  result.prepare_payload();
  (void)co_await response.WriteResponse(std::move(result), cancellation);
}

}  // namespace inferx::server::handlers
