#include "inferx/server/gateway/gateway_server.h"

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "inferx/server/admission/admission_controller.h"
#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/authenticator.h"
#include "inferx/server/handlers/api_routes.h"
#include "inferx/server/handlers/health_handler.h"
#include "inferx/server/middleware/authentication.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/observability/metrics_snapshot.h"
#include "inferx/server/request/managed_request_service.h"
#include "inferx/server/request/request_manager.h"
#include "inferx/server/scheduler_client/grpc_scheduler_transport.h"
#include "inferx/server/scheduler_client/remote_scheduler_client.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/beast_listener.h"
#include "inferx/server/transport/io_runtime.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx::server::gateway {
namespace {

size_t DefaultThreads(size_t requested, size_t maximum) {
  if (requested != 0) return requested;
  return std::max<size_t>(
      1, std::min<size_t>(maximum,
                          std::max(1u, std::thread::hardware_concurrency())));
}

bool IsSha256Hex(std::string_view value) {
  return value.size() == 64 &&
         value.find_first_not_of("0123456789abcdefABCDEF") ==
             std::string_view::npos;
}

auth::Principal ApiPrincipal(size_t index) {
  auth::Principal principal;
  principal.tenant_id = "default";
  principal.subject = "api-key";
  principal.key_id = "configured-" + std::to_string(index);
  principal.scopes = {"inference.invoke", "models.read"};
  return principal;
}

class GatewayMetrics final : public observability::MetricsSource {
 public:
  observability::MetricsSnapshot Snapshot() const override { return {}; }
};

}  // namespace

Status ValidateGatewayServerConfig(const GatewayServerConfig& config) {
  if (config.port < 0 || config.port > 65535) {
    return InvalidArgumentError("port must be in [0, 65535]");
  }
  if (config.scheduler_endpoint.empty()) {
    return InvalidArgumentError("scheduler_endpoint is required");
  }
  if (config.tokenizer_directory.empty()) {
    return InvalidArgumentError("tokenizer_directory is required");
  }
  if (config.model_id.empty() || config.model_version.empty()) {
    return InvalidArgumentError("model_id and model_version are required");
  }
  if (config.max_request_bytes == 0 || config.max_active_requests == 0 ||
      config.max_active_requests > UINT32_MAX) {
    return InvalidArgumentError("gateway limits must be positive and bounded");
  }
  if (config.read_timeout_seconds <= 0 || config.write_timeout_seconds <= 0 ||
      config.request_timeout_seconds <= 0 ||
      config.scheduler_connect_timeout_seconds <= 0) {
    return InvalidArgumentError("gateway timeouts must be positive");
  }
  for (const auto& hash : config.api_key_sha256) {
    if (!IsSha256Hex(hash)) {
      return InvalidArgumentError(
          "api_key_sha256 entries must contain 64 hexadecimal characters");
    }
  }
  return OkStatus();
}

struct GatewayServer::Impl {
  GatewayServerConfig config;
  std::unique_ptr<tokenizer::Tokenizer> tokenizer;
  std::unique_ptr<folly::CPUThreadPoolExecutor> blocking_executor;
  std::unique_ptr<transport::IoRuntime> runtime;
  std::shared_ptr<grpc::Channel> channel;
  std::shared_ptr<scheduler_client::GrpcSchedulerTransport> transport_client;
  std::unique_ptr<scheduler_client::RemoteSchedulerClient> scheduler;
  request::RequestManager request_manager;
  admission::AdmissionController admission;
  std::unique_ptr<request::ManagedRequestService> requests;
  std::unique_ptr<tokenization::InProcessTokenizationService> tokenization;
  model_registry::Registry models;
  handlers::HealthState health;
  GatewayMetrics metrics;
  std::shared_ptr<auth::ApiKeyStore> key_store;
  std::shared_ptr<auth::ApiKeyAuthenticator> authenticator;
  std::shared_ptr<middleware::BearerRouteGuard> route_guard;
  std::shared_ptr<transport::Routes> routes;
  std::unique_ptr<transport::BeastListener> listener;
  mutable std::mutex state_mutex;
  std::condition_variable state_cv;
  bool listening = false;
  bool finished = false;
  bool stopping = false;

  Impl(GatewayServerConfig config_value,
       std::unique_ptr<tokenizer::Tokenizer> tokenizer_value,
       std::unique_ptr<folly::CPUThreadPoolExecutor> blocking_value,
       std::unique_ptr<transport::IoRuntime> runtime_value)
      : config(std::move(config_value)),
        tokenizer(std::move(tokenizer_value)),
        blocking_executor(std::move(blocking_value)),
        runtime(std::move(runtime_value)),
        channel(grpc::CreateChannel(config.scheduler_endpoint,
                                    grpc::InsecureChannelCredentials())),
        transport_client(
            std::make_shared<scheduler_client::GrpcSchedulerTransport>(
                channel)),
        scheduler(std::make_unique<scheduler_client::RemoteSchedulerClient>(
            transport_client, blocking_executor.get())),
        admission({.max_active_requests =
                       static_cast<uint32_t>(config.max_active_requests),
                   .max_reserved_tokens = std::numeric_limits<uint64_t>::max(),
                   .max_active_per_tenant =
                       static_cast<uint32_t>(config.max_active_requests)}),
        requests(std::make_unique<request::ManagedRequestService>(
            &request_manager, scheduler.get(), &admission)),
        tokenization(
            std::make_unique<tokenization::InProcessTokenizationService>(
                tokenizer.get(), config.model_id + "@" + config.model_version)),
        key_store(std::make_shared<auth::ApiKeyStore>()) {}

  ~Impl() {
    Stop();
    runtime->Join();
    scheduler.reset();
    blocking_executor->stop();
    blocking_executor->join();
  }

  Status Initialize() {
    if (!channel->WaitForConnected(
            std::chrono::system_clock::now() +
            std::chrono::seconds(config.scheduler_connect_timeout_seconds))) {
      return absl::UnavailableError(
          "scheduler did not become ready before timeout");
    }
    std::vector<auth::ApiKeyRecord> records;
    for (size_t i = 0; i < config.api_key_sha256.size(); ++i) {
      records.push_back(
          {.hash_hex = config.api_key_sha256[i], .principal = ApiPrincipal(i)});
    }
    INFERX_RETURN_IF_ERROR(key_store->ReplaceAll(std::move(records)));
    authenticator = std::make_shared<auth::ApiKeyAuthenticator>(
        key_store.get(), config.api_key_sha256.empty());
    route_guard = std::make_shared<middleware::BearerRouteGuard>(authenticator);
    const int64_t created =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    INFERX_RETURN_IF_ERROR(
        models.Register({.id = config.model_id,
                         .version = config.model_version,
                         .created = created,
                         .state = model_registry::ModelState::kReady,
                         .supports_generation = true,
                         .supports_embeddings = false,
                         .tokenizer_revision = config.tokenizer_revision}));
    health.SetConfigurationLoaded(true);
    health.SetDependenciesLoaded(true);
    health.SetSchedulerConnected(true);
    health.SetReadyModelAvailable(true);
    INFERX_ASSIGN_OR_RETURN(
        routes, handlers::BuildApiRoutes(
                    {.health = &health,
                     .models = &models,
                     .tokenization = tokenization.get(),
                     .requests = requests.get(),
                     .metrics = &metrics,
                     .guard = route_guard,
                     .max_inference_body_bytes = config.max_request_bytes,
                     .request_timeout = std::chrono::seconds(
                         config.request_timeout_seconds)}));
    INFERX_ASSIGN_OR_RETURN(
        listener,
        transport::BeastListener::Create(
            runtime.get(),
            {.host = config.host,
             .port = config.port,
             .max_request_bytes = config.max_request_bytes,
             .header_read_timeout_seconds = config.read_timeout_seconds,
             .body_read_timeout_seconds = config.read_timeout_seconds,
             .write_timeout_seconds = config.write_timeout_seconds,
             .keep_alive_timeout_seconds = config.read_timeout_seconds},
            routes));
    return OkStatus();
  }

  Status Listen() {
    {
      std::lock_guard lock(state_mutex);
      if (stopping) return FailedPreconditionError("gateway is stopping");
      if (listening || finished) {
        return FailedPreconditionError("gateway may only listen once");
      }
    }
    Status status = runtime->Start();
    if (status.ok()) status = listener->Start();
    {
      std::lock_guard lock(state_mutex);
      listening = status.ok();
      finished = !status.ok();
    }
    if (status.ok()) health.SetListenerAccepting(true);
    state_cv.notify_all();
    if (!status.ok()) {
      runtime->Stop();
      runtime->Join();
      return status;
    }
    runtime->Join();
    {
      std::lock_guard lock(state_mutex);
      listening = false;
      finished = true;
    }
    state_cv.notify_all();
    return OkStatus();
  }

  void Stop() {
    {
      std::lock_guard lock(state_mutex);
      if (stopping) return;
      stopping = true;
    }
    health.SetListenerAccepting(false);
    health.SetSchedulerConnected(false);
    transport_client->Shutdown();
    if (listener) listener->Stop();
    runtime->Stop();
    state_cv.notify_all();
  }
};

GatewayServer::GatewayServer(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
GatewayServer::~GatewayServer() = default;

StatusOr<std::unique_ptr<GatewayServer>> GatewayServer::Create(
    const GatewayServerConfig& config) {
  INFERX_RETURN_IF_ERROR(ValidateGatewayServerConfig(config));
  INFERX_ASSIGN_OR_RETURN(auto tokenizer, tokenizer::Tokenizer::LoadFrom(
                                              config.tokenizer_directory));
  const size_t application_threads =
      DefaultThreads(config.application_threads, 32);
  auto blocking =
      std::make_unique<folly::CPUThreadPoolExecutor>(application_threads);
  INFERX_ASSIGN_OR_RETURN(
      auto runtime, transport::IoRuntime::Create(
                        {.io_shards = DefaultThreads(config.io_threads, 8),
                         .threads_per_shard = 1,
                         .coroutine_threads = application_threads}));
  auto impl = std::make_unique<Impl>(config, std::move(tokenizer),
                                     std::move(blocking), std::move(runtime));
  INFERX_RETURN_IF_ERROR(impl->Initialize());
  return std::unique_ptr<GatewayServer>(new GatewayServer(std::move(impl)));
}

Status GatewayServer::Listen() { return impl_->Listen(); }
void GatewayServer::Stop() { impl_->Stop(); }

bool GatewayServer::WaitUntilReady() {
  std::unique_lock lock(impl_->state_mutex);
  impl_->state_cv.wait(lock, [this] {
    return impl_->listening || impl_->finished || impl_->stopping;
  });
  return impl_->listening;
}

int GatewayServer::port() const {
  return impl_->listener == nullptr ? impl_->config.port
                                    : impl_->listener->port();
}

}  // namespace inferx::server::gateway
