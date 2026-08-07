#include "inferx/server/http_server.h"

#include <folly/executors/CPUThreadPoolExecutor.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "engine_adapters.h"
#include "inferx/server/admission/admission_controller.h"
#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/authenticator.h"
#include "inferx/server/handlers/api_routes.h"
#include "inferx/server/handlers/health_handler.h"
#include "inferx/server/middleware/authentication.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/request/managed_request_service.h"
#include "inferx/server/request/request_manager.h"
#include "inferx/server/scheduler_client/in_process_scheduler_client.h"
#include "inferx/server/scheduler_client/remote_scheduler_client.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/beast_listener.h"
#include "inferx/server/transport/io_runtime.h"

#if defined(INFERX_WITH_GRPC_SCHEDULER)
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include "inferx/server/scheduler_client/grpc_scheduler_transport.h"
#endif

namespace inferx::server {
namespace {

size_t DefaultThreads(size_t requested, size_t maximum) {
  if (requested != 0) return requested;
  return std::max<size_t>(
      1, std::min<size_t>(maximum,
                          std::max(1u, std::thread::hardware_concurrency())));
}

auth::Principal ApiPrincipal(size_t index) {
  auth::Principal principal;
  principal.tenant_id = "default";
  principal.subject = "api-key";
  principal.key_id = "configured-" + std::to_string(index);
  principal.scopes = {"inference.invoke", "models.read"};
  return principal;
}

bool IsSha256Hex(std::string_view value) {
  return value.size() == 64 &&
         value.find_first_not_of("0123456789abcdefABCDEF") ==
             std::string_view::npos;
}

}  // namespace

struct HttpServer::Impl {
  Engine* engine;
  HttpServerConfig config;
  std::unique_ptr<folly::CPUThreadPoolExecutor> blocking_executor;
  std::unique_ptr<transport::IoRuntime> runtime;
  std::shared_ptr<auth::ApiKeyStore> key_store;
  std::shared_ptr<auth::ApiKeyAuthenticator> authenticator;
  std::shared_ptr<middleware::BearerRouteGuard> route_guard;
  model_registry::Registry models;
  handlers::HealthState health;
  request::RequestManager request_manager;
  admission::AdmissionController admission;
  std::unique_ptr<EngineSchedulerBackend> scheduler_backend;
  std::shared_ptr<scheduler_client::RemoteSchedulerTransport>
      scheduler_transport;
  std::unique_ptr<scheduler_client::SchedulerClient> scheduler;
  std::unique_ptr<request::ManagedRequestService> requests;
  tokenization::InProcessTokenizationService tokenization;
  EngineMetricsSource metrics;
  std::shared_ptr<transport::Routes> routes;
  std::unique_ptr<transport::BeastListener> listener;

  mutable std::mutex state_mutex;
  std::condition_variable state_cv;
  bool listening = false;
  bool listen_finished = false;
  bool stopping = false;

  Impl(Engine* engine_value, HttpServerConfig config_value,
       std::unique_ptr<folly::CPUThreadPoolExecutor> blocking_value,
       std::unique_ptr<transport::IoRuntime> runtime_value)
      : engine(engine_value),
        config(std::move(config_value)),
        blocking_executor(std::move(blocking_value)),
        runtime(std::move(runtime_value)),
        key_store(std::make_shared<auth::ApiKeyStore>()),
        admission({.max_active_requests =
                       static_cast<uint32_t>(config.max_active_requests),
                   .max_reserved_tokens = std::numeric_limits<uint64_t>::max(),
                   .max_active_per_tenant =
                       static_cast<uint32_t>(config.max_active_requests)}),
        tokenization(&engine->tokenizer(), engine->model_name() + "@loaded"),
        metrics(engine) {
    if (config.scheduler_endpoint.empty()) {
      scheduler_backend = std::make_unique<EngineSchedulerBackend>(engine);
      scheduler = std::make_unique<scheduler_client::InProcessSchedulerClient>(
          scheduler_backend.get(), blocking_executor.get());
    } else {
#if defined(INFERX_WITH_GRPC_SCHEDULER)
      scheduler_transport =
          std::make_shared<scheduler_client::GrpcSchedulerTransport>(
              grpc::CreateChannel(config.scheduler_endpoint,
                                  grpc::InsecureChannelCredentials()));
      scheduler = std::make_unique<scheduler_client::RemoteSchedulerClient>(
          scheduler_transport, blocking_executor.get());
#endif
    }
    if (scheduler != nullptr) {
      requests = std::make_unique<request::ManagedRequestService>(
          &request_manager, scheduler.get(), &admission);
    }
  }

  ~Impl() {
    Stop();
    runtime->Join();
    blocking_executor->stop();
    blocking_executor->join();
  }

  Status Initialize() {
    if (scheduler == nullptr) {
      return FailedPreconditionError(
          "scheduler_endpoint requires a build with protobuf and gRPC");
    }
    std::vector<auth::ApiKeyRecord> records;
    records.reserve(config.api_key_sha256.size());
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
        models.Register({.id = engine->model_name(),
                         .version = "loaded",
                         .created = created,
                         .state = model_registry::ModelState::kReady,
                         .supports_generation = true,
                         .supports_embeddings = false}));

    health.SetConfigurationLoaded(true);
    health.SetDependenciesLoaded(true);
    health.SetSchedulerConnected(true);
    health.SetReadyModelAvailable(true);

    INFERX_ASSIGN_OR_RETURN(
        routes, handlers::BuildApiRoutes(
                    {.health = &health,
                     .models = &models,
                     .tokenization = &tokenization,
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
      if (stopping) return FailedPreconditionError("HTTP server is stopping");
      if (listening || listen_finished) {
        return FailedPreconditionError("HTTP server may only listen once");
      }
    }
    Status status = runtime->Start();
    if (status.ok()) status = listener->Start();
    {
      std::lock_guard lock(state_mutex);
      listening = status.ok();
      if (!status.ok()) listen_finished = true;
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
      listen_finished = true;
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
    if (listener) listener->Stop();
    if (runtime) runtime->Stop();
    state_cv.notify_all();
  }
};

HttpServer::HttpServer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

HttpServer::~HttpServer() = default;

StatusOr<std::unique_ptr<HttpServer>> HttpServer::Create(
    Engine* engine, const HttpServerConfig& config) {
  if (engine == nullptr) return InvalidArgumentError("engine is null");
  if (config.port < 0 || config.port > 65535) {
    return InvalidArgumentError("port must be in [0, 65535]");
  }
  if (config.max_request_bytes == 0) {
    return InvalidArgumentError("max_request_bytes must be positive");
  }
  if (config.max_active_requests == 0 ||
      config.max_active_requests > UINT32_MAX) {
    return InvalidArgumentError(
        "max_active_requests must be in [1, UINT32_MAX]");
  }
  if (config.read_timeout_seconds <= 0 || config.write_timeout_seconds <= 0 ||
      config.request_timeout_seconds <= 0) {
    return InvalidArgumentError("HTTP timeouts must be positive");
  }
  for (const std::string& hash : config.api_key_sha256) {
    if (!IsSha256Hex(hash)) {
      return InvalidArgumentError(
          "api_key_sha256 entries must contain exactly 64 hexadecimal "
          "characters");
    }
  }

  const size_t application_threads =
      DefaultThreads(config.application_threads, 32);
  auto blocking =
      std::make_unique<folly::CPUThreadPoolExecutor>(application_threads);
  INFERX_ASSIGN_OR_RETURN(
      auto runtime, transport::IoRuntime::Create(
                        {.io_shards = DefaultThreads(config.io_threads, 8),
                         .threads_per_shard = 1,
                         .coroutine_threads = application_threads}));
  auto impl = std::make_unique<Impl>(engine, config, std::move(blocking),
                                     std::move(runtime));
  INFERX_RETURN_IF_ERROR(impl->Initialize());
  return std::unique_ptr<HttpServer>(new HttpServer(std::move(impl)));
}

Status HttpServer::Listen() { return impl_->Listen(); }

void HttpServer::Stop() {
  if (impl_) impl_->Stop();
}

bool HttpServer::WaitUntilReady() {
  std::unique_lock lock(impl_->state_mutex);
  impl_->state_cv.wait(lock, [this] {
    return impl_->listening || impl_->listen_finished || impl_->stopping;
  });
  return impl_->listening && !impl_->stopping;
}

int HttpServer::port() const {
  return impl_->listener == nullptr ? 0 : impl_->listener->port();
}

}  // namespace inferx::server
