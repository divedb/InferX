#include "inferx/server/serve.h"

#include <algorithm>
#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "inferx/core/status_util.h"
#include "inferx/server/engine_client.h"
#include "inferx/server/engine_gateway.h"
#include "inferx/server/http_server.h"
#include "inferx/server/metrics.h"
#include "inferx/server/openai_handlers.h"
#include "inferx/server/request_model.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {
namespace {

namespace net = boost::asio;

/// \brief Pull-adapter from the gateway's push EventQueue onto
/// IEngineEventStream (the doc's in-process transport): events keep
/// arrival order and gain strict sequence numbers; TakeAll() closing
/// (cancel/shutdown) surfaces as a stream error, waking any pending
/// Next().
class GatewayEventStream : public IEngineEventStream {
 public:
  explicit GatewayEventStream(std::shared_ptr<EventQueue> events)
      : events_(std::move(events)) {}

  net::awaitable<StatusOr<EngineEvent>> Next() override {
    for (;;) {
      while (index_ < pending_.size()) {
        co_return MapEvent(pending_[index_++]);
      }
      auto batch = co_await events_->TakeAll();
      if (!batch.has_value()) {
        co_return absl::UnavailableError("engine stream closed");
      }
      pending_ = std::move(*batch);
      index_ = 0;
    }
  }

  void Close() override {}  // The gateway closes queues via Cancel.

 private:
  EngineEvent MapEvent(const CompletionEvent& event) {
    EngineEvent out;
    out.seq = seq_++;
    switch (event.kind) {
      case CompletionEvent::Kind::kDelta:
        out.kind = EngineEvent::Kind::kDelta;
        out.text = event.delta;
        break;
      case CompletionEvent::Kind::kFinish:
        out.kind = EngineEvent::Kind::kFinished;
        out.reason = event.reason;
        out.prompt_tokens = event.prompt_tokens;
        out.completion_tokens = event.completion_tokens;
        break;
      case CompletionEvent::Kind::kError:
        out.kind = EngineEvent::Kind::kError;
        out.admission = event.admission;
        out.error = absl::InternalError(event.message);
        break;
    }
    return out;
  }

  std::shared_ptr<EventQueue> events_;
  std::vector<CompletionEvent> pending_;
  std::size_t index_ = 0;
  std::uint64_t seq_ = 0;
};

/// \brief The production in-process engine client (the doc's
/// LocalEngineClient role): submits tokenized completions to the gateway
/// and aborts by engine id.
class GatewayEngineClient : public IEngineClient {
 public:
  explicit GatewayEngineClient(EngineGateway& gateway) : gateway_(gateway) {}

  StatusOr<EngineSubmission> Submit(EngineCompletion request) override {
    CompletionSpec spec;
    spec.prompt = std::move(request.prompt);
    spec.params = std::move(request.params);
    StatusOr<SubmitResult> submitted = gateway_.Submit(std::move(spec));
    if (!submitted.ok()) return submitted.status();
    EngineSubmission out;
    out.request_id = std::to_string(submitted->id);
    out.events = std::make_unique<GatewayEventStream>(submitted->events);
    return out;
  }

  void Abort(const std::string& request_id) override {
    gateway_.Cancel(std::stoull(request_id));
  }

 private:
  EngineGateway& gateway_;
};

std::vector<std::string> ResolveApiKeys(const FrontendConfig& frontend) {
  std::vector<std::string> keys = frontend.api_keys;
  if (const char* env = std::getenv("INFERX_API_KEY")) {
    if (env[0] != '\0') keys.push_back(env);
  }
  return keys;
}

}  // namespace

void MergeGenerationConfig(const GenerationConfig& gen, sampling::SamplingParams* sampling,
                           const std::set<std::string>& skip) {
  if (gen.temperature && !skip.count("temperature")) {
    sampling->temperature = *gen.temperature;
  }
  if (gen.top_p && !skip.count("top-p")) sampling->top_p = *gen.top_p;
  if (gen.min_p) sampling->min_p = *gen.min_p;
  if (gen.top_k) sampling->top_k = *gen.top_k > 0 ? static_cast<std::uint32_t>(*gen.top_k) : 0;
  if (gen.repetition_penalty) {
    sampling->repetition_penalty = *gen.repetition_penalty;
  }
  if (gen.max_new_tokens && !skip.count("max-tokens") && *gen.max_new_tokens > 0) {
    sampling->max_tokens = static_cast<std::uint32_t>(*gen.max_new_tokens);
  }
}

Status EngineConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(cache.Validate());
  return OkStatus();
}

Status RunServe(const ServeParams& params) {
  return Guarded([&]() -> Status {
    INFERX_RETURN_IF_ERROR(params.engine.Validate());
    const std::string tokenizer_path =
        params.engine.model.ResolvedTokenizerDir() + "/tokenizer.json";
    // Decode side only: the engine thread owns this instance exclusively;
    // prompt encoding lives in the pool's worker threads.
    auto decode_tokenizer = Take(Tokenizer::FromFile(tokenizer_path));

    net::io_context io;
    // The pool's workers load their tokenizer instances in parallel while
    // the gateway loads weights below; the startup gate then verifies both
    // finished before any request is accepted.
    PromptPoolConfig pool_config = params.frontend.tokenizer;
    pool_config.tokenizer_path = tokenizer_path;
    INFERX_RETURN_IF_ERROR(pool_config.Validate());
    std::shared_ptr<PromptProcessorPool> pool = CreatePromptProcessorPool(io, pool_config);

    const unsigned hardware = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads_count = std::clamp(hardware, 2u, 4u);
    std::vector<std::thread> workers;
    workers.reserve(threads_count - 1);
    for (unsigned i = 1; i < threads_count; ++i) {
      workers.emplace_back([&io] { io.run(); });
    }

    try {
      EngineGateway gateway(io, params.engine.model, params.engine.cache,
                            params.engine.scheduler, params.engine.execution, decode_tokenizer,
                            params.engine.parallel);
      GatewayEngineClient engine_client(gateway);
      MetricsService metrics;
      auto draining = std::make_shared<std::atomic<bool>>(false);

      HandlerConfig handler_config;
      handler_config.model_name = params.engine.model.ServedName();
      handler_config.created = static_cast<std::int64_t>(std::time(nullptr));
      handler_config.default_sampling = params.frontend.default_sampling;
      handler_config.api_keys = ResolveApiKeys(params.frontend);
      OpenAiHandler handler(
          std::move(handler_config), *pool, engine_client, metrics,
          [draining] { return draining->load(); },
          /*engine_ready=*/[draining] { return !draining->load(); });

      // Startup gate: refuse to serve rather than fall back to encoding on
      // an I/O thread (docs/tokenizer_design.md).
      const Status ready = pool->WaitUntilReady(pool_config.startup_timeout);
      if (!ready.ok()) throw std::runtime_error(std::string(ready.message()));

      HttpServer server(io, params.frontend.host,
                        static_cast<std::uint16_t>(params.frontend.port), handler);
      server.Listen();

      // SIGINT/SIGTERM: stop accepting, reject new preprocessing, let
      // admitted work drain under one deadline, then stop io. The deadline
      // is armed before draining and no joining call runs on an io callback.
      const std::chrono::milliseconds grace_ms = pool_config.shutdown_grace;
      net::signal_set signals(io, SIGINT, SIGTERM);
      auto grace = std::make_shared<net::steady_timer>(io);
      signals.async_wait([&server, &gateway, pool, &io, grace, grace_ms, draining](
                             boost::system::error_code ec, int) {
        if (ec) return;
        const auto deadline = std::chrono::steady_clock::now() + grace_ms;
        draining->store(true);
        server.SetDraining(true);        // New non-health requests get 503.
        server.Stop();                   // Stop accepting new connections.
        pool->BeginDrain();              // Rejects new work; drains queued jobs.
        gateway.RequestStop();           // Engine drains live requests, then exits.
        grace->expires_after(grace_ms);  // Hard backstop (stuck GPU call).
        grace->async_wait([&io, grace](boost::system::error_code timer_ec) {
          if (!timer_ec) io.stop();
        });
        net::co_spawn(
            io,
            [&gateway, pool, &io, deadline]() -> net::awaitable<void> {
              co_await gateway.WaitDrained();
              co_await pool->JoinUntil(deadline);
              io.stop();
            },
            net::detached);
      });

      io.run();
      for (std::thread& worker : workers) worker.join();
      gateway.Shutdown();  // Idempotent; joins the engine thread.
      return OkStatus();
    } catch (...) {
      io.stop();
      for (std::thread& worker : workers) worker.join();
      throw;  // Unwinds: pool/server destructors stop and join the workers.
    }
  });
}

}  // namespace inferx
