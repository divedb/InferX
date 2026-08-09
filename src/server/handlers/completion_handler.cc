#include "inferx/server/handlers/completion_handler.h"

#include "inferx/api/openai.h"
#include "inferx/server/api/error_mapping.h"
#include "inferx/server/request/request_id.h"
#include "inferx/server/transport/sse_writer.h"
#include "inferx/support/log.h"

namespace inferx::server::handlers {
namespace {

folly::coro::Task<void> WriteError(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    const Status& status, std::string_view param,
    folly::CancellationToken cancellation) {
  auto error = api::MapStatus(status, param);
  transport::HttpResponse response{
      static_cast<boost::beast::http::status>(error.status), request.version()};
  response.keep_alive(request.keep_alive());
  response.set(boost::beast::http::field::content_type, "application/json");
  response.body() = error.Json({});
  response.prepare_payload();
  (void)co_await writer.WriteResponse(std::move(response), cancellation);
}

::inferx::api::FinishReason ApiFinish(
    scheduler_client::FinishReason reason) {
  return reason == scheduler_client::FinishReason::kStop
             ? ::inferx::api::FinishReason::kStop
             : ::inferx::api::FinishReason::kLength;
}

const char* FinishName(scheduler_client::FinishReason reason) {
  return reason == scheduler_client::FinishReason::kStop ? "stop" : "length";
}

}  // namespace

folly::coro::Task<void> CompletionHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  const auto request_started = std::chrono::steady_clock::now();
  if (!context.authenticated) {
    co_await WriteError(request, response,
                        absl::UnauthenticatedError(
                            "authenticated request context is required"),
                        {}, cancellation);
    co_return;
  }
  auto parsed = ::inferx::api::ParseCompletionRequest(request.body());
  if (!parsed.ok()) {
    co_await WriteError(request, response, parsed.status(), {}, cancellation);
    co_return;
  }
  if (registry_ == nullptr || tokenization_ == nullptr || requests_ == nullptr) {
    co_await WriteError(request, response,
                        InternalError("generation services are not configured"),
                        {}, cancellation);
    co_return;
  }
  auto model = registry_->Resolve(parsed->model, context.tenant_id);
  if (!model.ok()) {
    co_await WriteError(request, response, model.status(), "model",
                        cancellation);
    co_return;
  }
  if (!model->supports_generation) {
    co_await WriteError(request, response,
                        UnimplementedError(
                            "model does not support generation workloads"),
                        "model", cancellation);
    co_return;
  }
  const ::inferx::server::request::ModelVersion version =
      model->id + "@" + model->version;
  auto tokenized = tokenization_->TokenizeCompletion(version, parsed->prompt);
  if (!tokenized.ok()) {
    co_await WriteError(request, response, tokenized.status(), "prompt",
                        cancellation);
    co_return;
  }
  if (model->context_limit != 0 &&
      tokenized->prompt_tokens + parsed->sampling.max_tokens >
          model->context_limit) {
    co_await WriteError(request, response,
                        InvalidArgumentError(
                            "prompt and max_tokens exceed model context limit"),
                        "max_tokens", cancellation);
    co_return;
  }

  scheduler_client::ScheduledRequest command;
  command.request_id = ::inferx::server::request::GenerateRequestId();
  command.tenant_id = context.tenant_id;
  command.model_version = version;
  command.tokenizer_revision = model->tokenizer_revision;
  command.workload = scheduler_client::WorkloadClass::kGeneration;
  command.prompt_tokens = std::move(tokenized->token_ids);
  command.sampling.max_tokens = parsed->sampling.max_tokens;
  command.sampling.temperature = parsed->sampling.temperature;
  command.sampling.top_p = parsed->sampling.top_p;
  command.sampling.seed = parsed->sampling.seed;
  command.sampling.stop = parsed->sampling.stop;
  command.deadline = std::chrono::steady_clock::now() + timeout_;
  auto submitted = co_await requests_->Submit(std::move(command), cancellation);
  if (!submitted.ok()) {
    co_await WriteError(request, response, submitted.status(), {}, cancellation);
    co_return;
  }
  const auto queued_at = std::chrono::steady_clock::now();

  const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
  const bool sampled = parsed->sampling.temperature > 0;
  auto events = requests_->Events(submitted->request_id, cancellation);
  std::string collected;
  scheduler_client::Usage usage;
  scheduler_client::FinishReason finish = scheduler_client::FinishReason::kNone;
  std::optional<std::chrono::steady_clock::time_point> first_token_at;
  transport::SseWriter sse(&response);
  if (parsed->sampling.stream) {
    Status started = co_await sse.Start(request.version(), request.keep_alive(),
                                        submitted->request_id, cancellation);
    if (!started.ok()) {
      (void)co_await requests_->Cancel(
          submitted->request_id,
          ::inferx::server::request::CancellationReason::kClientDisconnected);
      co_return;
    }
  }
  while (auto next = co_await events.next()) {
    auto event = std::move(*next);
    if (!first_token_at.has_value() &&
        (!event.text_delta.empty() || !event.token_ids.empty() ||
         event.terminal)) {
      first_token_at = std::chrono::steady_clock::now();
    }
    if (!event.error.ok()) {
      if (!parsed->sampling.stream) {
        co_await WriteError(request, response, event.error, {}, cancellation);
      } else {
        auto error = api::MapStatus(event.error);
        (void)co_await sse.Event(error.Json(submitted->request_id), cancellation);
        (void)co_await sse.Finish(cancellation);
      }
      co_return;
    }
    usage = event.usage;
    if (parsed->sampling.stream && !event.text_delta.empty()) {
      const Status written = co_await sse.Event(
          ::inferx::api::CompletionChunkJson(
              submitted->request_id, parsed->model, event.text_delta, nullptr,
              created, sampled),
          cancellation);
      if (!written.ok()) {
        (void)co_await requests_->Cancel(
            submitted->request_id,
            ::inferx::server::request::CancellationReason::kClientDisconnected);
        co_return;
      }
    } else {
      collected += event.text_delta;
    }
    if (event.terminal) {
      finish = event.finish_reason;
      break;
    }
  }
  if (finish == scheduler_client::FinishReason::kNone) {
    if (!parsed->sampling.stream) {
      co_await WriteError(request, response,
                          InternalError("generation stream ended without a "
                                        "terminal event"),
                          {}, cancellation);
    } else {
      (void)co_await sse.Finish(cancellation);
    }
    co_return;
  }
  const auto api_finish = ApiFinish(finish);
  const ::inferx::api::Usage api_usage{
      static_cast<int32_t>(usage.prompt_tokens),
      static_cast<int32_t>(usage.completion_tokens)};
  const auto finished_at = std::chrono::steady_clock::now();
  const auto first = first_token_at.value_or(finished_at);
  LOG(INFO) << Rid(submitted->request_id)
            << "access route=/v1/completions model=" << parsed->model
            << " status=200 prompt_tokens=" << usage.prompt_tokens
            << " completion_tokens=" << usage.completion_tokens
            << " queue_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   first - queued_at).count()
            << " ttft_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   first - request_started).count()
            << " latency_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   finished_at - request_started).count()
            << " finish_reason=" << FinishName(finish);
  if (parsed->sampling.stream) {
    (void)co_await sse.Event(
        ::inferx::api::CompletionChunkJson(
            submitted->request_id, parsed->model, {}, &api_finish, created,
            sampled),
        cancellation);
    if (parsed->sampling.include_usage) {
      (void)co_await sse.Event(
          ::inferx::api::UsageChunkJson(submitted->request_id, parsed->model,
                                        api_usage, created, false, sampled),
          cancellation);
    }
    (void)co_await sse.Event("[DONE]", cancellation);
    (void)co_await sse.Finish(cancellation);
    co_return;
  }
  transport::HttpResponse result{boost::beast::http::status::ok,
                                 request.version()};
  result.keep_alive(request.keep_alive());
  result.set(boost::beast::http::field::content_type, "application/json");
  result.body() = ::inferx::api::CompletionJson(
      submitted->request_id, parsed->model, collected, api_finish, api_usage,
      created, sampled);
  result.prepare_payload();
  (void)co_await response.WriteResponse(std::move(result), cancellation);
}

}  // namespace inferx::server::handlers
