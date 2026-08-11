#include "inferx/server/handlers/vllm_compat_handler.h"

#include <string_view>

#include "inferx/api/openai.h"
#include "inferx/server/api/error_mapping.h"

// Injected by the build; the fallback keeps non-CMake tooling compiling.
#ifndef INFERX_VERSION
#define INFERX_VERSION "0.0.0"
#endif

namespace inferx::server::handlers {
namespace {

folly::coro::Task<void> Write(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    unsigned status, std::string body, folly::CancellationToken cancellation) {
  transport::HttpResponse response{
      static_cast<boost::beast::http::status>(status), request.version()};
  response.keep_alive(request.keep_alive());
  response.set(boost::beast::http::field::content_type, "application/json");
  response.body() = std::move(body);
  response.prepare_payload();
  (void)co_await writer.WriteResponse(std::move(response), cancellation);
}

folly::coro::Task<void> WriteError(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    const Status& status, std::string_view param,
    folly::CancellationToken cancellation) {
  auto error = api::MapStatus(status, param);
  co_await Write(request, writer, error.status, error.Json({}), cancellation);
}

}  // namespace

folly::coro::Task<void> VersionHandler::Handle(
    transport::HttpRequest request, transport::RequestContext,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  co_await Write(request, response, 200,
                 "{\"version\":\"" INFERX_VERSION "\"}", cancellation);
}

folly::coro::Task<void> PingHandler::Handle(
    transport::HttpRequest request, transport::RequestContext,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  co_await Write(request, response, 200, "{}", cancellation);
}

folly::coro::Task<void> NotSupportedHandler::Handle(
    transport::HttpRequest request, transport::RequestContext,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  const std::string message =
      std::string(request.target().data(), request.target().size()) +
      " is not supported by InferX (" + feature_ + ")";
  co_await Write(request, response, 501,
                 ::inferx::api::ErrorJson(message, "not_supported_error"),
                 cancellation);
}

folly::coro::Task<void> VllmTokenizeHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  if (!context.authenticated) {
    co_await WriteError(request, response,
                        absl::UnauthenticatedError(
                            "authenticated request context is required"),
                        {}, cancellation);
    co_return;
  }
  auto parsed = ::inferx::api::ParseVllmTokenizeRequest(request.body());
  if (!parsed.ok()) {
    co_await WriteError(request, response, parsed.status(), {}, cancellation);
    co_return;
  }
  if (registry_ == nullptr || tokenization_ == nullptr) {
    co_await WriteError(request, response,
                        InternalError("tokenization service is not configured"),
                        {}, cancellation);
    co_return;
  }
  auto model = registry_->Resolve(parsed->model, context.tenant_id);
  if (!model.ok()) {
    co_await WriteError(request, response, model.status(), "model",
                        cancellation);
    co_return;
  }
  const ::inferx::server::request::ModelVersion version =
      model->id + "@" + model->version;
  auto tokenized = tokenization_->TokenizeCompletion(
      version, parsed->prompt, parsed->add_special_tokens);
  if (!tokenized.ok()) {
    co_await WriteError(request, response, tokenized.status(), "prompt",
                        cancellation);
    co_return;
  }
  // vLLM's /tokenize is diagnostic; it reports rather than enforces the
  // context limit, so no limit check here unlike /v1/tokenize.
  co_await Write(request, response, 200,
                 ::inferx::api::VllmTokenizeJson(tokenized->token_ids),
                 cancellation);
}

folly::coro::Task<void> DetokenizeHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  if (!context.authenticated) {
    co_await WriteError(request, response,
                        absl::UnauthenticatedError(
                            "authenticated request context is required"),
                        {}, cancellation);
    co_return;
  }
  auto parsed = ::inferx::api::ParseDetokenizeRequest(request.body());
  if (!parsed.ok()) {
    co_await WriteError(request, response, parsed.status(), {}, cancellation);
    co_return;
  }
  if (registry_ == nullptr || tokenization_ == nullptr) {
    co_await WriteError(request, response,
                        InternalError("tokenization service is not configured"),
                        {}, cancellation);
    co_return;
  }
  auto model = registry_->Resolve(parsed->model, context.tenant_id);
  if (!model.ok()) {
    co_await WriteError(request, response, model.status(), "model",
                        cancellation);
    co_return;
  }
  const ::inferx::server::request::ModelVersion version =
      model->id + "@" + model->version;
  auto prompt = tokenization_->Detokenize(version, parsed->tokens);
  if (!prompt.ok()) {
    co_await WriteError(request, response, prompt.status(), "tokens",
                        cancellation);
    co_return;
  }
  co_await Write(request, response, 200,
                 ::inferx::api::DetokenizeJson(*prompt), cancellation);
}

}  // namespace inferx::server::handlers
