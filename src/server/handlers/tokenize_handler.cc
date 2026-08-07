#include "inferx/server/handlers/tokenize_handler.h"

#include "inferx/api/openai.h"
#include "inferx/server/api/error_mapping.h"

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

folly::coro::Task<void> TokenizeHandler::Handle(
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
  auto parsed = ::inferx::api::ParseTokenizeRequest(request.body());
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
      version, parsed->text, parsed->add_special_tokens);
  if (!tokenized.ok()) {
    co_await WriteError(request, response, tokenized.status(), "text",
                        cancellation);
    co_return;
  }
  if (model->context_limit != 0 &&
      tokenized->prompt_tokens > model->context_limit) {
    co_await WriteError(request, response,
                        InvalidArgumentError("text exceeds model context limit"),
                        "text", cancellation);
    co_return;
  }
  co_await Write(request, response, 200,
                 ::inferx::api::TokenizeJson(parsed->model,
                                             tokenized->token_ids),
                 cancellation);
}

}  // namespace inferx::server::handlers
